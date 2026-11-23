//! Background Sync Manager
//!
//! Manages Background Sync API for offline operations

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Background Sync Manager
///
/// Manages Background Sync API for offline operations

pub struct BackgroundSyncManager {
    sync_registration: Arc<RwLock<Option<js_sys::Object>>>,
}

impl BackgroundSyncManager {
    pub fn new() -> Self {
        Self {
            sync_registration: Arc::new(RwLock::new(None)),
        }
    }

    /// Register sync tag

    #[cfg(target_arch = "wasm32")]
    pub async fn register_sync(&self, tag: &str) -> Result<(), JsValue> {
        let navigator = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window"))?
            .navigator();

        let service_worker = js_sys::Reflect::get(&navigator, &JsValue::from_str("serviceWorker"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("Service Worker not supported"))?;

        let ready = js_sys::Reflect::get(&service_worker, &JsValue::from_str("ready"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Promise>().ok())
            .ok_or_else(|| JsValue::from_str("ready not found"))?;

        let registration = wasm_bindgen_futures::JsFuture::from(ready).await?;
        let reg_obj = registration.dyn_into::<js_sys::Object>().unwrap();

        let sync = js_sys::Reflect::get(&reg_obj, &JsValue::from_str("sync"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("sync not found"))?;

        let register_func = js_sys::Reflect::get(&sync, &JsValue::from_str("register"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("register not found"))?;

        let promise = register_func
            .call1(&sync, &JsValue::from_str(tag))
            .map_err(|_| JsValue::from_str("Failed to register sync"))?;

        wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        *self.sync_registration.write().unwrap() = Some(sync);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn register_sync(&self, _tag: &str) -> Result<(), JsValue> {
        Err(JsValue::from_str(
            "Background Sync not available outside browser",
        ))
    }
}

impl Default for BackgroundSyncManager {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for BackgroundSyncManager {}
unsafe impl Sync for BackgroundSyncManager {}

/// Periodic Background Sync Manager
///
/// Manages Periodic Background Sync API for periodic background tasks

pub struct PeriodicBackgroundSyncManager {
    sync_registration: Arc<RwLock<Option<js_sys::Object>>>,
}

impl PeriodicBackgroundSyncManager {
    pub fn new() -> Self {
        Self {
            sync_registration: Arc::new(RwLock::new(None)),
        }
    }

    /// Register periodic sync with minimum interval in milliseconds

    #[cfg(target_arch = "wasm32")]
    pub async fn register(&self, tag: &str, min_interval_ms: u32) -> Result<(), JsValue> {
        let navigator = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window"))?
            .navigator();

        let service_worker = js_sys::Reflect::get(&navigator, &JsValue::from_str("serviceWorker"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("Service Worker not supported"))?;

        let ready = js_sys::Reflect::get(&service_worker, &JsValue::from_str("ready"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Promise>().ok())
            .ok_or_else(|| JsValue::from_str("ready not found"))?;

        let registration = wasm_bindgen_futures::JsFuture::from(ready).await?;
        let reg_obj = registration.dyn_into::<js_sys::Object>().unwrap();

        let periodic_sync = js_sys::Reflect::get(&reg_obj, &JsValue::from_str("periodicSync"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("periodicSync not found"))?;

        let register_func = js_sys::Reflect::get(&periodic_sync, &JsValue::from_str("register"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("register not found"))?;

        let options = js_sys::Object::new();
        js_sys::Reflect::set(
            &options,
            &JsValue::from_str("minInterval"),
            &JsValue::from_f64(min_interval_ms as f64),
        )
        .ok();

        let promise = register_func
            .call2(&periodic_sync, &JsValue::from_str(tag), &options)
            .map_err(|_| JsValue::from_str("Failed to register periodic sync"))?;

        wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        *self.sync_registration.write().unwrap() = Some(periodic_sync);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn register(&self, _tag: &str, _min_interval_ms: u32) -> Result<(), JsValue> {
        Err(JsValue::from_str(
            "Periodic Background Sync not available outside browser",
        ))
    }

    /// Unregister periodic sync

    #[cfg(target_arch = "wasm32")]
    pub async fn unregister(&self, tag: &str) -> Result<(), JsValue> {
        if let Some(periodic_sync) = self.sync_registration.read().unwrap().clone() {
            let unregister_func =
                js_sys::Reflect::get(&periodic_sync, &JsValue::from_str("unregister"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| JsValue::from_str("unregister not found"))?;

            let promise = unregister_func
                .call1(&periodic_sync, &JsValue::from_str(tag))
                .map_err(|_| JsValue::from_str("Failed to unregister periodic sync"))?;

            wasm_bindgen_futures::JsFuture::from(
                promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| JsValue::from_str("Invalid promise"))?,
            )
            .await?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn unregister(&self, _tag: &str) -> Result<(), JsValue> {
        Ok(())
    }

    /// Get tags for all registered periodic syncs

    #[cfg(target_arch = "wasm32")]
    pub async fn get_tags(&self) -> Result<js_sys::Array, JsValue> {
        if let Some(periodic_sync) = self.sync_registration.read().unwrap().clone() {
            let get_tags_func = js_sys::Reflect::get(&periodic_sync, &JsValue::from_str("getTags"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("getTags not found"))?;

            let promise = get_tags_func
                .call0(&periodic_sync)
                .map_err(|_| JsValue::from_str("Failed to get tags"))?;

            let result = wasm_bindgen_futures::JsFuture::from(
                promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| JsValue::from_str("Invalid promise"))?,
            )
            .await?;

            Ok(result
                .dyn_into::<js_sys::Array>()
                .unwrap_or_else(|_| js_sys::Array::new()))
        } else {
            Ok(js_sys::Array::new())
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn get_tags(&self) -> Result<js_sys::Array, JsValue> {
        Ok(js_sys::Array::new())
    }
}

impl Default for PeriodicBackgroundSyncManager {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for PeriodicBackgroundSyncManager {}
unsafe impl Sync for PeriodicBackgroundSyncManager {}
