//! Cache API Manager
//!
//! Manages Cache API for HTTP response caching

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Cache API Manager
///
/// Manages Cache API for HTTP response caching

#[wasm_bindgen]
pub struct CacheManager {
    cache_name: String,
    cache: Arc<RwLock<Option<js_sys::Object>>>,
}

#[wasm_bindgen]
impl CacheManager {
    pub fn new(cache_name: String) -> Self {
        Self {
            cache_name,
            cache: Arc::new(RwLock::new(None)),
        }
    }

    /// Open cache

    #[cfg(target_arch = "wasm32")]
    pub async fn open(&self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;

        let caches = js_sys::Reflect::get(&window, &JsValue::from_str("caches"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("Cache API not supported"))?;

        let open_func = js_sys::Reflect::get(&caches, &JsValue::from_str("open"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("open not found"))?;

        let promise = open_func
            .call1(&caches, &JsValue::from_str(&self.cache_name))
            .map_err(|_| JsValue::from_str("Failed to open cache"))?;

        let cache = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        *self.cache.write().unwrap() = Some(cache.dyn_into::<js_sys::Object>().unwrap());
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn open(&self) -> Result<(), JsValue> {
        Err(JsValue::from_str("Cache API not available outside browser"))
    }

    /// Add to cache

    #[cfg(target_arch = "wasm32")]
    pub async fn add(&self, url: &str) -> Result<(), JsValue> {
        if let Some(cache) = self.cache.read().unwrap().clone() {
            let add_func = js_sys::Reflect::get(&cache, &JsValue::from_str("add"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("add not found"))?;

            let promise = add_func
                .call1(&cache, &JsValue::from_str(url))
                .map_err(|_| JsValue::from_str("Failed to add to cache"))?;

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
    pub async fn add(&self, _url: &str) -> Result<(), JsValue> {
        Ok(())
    }

    /// Match from cache
    #[wasm_bindgen(js_name = "match")]
    #[cfg(target_arch = "wasm32")]
    pub async fn cache_match(&self, url: &str) -> Result<Option<js_sys::Object>, JsValue> {
        if let Some(cache) = self.cache.read().unwrap().clone() {
            let match_func = js_sys::Reflect::get(&cache, &JsValue::from_str("match"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("match not found"))?;

            let promise = match_func
                .call1(&cache, &JsValue::from_str(url))
                .map_err(|_| JsValue::from_str("Failed to match cache"))?;

            let result = wasm_bindgen_futures::JsFuture::from(
                promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| JsValue::from_str("Invalid promise"))?,
            )
            .await?;

            if result.is_null() || result.is_undefined() {
                Ok(None)
            } else {
                Ok(Some(result.dyn_into::<js_sys::Object>().unwrap()))
            }
        } else {
            Ok(None)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn cache_match(&self, _url: &str) -> Result<Option<js_sys::Object>, JsValue> {
        Ok(None)
    }

    /// Delete cache

    #[cfg(target_arch = "wasm32")]
    pub async fn delete(&self) -> Result<bool, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;

        let caches = js_sys::Reflect::get(&window, &JsValue::from_str("caches"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("Cache API not supported"))?;

        let delete_func = js_sys::Reflect::get(&caches, &JsValue::from_str("delete"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("delete not found"))?;

        let promise = delete_func
            .call1(&caches, &JsValue::from_str(&self.cache_name))
            .map_err(|_| JsValue::from_str("Failed to delete cache"))?;

        let result = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        *self.cache.write().unwrap() = None;
        Ok(result.as_bool().unwrap_or(false))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn delete(&self) -> Result<bool, JsValue> {
        Ok(false)
    }
}

unsafe impl Send for CacheManager {}
unsafe impl Sync for CacheManager {}
