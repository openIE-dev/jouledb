//! Beacon Manager
//!
//! Manages Beacon API for reliable data sending

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Beacon Manager
///
/// Manages Beacon API for reliable data sending

pub struct BeaconManager {
    queued_data: Arc<RwLock<Vec<String>>>,
}

impl BeaconManager {
    pub fn new() -> Self {
        Self {
            queued_data: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Send beacon data to URL

    #[cfg(target_arch = "wasm32")]
    pub fn send_beacon(&self, url: &str, data: &str) -> Result<bool, JsValue> {
        let navigator = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window"))?
            .navigator();

        let send_beacon_func = js_sys::Reflect::get(&navigator, &JsValue::from_str("sendBeacon"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("sendBeacon not supported"))?;

        let result = send_beacon_func
            .call2(
                &navigator,
                &JsValue::from_str(url),
                &JsValue::from_str(data),
            )
            .map_err(|_| JsValue::from_str("Failed to send beacon"))?;

        Ok(result.as_bool().unwrap_or(false))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn send_beacon(&self, _url: &str, _data: &str) -> Result<bool, JsValue> {
        // Queue data for later in non-browser environment
        Ok(false)
    }

    /// Send beacon with blob data

    #[cfg(target_arch = "wasm32")]
    pub fn send_beacon_blob(&self, url: &str, data: &[u8]) -> Result<bool, JsValue> {
        let navigator = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window"))?
            .navigator();

        let send_beacon_func = js_sys::Reflect::get(&navigator, &JsValue::from_str("sendBeacon"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("sendBeacon not supported"))?;

        // Create Blob
        let uint8_array = js_sys::Uint8Array::from(data);
        let array = js_sys::Array::new();
        array.push(&uint8_array);

        let blob_parts = array;
        let blob_options = js_sys::Object::new();
        js_sys::Reflect::set(
            &blob_options,
            &JsValue::from_str("type"),
            &JsValue::from_str("application/octet-stream"),
        )
        .ok();

        let blob_constructor =
            js_sys::Reflect::get(&web_sys::window().unwrap(), &JsValue::from_str("Blob"))
                .ok()
                .and_then(|v| v.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("Blob constructor not found"))?;

        let blob = js_sys::Reflect::construct(
            &blob_constructor,
            &js_sys::Array::of2(&blob_parts, &blob_options),
        )
        .map_err(|_| JsValue::from_str("Failed to create Blob"))?;

        let result = send_beacon_func
            .call2(&navigator, &JsValue::from_str(url), &blob)
            .map_err(|_| JsValue::from_str("Failed to send beacon"))?;

        Ok(result.as_bool().unwrap_or(false))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn send_beacon_blob(&self, _url: &str, _data: &[u8]) -> Result<bool, JsValue> {
        Ok(false)
    }

    /// Send beacon with FormData

    #[cfg(target_arch = "wasm32")]
    pub fn send_beacon_form(
        &self,
        url: &str,
        form_data_obj: js_sys::Object,
    ) -> Result<bool, JsValue> {
        let navigator = web_sys::window()
            .ok_or_else(|| JsValue::from_str("No window"))?
            .navigator();

        let send_beacon_func = js_sys::Reflect::get(&navigator, &JsValue::from_str("sendBeacon"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("sendBeacon not supported"))?;

        let result = send_beacon_func
            .call2(&navigator, &JsValue::from_str(url), &form_data_obj)
            .map_err(|_| JsValue::from_str("Failed to send beacon"))?;

        Ok(result.as_bool().unwrap_or(false))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn send_beacon_form(
        &self,
        _url: &str,
        _form_data_obj: js_sys::Object,
    ) -> Result<bool, JsValue> {
        Ok(false)
    }

    /// Queue data for later sending (fallback when beacon fails)

    pub fn queue_data(&self, data: String) {
        self.queued_data.write().unwrap().push(data);
    }

    /// Get queued data count

    pub fn queued_count(&self) -> usize {
        self.queued_data.read().unwrap().len()
    }

    /// Clear queued data

    pub fn clear_queue(&self) {
        self.queued_data.write().unwrap().clear();
    }
}

impl Default for BeaconManager {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for BeaconManager {}
unsafe impl Sync for BeaconManager {}
