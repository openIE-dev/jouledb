//! Service Worker Replication
//!
//! Background replication using Service Workers

use wasm_bindgen::prelude::*;

/// Replication configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReplicationConfig {
    /// Enable replication
    pub enabled: bool,
    /// Service Worker script path
    pub worker_script: String,
    /// Replication interval (ms)
    pub interval: u32,
}

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            worker_script: "/sw.js".to_string(),
            interval: 1000,
        }
    }
}

/// Service Worker replication

pub struct ServiceWorkerReplication {
    config: ReplicationConfig,
    #[cfg(target_arch = "wasm32")]
    worker: Option<JsValue>, // ServiceWorker as JsValue since web_sys::ServiceWorker may not be available
    #[cfg(not(target_arch = "wasm32"))]
    worker: Option<()>,
}

impl ServiceWorkerReplication {
    /// Create new replication manager

    pub async fn new(config: JsValue) -> Result<ServiceWorkerReplication, JsValue> {
        let config: ReplicationConfig = serde_wasm_bindgen::from_value(config)
            .map_err(|e| JsValue::from_str(&format!("Invalid config: {:?}", e)))?;

        // Register service worker
        let worker = if config.enabled {
            Self::register_service_worker(&config.worker_script).await?
        } else {
            None
        };

        Ok(Self { config, worker })
    }

    /// Register service worker
    #[cfg(target_arch = "wasm32")]
    async fn register_service_worker(script_path: &str) -> Result<Option<JsValue>, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window object"))?;

        let navigator = window.navigator();
        // Use dynamic access to service_worker since web_sys::ServiceWorker may not be available
        let service_worker_obj =
            js_sys::Reflect::get(&navigator, &JsValue::from_str("serviceWorker"))
                .ok()
                .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
                .ok_or_else(|| JsValue::from_str("ServiceWorker not available"))?;

        let service_worker = service_worker_obj;

        let register_func = js_sys::Reflect::get(&service_worker, &JsValue::from_str("register"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("register not available"))?;

        let register_promise = register_func
            .call1(&service_worker, &JsValue::from_str(script_path))
            .map_err(|_| JsValue::from_str("Service worker registration failed"))?;

        let registration = wasm_bindgen_futures::JsFuture::from(
            register_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await
        .map_err(|_| JsValue::from_str("Service worker registration failed"))?;

        // Get active worker
        let ready_func = js_sys::Reflect::get(&registration, &JsValue::from_str("ready"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("ready not available"))?;

        let ready_promise = ready_func
            .call0(&registration)
            .map_err(|_| JsValue::from_str("Service worker ready failed"))?;

        let ready_registration = wasm_bindgen_futures::JsFuture::from(
            ready_promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await
        .map_err(|_| JsValue::from_str("Service worker ready failed"))?;

        let active = js_sys::Reflect::get(&ready_registration, &JsValue::from_str("active"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok());

        Ok(active)
    }

    #[cfg(not(target_arch = "wasm32"))]
    async fn register_service_worker(_script_path: &str) -> Result<Option<()>, JsValue> {
        Ok(None)
    }

    /// Start replication
    #[cfg(target_arch = "wasm32")]
    pub fn start(&self) -> Result<(), JsValue> {
        if let Some(ref worker) = self.worker {
            // Send start message to service worker
            let post_message = js_sys::Reflect::get(worker, &JsValue::from_str("postMessage"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("postMessage not available"))?;

            // Create message object
            let message = js_sys::Object::new();
            js_sys::Reflect::set(
                &message,
                &JsValue::from_str("type"),
                &JsValue::from_str("START_REPLICATION"),
            )?;
            js_sys::Reflect::set(
                &message,
                &JsValue::from_str("interval"),
                &JsValue::from_f64(self.config.interval as f64),
            )?;

            post_message.call1(worker, &message)?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn start(&self) -> Result<(), JsValue> {
        // No-op on non-WASM platforms
        Ok(())
    }

    /// Stop replication
    #[cfg(target_arch = "wasm32")]
    pub fn stop(&self) -> Result<(), JsValue> {
        if let Some(ref worker) = self.worker {
            // Send stop message to service worker
            let post_message = js_sys::Reflect::get(worker, &JsValue::from_str("postMessage"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("postMessage not available"))?;

            // Create message object
            let message = js_sys::Object::new();
            js_sys::Reflect::set(
                &message,
                &JsValue::from_str("type"),
                &JsValue::from_str("STOP_REPLICATION"),
            )?;

            post_message.call1(worker, &message)?;
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn stop(&self) -> Result<(), JsValue> {
        // No-op on non-WASM platforms
        Ok(())
    }

    /// Check if replication is active
    pub fn is_active(&self) -> bool {
        #[cfg(target_arch = "wasm32")]
        {
            self.worker.is_some()
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            false
        }
    }

    /// Get configuration
    pub fn config(&self) -> &ReplicationConfig {
        &self.config
    }
}
