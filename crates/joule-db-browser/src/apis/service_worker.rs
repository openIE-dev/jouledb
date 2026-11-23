//! Service Worker support for background database operations

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::{JsCast, prelude::*};
#[cfg(target_arch = "wasm32")]
use web_sys::ServiceWorkerRegistration;

/// Service Worker manager for background operations
pub struct ServiceWorkerManager {
    #[cfg(target_arch = "wasm32")]
    registration: Option<ServiceWorkerRegistration>,
}

impl ServiceWorkerManager {
    /// Register service worker
    #[cfg(target_arch = "wasm32")]
    pub async fn register(script_url: &str) -> Result<Self, String> {
        use wasm_bindgen_futures::JsFuture;
        use web_sys::window;

        let window = window().ok_or("No window")?;
        let navigator = window.navigator();
        // Use dynamic access for service_worker
        let service_worker = js_sys::Reflect::get(&navigator, &JsValue::from_str("serviceWorker"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or("Service Worker not supported")?;

        // Use dynamic method call for register
        let register_func = js_sys::Reflect::get(&service_worker, &JsValue::from_str("register"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or("register method not available")?;

        let registration_promise = register_func
            .call1(&service_worker, &JsValue::from_str(script_url))
            .map_err(|e| format!("Registration failed: {:?}", e))?;

        let registration_promise: js_sys::Promise = registration_promise
            .dyn_into()
            .map_err(|e| format!("Not a promise: {:?}", e))?;

        let registration = JsFuture::from(registration_promise)
            .await
            .map_err(|e| format!("Registration await failed: {:?}", e))?;

        let registration: ServiceWorkerRegistration = registration.into();

        Ok(Self {
            registration: Some(registration),
        })
    }

    /// Register service worker (non-WASM: Service Workers only exist in browsers)
    /// Returns a no-op manager that reports no active worker.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn register(_script_url: &str) -> Result<Self, String> {
        tracing::debug!("ServiceWorkerManager: Service Workers not available outside browser");
        Ok(Self {})
    }

    /// Get active service worker
    #[cfg(target_arch = "wasm32")]
    pub fn active_worker(&self) -> Option<js_sys::Object> {
        self.registration
            .as_ref()
            .and_then(|r: &web_sys::ServiceWorkerRegistration| r.active())
            .and_then(|sw: web_sys::ServiceWorker| sw.dyn_into::<js_sys::Object>().ok())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn active_worker(&self) -> Option<()> {
        None
    }
}
