//! Push Manager
//!
//! Manages Push API for push notifications

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Push Manager
///
/// Manages Push API for push notifications

pub struct PushManager {
    push_manager: Arc<RwLock<Option<js_sys::Object>>>,
    subscription: Arc<RwLock<Option<js_sys::Object>>>,
}

impl PushManager {
    pub fn new() -> Self {
        Self {
            push_manager: Arc::new(RwLock::new(None)),
            subscription: Arc::new(RwLock::new(None)),
        }
    }

    /// Initialize push manager

    #[cfg(target_arch = "wasm32")]
    pub async fn initialize(&self) -> Result<(), JsValue> {
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
        let reg_obj = registration
            .dyn_into::<js_sys::Object>()
            .map_err(|_| JsValue::from_str("Invalid registration object"))?;

        let push_manager = js_sys::Reflect::get(&reg_obj, &JsValue::from_str("pushManager"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("pushManager not found"))?;

        *self.push_manager.write().unwrap_or_else(|e| e.into_inner()) = Some(push_manager);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn initialize(&self) -> Result<(), JsValue> {
        Err(JsValue::from_str("Push API not available outside browser"))
    }

    /// Subscribe to push notifications with public key

    #[cfg(target_arch = "wasm32")]
    pub async fn subscribe(
        &self,
        public_key: &str,
        user_visible_only: bool,
    ) -> Result<js_sys::Object, JsValue> {
        let push_manager = self
            .push_manager
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| JsValue::from_str("Push manager not initialized"))?;

        let subscribe_func = js_sys::Reflect::get(&push_manager, &JsValue::from_str("subscribe"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("subscribe not found"))?;

        let options = js_sys::Object::new();
        js_sys::Reflect::set(
            &options,
            &JsValue::from_str("userVisibleOnly"),
            &JsValue::from_bool(user_visible_only),
        )
        .ok();

        let application_server_key = js_sys::Uint8Array::from(public_key.as_bytes());
        js_sys::Reflect::set(
            &options,
            &JsValue::from_str("applicationServerKey"),
            &application_server_key,
        )
        .ok();

        let promise = subscribe_func
            .call1(&push_manager, &options)
            .map_err(|_| JsValue::from_str("Failed to subscribe"))?;

        let subscription = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        let subscription_obj = subscription
            .dyn_into::<js_sys::Object>()
            .map_err(|_| JsValue::from_str("Invalid subscription object"))?;
        *self.subscription.write().unwrap_or_else(|e| e.into_inner()) =
            Some(subscription_obj.clone());

        Ok(subscription_obj)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn subscribe(
        &self,
        _public_key: &str,
        _user_visible_only: bool,
    ) -> Result<js_sys::Object, JsValue> {
        Err(JsValue::from_str("Push API not available outside browser"))
    }

    /// Get existing subscription

    #[cfg(target_arch = "wasm32")]
    pub async fn get_subscription(&self) -> Result<Option<js_sys::Object>, JsValue> {
        let push_manager = self
            .push_manager
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
            .ok_or_else(|| JsValue::from_str("Push manager not initialized"))?;

        let get_subscription_func =
            js_sys::Reflect::get(&push_manager, &JsValue::from_str("getSubscription"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("getSubscription not found"))?;

        let promise = get_subscription_func
            .call0(&push_manager)
            .map_err(|_| JsValue::from_str("Failed to get subscription"))?;

        let result = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        if result.is_null() || result.is_undefined() {
            Ok(None)
        } else {
            Ok(Some(result.dyn_into::<js_sys::Object>().map_err(|_| {
                JsValue::from_str("Invalid subscription object")
            })?))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn get_subscription(&self) -> Result<Option<js_sys::Object>, JsValue> {
        Ok(None)
    }

    /// Unsubscribe from push notifications

    #[cfg(target_arch = "wasm32")]
    pub async fn unsubscribe(&self) -> Result<bool, JsValue> {
        if let Some(subscription) = self
            .subscription
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
        {
            let unsubscribe_func =
                js_sys::Reflect::get(&subscription, &JsValue::from_str("unsubscribe"))
                    .ok()
                    .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                    .ok_or_else(|| JsValue::from_str("unsubscribe not found"))?;

            let promise = unsubscribe_func
                .call0(&subscription)
                .map_err(|_| JsValue::from_str("Failed to unsubscribe"))?;

            let result = wasm_bindgen_futures::JsFuture::from(
                promise
                    .dyn_into::<js_sys::Promise>()
                    .map_err(|_| JsValue::from_str("Invalid promise"))?,
            )
            .await?;

            *self.subscription.write().unwrap_or_else(|e| e.into_inner()) = None;
            Ok(result.as_bool().unwrap_or(false))
        } else {
            Ok(false)
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn unsubscribe(&self) -> Result<bool, JsValue> {
        Ok(false)
    }

    /// Request permission for notifications

    #[cfg(target_arch = "wasm32")]
    pub async fn request_permission(&self) -> Result<String, JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;

        let notification = js_sys::Reflect::get(&window, &JsValue::from_str("Notification"))
            .ok()
            .and_then(|v| v.dyn_into::<js_sys::Object>().ok())
            .ok_or_else(|| JsValue::from_str("Notification API not supported"))?;

        let request_permission_func =
            js_sys::Reflect::get(&notification, &JsValue::from_str("requestPermission"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("requestPermission not found"))?;

        let promise = request_permission_func
            .call0(&notification)
            .map_err(|_| JsValue::from_str("Failed to request permission"))?;

        let result = wasm_bindgen_futures::JsFuture::from(
            promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| JsValue::from_str("Invalid promise"))?,
        )
        .await?;

        Ok(result.as_string().unwrap_or_else(|| "default".to_string()))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn request_permission(&self) -> Result<String, JsValue> {
        Ok("denied".to_string())
    }
}

impl Default for PushManager {
    fn default() -> Self {
        Self::new()
    }
}

unsafe impl Send for PushManager {}
unsafe impl Sync for PushManager {}
