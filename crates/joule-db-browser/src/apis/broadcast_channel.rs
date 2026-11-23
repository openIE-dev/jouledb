//! Broadcast Channel Manager
//!
//! Manages Broadcast Channel API for cross-tab communication

use std::sync::{Arc, RwLock};
use wasm_bindgen::prelude::*;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::JsCast;

/// Broadcast Channel Manager
///
/// Manages Broadcast Channel API for cross-tab communication

pub struct BroadcastChannelManager {
    channel: Arc<RwLock<Option<js_sys::Object>>>,
    channel_name: String,
    message_handlers: Arc<RwLock<Vec<js_sys::Function>>>,
}

impl BroadcastChannelManager {
    pub fn new(channel_name: String) -> Self {
        Self {
            channel: Arc::new(RwLock::new(None)),
            channel_name,
            message_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create and open broadcast channel

    #[cfg(target_arch = "wasm32")]
    pub fn open(&self) -> Result<(), JsValue> {
        let window = web_sys::window().ok_or_else(|| JsValue::from_str("No window"))?;

        let broadcast_channel_constructor =
            js_sys::Reflect::get(&window, &JsValue::from_str("BroadcastChannel"))
                .ok()
                .and_then(|v| v.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("BroadcastChannel not supported"))?;

        let channel = js_sys::Reflect::construct(
            &broadcast_channel_constructor,
            &js_sys::Array::of1(&JsValue::from_str(&self.channel_name)),
        )
        .map_err(|_| JsValue::from_str("Failed to create BroadcastChannel"))?;

        *self.channel.write().unwrap() = Some(channel.dyn_into::<js_sys::Object>().unwrap());
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn open(&self) -> Result<(), JsValue> {
        // In non-browser, just mark as "open" for testing
        Ok(())
    }

    /// Post message to channel

    #[cfg(target_arch = "wasm32")]
    pub fn post_message(&self, message: &str) -> Result<(), JsValue> {
        let channel = self
            .channel
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| JsValue::from_str("Channel not opened"))?;

        let post_message_func = js_sys::Reflect::get(&channel, &JsValue::from_str("postMessage"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("postMessage not found"))?;

        post_message_func
            .call1(&channel, &JsValue::from_str(message))
            .map_err(|_| JsValue::from_str("Failed to post message"))?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn post_message(&self, _message: &str) -> Result<(), JsValue> {
        Ok(())
    }

    /// Post object message to channel

    #[cfg(target_arch = "wasm32")]
    pub fn post_message_object(&self, message: js_sys::Object) -> Result<(), JsValue> {
        let channel = self
            .channel
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| JsValue::from_str("Channel not opened"))?;

        let post_message_func = js_sys::Reflect::get(&channel, &JsValue::from_str("postMessage"))
            .ok()
            .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
            .ok_or_else(|| JsValue::from_str("postMessage not found"))?;

        post_message_func
            .call1(&channel, &message)
            .map_err(|_| JsValue::from_str("Failed to post message"))?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn post_message_object(&self, _message: js_sys::Object) -> Result<(), JsValue> {
        Ok(())
    }

    /// Add message event listener

    #[cfg(target_arch = "wasm32")]
    pub fn add_message_listener(&self, callback: js_sys::Function) -> Result<(), JsValue> {
        let channel = self
            .channel
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| JsValue::from_str("Channel not opened"))?;

        let add_event_listener_func =
            js_sys::Reflect::get(&channel, &JsValue::from_str("addEventListener"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("addEventListener not found"))?;

        add_event_listener_func
            .call2(&channel, &JsValue::from_str("message"), &callback)
            .map_err(|_| JsValue::from_str("Failed to add event listener"))?;

        self.message_handlers.write().unwrap().push(callback);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn add_message_listener(&self, callback: js_sys::Function) -> Result<(), JsValue> {
        self.message_handlers.write().unwrap().push(callback);
        Ok(())
    }

    /// Remove message event listener

    #[cfg(target_arch = "wasm32")]
    pub fn remove_message_listener(&self, callback: js_sys::Function) -> Result<(), JsValue> {
        let channel = self
            .channel
            .read()
            .unwrap()
            .clone()
            .ok_or_else(|| JsValue::from_str("Channel not opened"))?;

        let remove_event_listener_func =
            js_sys::Reflect::get(&channel, &JsValue::from_str("removeEventListener"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("removeEventListener not found"))?;

        remove_event_listener_func
            .call2(&channel, &JsValue::from_str("message"), &callback)
            .map_err(|_| JsValue::from_str("Failed to remove event listener"))?;

        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn remove_message_listener(&self, _callback: js_sys::Function) -> Result<(), JsValue> {
        Ok(())
    }

    /// Close broadcast channel

    #[cfg(target_arch = "wasm32")]
    pub fn close(&self) -> Result<(), JsValue> {
        if let Some(channel) = self.channel.read().unwrap().clone() {
            let close_func = js_sys::Reflect::get(&channel, &JsValue::from_str("close"))
                .ok()
                .and_then(|f| f.dyn_into::<js_sys::Function>().ok())
                .ok_or_else(|| JsValue::from_str("close not found"))?;

            close_func
                .call0(&channel)
                .map_err(|_| JsValue::from_str("Failed to close channel"))?;

            *self.channel.write().unwrap() = None;
            self.message_handlers.write().unwrap().clear();
        }
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn close(&self) -> Result<(), JsValue> {
        *self.channel.write().unwrap() = None;
        self.message_handlers.write().unwrap().clear();
        Ok(())
    }

    /// Get channel name

    pub fn get_name(&self) -> String {
        self.channel_name.clone()
    }

    /// Check if channel is open

    pub fn is_open(&self) -> bool {
        self.channel.read().unwrap().is_some()
    }
}

unsafe impl Send for BroadcastChannelManager {}
unsafe impl Sync for BroadcastChannelManager {}
