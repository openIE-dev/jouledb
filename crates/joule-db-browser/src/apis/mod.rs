//! Browser API Integration for JouleDB
//!
//! Provides browser-specific API integrations:
//! - Service Worker management
//! - Background Sync API
//! - Cache API
//! - Web Locks API
//! - Push notifications
//! - Storage management
//! - Beacon API
//! - Broadcast Channel API

mod background_sync;
mod beacon;
mod broadcast_channel;
mod cache;
mod push;
mod service_worker;
mod storage;
mod web_locks;

pub use background_sync::{BackgroundSyncManager, PeriodicBackgroundSyncManager};
pub use beacon::BeaconManager;
pub use broadcast_channel::BroadcastChannelManager;
pub use cache::CacheManager;
pub use push::PushManager;
pub use service_worker::ServiceWorkerManager;
pub use storage::NavigatorStorageManager;
pub use web_locks::WebLocksManager;
