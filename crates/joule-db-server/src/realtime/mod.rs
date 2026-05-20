//! Real-Time Features Module
//!
//! Provides change streams, triggers, and webhooks for real-time event-driven architectures.

pub mod changestream;
pub mod triggers;

pub use changestream::{
    ChangeStream, ChangeStreamFilter, ChangeStreamManager, ChangeStreamOptions, ChangeStreamToken,
};
pub use triggers::{Trigger, TriggerContext, TriggerEventType, TriggerManager, TriggerResult};
