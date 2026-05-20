//! BroadcastChannel — cross-context message broadcasting.
//!
//! Headless model of the BroadcastChannel API. A `BroadcastHub` manages
//! named channels; posting a message delivers it to all subscribers except
//! the sender.

use std::collections::{HashMap, VecDeque};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BroadcastMessage {
    pub channel: String,
    pub data: serde_json::Value,
    pub sender_id: Uuid,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug)]
pub struct BroadcastChannel {
    pub name: String,
    pub id: Uuid,
    pub inbox: VecDeque<BroadcastMessage>,
    pub closed: bool,
}

impl BroadcastChannel {
    fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: Uuid::new_v4(),
            inbox: VecDeque::new(),
            closed: false,
        }
    }
}

// ── Hub ─────────────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct BroadcastHub {
    /// channel_name -> list of subscriber UUIDs
    channels: HashMap<String, Vec<Uuid>>,
    /// subscriber_id -> pending messages
    pending: HashMap<Uuid, VecDeque<BroadcastMessage>>,
}

impl BroadcastHub {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn create_channel(&mut self, name: impl Into<String>) -> BroadcastChannel {
        let ch = BroadcastChannel::new(name);
        self.channels
            .entry(ch.name.clone())
            .or_default()
            .push(ch.id);
        self.pending.insert(ch.id, VecDeque::new());
        ch
    }

    /// Post a message to all subscribers of `channel_name` except `sender_id`.
    pub fn post(&mut self, channel_name: &str, sender_id: Uuid, data: serde_json::Value) {
        let now = Utc::now();
        if let Some(subscribers) = self.channels.get(channel_name) {
            let msg = BroadcastMessage {
                channel: channel_name.to_string(),
                data,
                sender_id,
                timestamp: now,
            };
            for sub in subscribers {
                if *sub != sender_id {
                    if let Some(queue) = self.pending.get_mut(sub) {
                        queue.push_back(msg.clone());
                    }
                }
            }
        }
    }

    /// Drain all pending messages for the given channel subscriber.
    pub fn receive(&mut self, channel_id: Uuid) -> Vec<BroadcastMessage> {
        self.pending
            .get_mut(&channel_id)
            .map(|q| q.drain(..).collect())
            .unwrap_or_default()
    }

    /// Close and remove a channel subscriber.
    pub fn close_channel(&mut self, id: Uuid) {
        self.pending.remove(&id);
        for subs in self.channels.values_mut() {
            subs.retain(|s| *s != id);
        }
    }

    pub fn subscriber_count(&self, name: &str) -> usize {
        self.channels.get(name).map(|v| v.len()).unwrap_or(0)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn post_delivers_to_others_not_self() {
        let mut hub = BroadcastHub::new();
        let ch_a = hub.create_channel("news");
        let ch_b = hub.create_channel("news");

        hub.post("news", ch_a.id, json!("hello"));

        let msgs_a = hub.receive(ch_a.id);
        let msgs_b = hub.receive(ch_b.id);
        assert!(msgs_a.is_empty(), "sender should not receive own message");
        assert_eq!(msgs_b.len(), 1);
        assert_eq!(msgs_b[0].data, json!("hello"));
    }

    #[test]
    fn multiple_subscribers() {
        let mut hub = BroadcastHub::new();
        let ch_a = hub.create_channel("ch");
        let ch_b = hub.create_channel("ch");
        let ch_c = hub.create_channel("ch");

        hub.post("ch", ch_a.id, json!(42));

        assert_eq!(hub.receive(ch_b.id).len(), 1);
        assert_eq!(hub.receive(ch_c.id).len(), 1);
        assert!(hub.receive(ch_a.id).is_empty());
    }

    #[test]
    fn close_removes_subscriber() {
        let mut hub = BroadcastHub::new();
        let ch_a = hub.create_channel("ch");
        let ch_b = hub.create_channel("ch");
        assert_eq!(hub.subscriber_count("ch"), 2);

        hub.close_channel(ch_b.id);
        assert_eq!(hub.subscriber_count("ch"), 1);

        // Post after close — only ch_a remains (and it's the sender, so no delivery).
        hub.post("ch", ch_a.id, json!("late"));
        assert!(hub.receive(ch_a.id).is_empty());
    }

    #[test]
    fn receive_drains() {
        let mut hub = BroadcastHub::new();
        let ch_a = hub.create_channel("ch");
        let ch_b = hub.create_channel("ch");

        hub.post("ch", ch_a.id, json!(1));
        hub.post("ch", ch_a.id, json!(2));

        let msgs = hub.receive(ch_b.id);
        assert_eq!(msgs.len(), 2);
        // Second receive returns empty.
        assert!(hub.receive(ch_b.id).is_empty());
    }

    #[test]
    fn different_channels_isolated() {
        let mut hub = BroadcastHub::new();
        let alpha = hub.create_channel("alpha");
        let beta = hub.create_channel("beta");

        hub.post("alpha", alpha.id, json!("a-msg"));
        assert!(hub.receive(beta.id).is_empty());
    }

    #[test]
    fn subscriber_count() {
        let mut hub = BroadcastHub::new();
        assert_eq!(hub.subscriber_count("ch"), 0);
        let _a = hub.create_channel("ch");
        assert_eq!(hub.subscriber_count("ch"), 1);
        let b = hub.create_channel("ch");
        assert_eq!(hub.subscriber_count("ch"), 2);
        hub.close_channel(b.id);
        assert_eq!(hub.subscriber_count("ch"), 1);
    }

    #[test]
    fn message_has_timestamp_and_sender() {
        let mut hub = BroadcastHub::new();
        let ch_a = hub.create_channel("ch");
        let ch_b = hub.create_channel("ch");

        hub.post("ch", ch_a.id, json!("data"));
        let msgs = hub.receive(ch_b.id);
        assert_eq!(msgs[0].sender_id, ch_a.id);
        assert_eq!(msgs[0].channel, "ch");
    }

    #[test]
    fn post_to_nonexistent_channel_is_noop() {
        let mut hub = BroadcastHub::new();
        // Should not panic.
        hub.post("nonexistent", Uuid::new_v4(), json!("nothing"));
    }
}
