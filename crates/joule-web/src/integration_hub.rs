//! Integration hub / message router — route definitions, content-based routing,
//! message transformation, error channel, dead letter, message enrichment,
//! split/aggregate patterns, and route statistics.
//!
//! Replaces Node.js integration libraries (Apache Camel JS, NestJS microservices)
//! with a pure-Rust message routing engine inspired by Enterprise Integration Patterns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Integration hub domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrationError {
    /// Route not found.
    RouteNotFound(String),
    /// Duplicate route ID.
    DuplicateRoute(String),
    /// No matching route for message.
    NoMatchingRoute { message_type: String },
    /// Transformation error.
    TransformError(String),
    /// Dead letter queue full.
    DeadLetterFull(usize),
    /// Aggregation group not found.
    AggregationGroupNotFound(String),
    /// Channel not found.
    ChannelNotFound(String),
    /// Message not found.
    MessageNotFound(String),
}

impl std::fmt::Display for IntegrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RouteNotFound(id) => write!(f, "route not found: {id}"),
            Self::DuplicateRoute(id) => write!(f, "duplicate route: {id}"),
            Self::NoMatchingRoute { message_type } => {
                write!(f, "no matching route for message type: {message_type}")
            }
            Self::TransformError(msg) => write!(f, "transform error: {msg}"),
            Self::DeadLetterFull(cap) => {
                write!(f, "dead letter queue full (capacity {cap})")
            }
            Self::AggregationGroupNotFound(id) => {
                write!(f, "aggregation group not found: {id}")
            }
            Self::ChannelNotFound(id) => write!(f, "channel not found: {id}"),
            Self::MessageNotFound(id) => write!(f, "message not found: {id}"),
        }
    }
}

impl std::error::Error for IntegrationError {}

// ── Enums ───────────────────────────────────────────────────────

/// Message status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageStatus {
    Pending,
    Routed,
    Transformed,
    Enriched,
    Delivered,
    Failed,
    DeadLettered,
    Split,
    Aggregated,
}

/// Routing strategy.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingStrategy {
    /// Route based on message type header.
    ContentBased,
    /// Route to all matching routes (multicast).
    Multicast,
    /// Route to one route in round-robin fashion.
    RoundRobin,
    /// Route based on a header value matching a key.
    HeaderBased { header_key: String },
}

/// Routing condition for content-based routing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RouteCondition {
    /// Match any message.
    Always,
    /// Match on message type.
    MessageType(String),
    /// Match when a header equals a value.
    HeaderEquals { key: String, value: String },
    /// Match when a header is present.
    HeaderPresent(String),
    /// Match when body contains a key.
    BodyContains(String),
    /// Match all sub-conditions.
    And(Vec<RouteCondition>),
    /// Match any sub-condition.
    Or(Vec<RouteCondition>),
}

/// Message transformation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransformAction {
    /// Set a header.
    SetHeader { key: String, value: String },
    /// Remove a header.
    RemoveHeader(String),
    /// Rename a body field.
    RenameField { from: String, to: String },
    /// Set a body field to a static value.
    SetField { key: String, value: String },
    /// Remove a body field.
    RemoveField(String),
    /// Wrap the body in an envelope with a key.
    WrapBody(String),
}

// ── Data Structures ─────────────────────────────────────────────

/// A message flowing through the integration hub.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub id: String,
    pub message_type: String,
    pub headers: HashMap<String, String>,
    pub body: Value,
    pub status: MessageStatus,
    pub created_at: DateTime<Utc>,
    pub correlation_id: Option<String>,
    pub source_route: Option<String>,
}

impl Message {
    pub fn new(id: impl Into<String>, message_type: impl Into<String>, body: Value) -> Self {
        Self {
            id: id.into(),
            message_type: message_type.into(),
            headers: HashMap::new(),
            body,
            status: MessageStatus::Pending,
            created_at: Utc::now(),
            correlation_id: None,
            source_route: None,
        }
    }

    pub fn with_header(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.headers.insert(key.into(), value.into());
        self
    }

    pub fn with_correlation(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }
}

/// A route definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteDef {
    pub id: String,
    pub name: String,
    pub condition: RouteCondition,
    pub transforms: Vec<TransformAction>,
    pub enrichments: Vec<EnrichmentDef>,
    pub destination: String,
    pub enabled: bool,
    pub error_channel: Option<String>,
}

/// Enrichment definition — adds data from a source to the message.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnrichmentDef {
    pub source_key: String,
    pub target_header: String,
    pub lookup_table: HashMap<String, String>,
}

/// Route statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RouteStats {
    pub messages_routed: u64,
    pub messages_failed: u64,
    pub messages_transformed: u64,
    pub last_message_at: Option<DateTime<Utc>>,
}

/// Dead letter entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeadLetterEntry {
    pub message: Message,
    pub error: String,
    pub route_id: Option<String>,
    pub dead_lettered_at: DateTime<Utc>,
}

/// Aggregation group.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationGroup {
    pub group_id: String,
    pub correlation_id: String,
    pub messages: Vec<Message>,
    pub expected_count: Option<usize>,
    pub created_at: DateTime<Utc>,
}

// ── Engine ──────────────────────────────────────────────────────

/// Integration hub / message router.
pub struct IntegrationHub {
    routes: Vec<RouteDef>,
    route_stats: HashMap<String, RouteStats>,
    dead_letter: Vec<DeadLetterEntry>,
    dead_letter_capacity: usize,
    aggregation_groups: HashMap<String, AggregationGroup>,
    processed_messages: Vec<Message>,
    routing_strategy: RoutingStrategy,
    round_robin_index: usize,
}

impl IntegrationHub {
    pub fn new() -> Self {
        Self {
            routes: Vec::new(),
            route_stats: HashMap::new(),
            dead_letter: Vec::new(),
            dead_letter_capacity: 1000,
            aggregation_groups: HashMap::new(),
            processed_messages: Vec::new(),
            routing_strategy: RoutingStrategy::ContentBased,
            round_robin_index: 0,
        }
    }

    pub fn with_capacity(mut self, dead_letter_cap: usize) -> Self {
        self.dead_letter_capacity = dead_letter_cap;
        self
    }

    pub fn with_strategy(mut self, strategy: RoutingStrategy) -> Self {
        self.routing_strategy = strategy;
        self
    }

    // ── Route Management ────────────────────────────────────────

    /// Add a route.
    pub fn add_route(&mut self, route: RouteDef) -> Result<(), IntegrationError> {
        if self.routes.iter().any(|r| r.id == route.id) {
            return Err(IntegrationError::DuplicateRoute(route.id.clone()));
        }
        self.route_stats
            .insert(route.id.clone(), RouteStats::default());
        self.routes.push(route);
        Ok(())
    }

    /// Remove a route.
    pub fn remove_route(&mut self, route_id: &str) -> Result<RouteDef, IntegrationError> {
        let idx = self
            .routes
            .iter()
            .position(|r| r.id == route_id)
            .ok_or_else(|| IntegrationError::RouteNotFound(route_id.to_string()))?;
        self.route_stats.remove(route_id);
        Ok(self.routes.remove(idx))
    }

    /// Get a route by ID.
    pub fn get_route(&self, id: &str) -> Option<&RouteDef> {
        self.routes.iter().find(|r| r.id == id)
    }

    /// List all route IDs.
    pub fn list_routes(&self) -> Vec<&str> {
        self.routes.iter().map(|r| r.id.as_str()).collect()
    }

    // ── Message Routing ─────────────────────────────────────────

    /// Route a message through the hub.
    pub fn route(&mut self, mut message: Message) -> Result<Vec<Message>, IntegrationError> {
        let matching_routes = self.find_matching_routes(&message);

        if matching_routes.is_empty() {
            let msg_type = message.message_type.clone();
            self.to_dead_letter(message, "no matching route", None)?;
            return Err(IntegrationError::NoMatchingRoute {
                message_type: msg_type,
            });
        }

        let selected = match &self.routing_strategy {
            RoutingStrategy::ContentBased => matching_routes,
            RoutingStrategy::Multicast => matching_routes,
            RoutingStrategy::RoundRobin => {
                if matching_routes.is_empty() {
                    return Ok(Vec::new());
                }
                let idx = self.round_robin_index % matching_routes.len();
                self.round_robin_index += 1;
                vec![matching_routes[idx].clone()]
            }
            RoutingStrategy::HeaderBased { header_key } => {
                let hval = message.headers.get(header_key).cloned().unwrap_or_default();
                matching_routes
                    .into_iter()
                    .filter(|r| r.destination == hval)
                    .collect()
            }
        };

        let mut results = Vec::new();

        for route in &selected {
            let mut routed_msg = message.clone();
            routed_msg.source_route = Some(route.id.clone());
            routed_msg.status = MessageStatus::Routed;

            // Apply transforms.
            for transform in &route.transforms {
                apply_transform(&mut routed_msg, transform)?;
                routed_msg.status = MessageStatus::Transformed;
            }

            // Apply enrichments.
            for enrichment in &route.enrichments {
                apply_enrichment(&mut routed_msg, enrichment);
                routed_msg.status = MessageStatus::Enriched;
            }

            routed_msg.status = MessageStatus::Delivered;

            // Update stats.
            if let Some(stats) = self.route_stats.get_mut(&route.id) {
                stats.messages_routed += 1;
                if !route.transforms.is_empty() {
                    stats.messages_transformed += 1;
                }
                stats.last_message_at = Some(Utc::now());
            }

            results.push(routed_msg);
        }

        // Store for querying.
        for r in &results {
            self.processed_messages.push(r.clone());
        }

        Ok(results)
    }

    /// Route a message and send failures to error channel.
    pub fn route_with_error_handling(
        &mut self,
        message: Message,
    ) -> Vec<Result<Message, IntegrationError>> {
        let matching = self.find_matching_routes(&message);
        let mut results = Vec::new();

        for route in &matching {
            let mut routed_msg = message.clone();
            routed_msg.source_route = Some(route.id.clone());

            let mut failed = false;
            for transform in &route.transforms {
                if let Err(e) = apply_transform(&mut routed_msg, transform) {
                    if let Some(stats) = self.route_stats.get_mut(&route.id) {
                        stats.messages_failed += 1;
                    }
                    let _ = self.to_dead_letter(
                        routed_msg.clone(),
                        &e.to_string(),
                        Some(route.id.clone()),
                    );
                    results.push(Err(e));
                    failed = true;
                    break;
                }
            }

            if !failed {
                routed_msg.status = MessageStatus::Delivered;
                if let Some(stats) = self.route_stats.get_mut(&route.id) {
                    stats.messages_routed += 1;
                    stats.last_message_at = Some(Utc::now());
                }
                self.processed_messages.push(routed_msg.clone());
                results.push(Ok(routed_msg));
            }
        }
        results
    }

    fn find_matching_routes(&self, message: &Message) -> Vec<RouteDef> {
        self.routes
            .iter()
            .filter(|r| r.enabled && evaluate_route_condition(&r.condition, message))
            .cloned()
            .collect()
    }

    // ── Split / Aggregate ───────────────────────────────────────

    /// Split a message with an array body into individual messages.
    pub fn split(&self, message: &Message, array_field: &str) -> Vec<Message> {
        let items = match &message.body {
            Value::Object(map) => {
                if let Some(Value::Array(arr)) = map.get(array_field) {
                    arr.clone()
                } else {
                    return vec![];
                }
            }
            Value::Array(arr) => arr.clone(),
            _ => return vec![],
        };

        items
            .into_iter()
            .enumerate()
            .map(|(i, item)| {
                let mut msg = message.clone();
                msg.id = format!("{}-split-{i}", message.id);
                msg.body = item;
                msg.status = MessageStatus::Split;
                msg.correlation_id = Some(message.id.clone());
                msg
            })
            .collect()
    }

    /// Start an aggregation group.
    pub fn start_aggregation(
        &mut self,
        group_id: impl Into<String>,
        correlation_id: impl Into<String>,
        expected_count: Option<usize>,
    ) -> String {
        let gid = group_id.into();
        self.aggregation_groups.insert(
            gid.clone(),
            AggregationGroup {
                group_id: gid.clone(),
                correlation_id: correlation_id.into(),
                messages: Vec::new(),
                expected_count,
                created_at: Utc::now(),
            },
        );
        gid
    }

    /// Add a message to an aggregation group.
    pub fn aggregate(
        &mut self,
        group_id: &str,
        message: Message,
    ) -> Result<bool, IntegrationError> {
        let group = self
            .aggregation_groups
            .get_mut(group_id)
            .ok_or_else(|| IntegrationError::AggregationGroupNotFound(group_id.to_string()))?;
        group.messages.push(message);
        if let Some(expected) = group.expected_count {
            Ok(group.messages.len() >= expected)
        } else {
            Ok(false)
        }
    }

    /// Complete an aggregation group and merge messages into one.
    pub fn complete_aggregation(
        &mut self,
        group_id: &str,
    ) -> Result<Message, IntegrationError> {
        let group = self
            .aggregation_groups
            .remove(group_id)
            .ok_or_else(|| IntegrationError::AggregationGroupNotFound(group_id.to_string()))?;

        let bodies: Vec<Value> = group.messages.iter().map(|m| m.body.clone()).collect();
        let mut aggregated = Message::new(
            format!("{group_id}-aggregated"),
            "aggregated",
            Value::Array(bodies),
        );
        aggregated.status = MessageStatus::Aggregated;
        aggregated.correlation_id = Some(group.correlation_id);
        Ok(aggregated)
    }

    /// Get aggregation group.
    pub fn get_aggregation_group(&self, group_id: &str) -> Option<&AggregationGroup> {
        self.aggregation_groups.get(group_id)
    }

    // ── Dead Letter ─────────────────────────────────────────────

    fn to_dead_letter(
        &mut self,
        mut message: Message,
        error: &str,
        route_id: Option<String>,
    ) -> Result<(), IntegrationError> {
        if self.dead_letter.len() >= self.dead_letter_capacity {
            return Err(IntegrationError::DeadLetterFull(self.dead_letter_capacity));
        }
        message.status = MessageStatus::DeadLettered;
        self.dead_letter.push(DeadLetterEntry {
            message,
            error: error.to_string(),
            route_id,
            dead_lettered_at: Utc::now(),
        });
        Ok(())
    }

    /// Get dead letter entries.
    pub fn dead_letter_queue(&self) -> &[DeadLetterEntry] {
        &self.dead_letter
    }

    /// Drain dead letter queue.
    pub fn drain_dead_letter(&mut self) -> Vec<DeadLetterEntry> {
        std::mem::take(&mut self.dead_letter)
    }

    // ── Statistics ──────────────────────────────────────────────

    /// Get stats for a route.
    pub fn route_stats(&self, route_id: &str) -> Option<&RouteStats> {
        self.route_stats.get(route_id)
    }

    /// Get all route statistics.
    pub fn all_stats(&self) -> &HashMap<String, RouteStats> {
        &self.route_stats
    }

    /// Total messages processed.
    pub fn total_processed(&self) -> usize {
        self.processed_messages.len()
    }

    /// Total dead lettered.
    pub fn total_dead_lettered(&self) -> usize {
        self.dead_letter.len()
    }

    /// Get processed messages for a route.
    pub fn messages_for_route(&self, route_id: &str) -> Vec<&Message> {
        self.processed_messages
            .iter()
            .filter(|m| m.source_route.as_deref() == Some(route_id))
            .collect()
    }
}

impl Default for IntegrationHub {
    fn default() -> Self {
        Self::new()
    }
}

// ── Condition Evaluation ────────────────────────────────────────

fn evaluate_route_condition(condition: &RouteCondition, message: &Message) -> bool {
    match condition {
        RouteCondition::Always => true,
        RouteCondition::MessageType(mt) => message.message_type == *mt,
        RouteCondition::HeaderEquals { key, value } => {
            message.headers.get(key).map_or(false, |v| v == value)
        }
        RouteCondition::HeaderPresent(key) => message.headers.contains_key(key),
        RouteCondition::BodyContains(key) => {
            if let Value::Object(map) = &message.body {
                map.contains_key(key)
            } else {
                false
            }
        }
        RouteCondition::And(conditions) => {
            conditions.iter().all(|c| evaluate_route_condition(c, message))
        }
        RouteCondition::Or(conditions) => {
            conditions.iter().any(|c| evaluate_route_condition(c, message))
        }
    }
}

// ── Transform Application ───────────────────────────────────────

fn apply_transform(
    message: &mut Message,
    action: &TransformAction,
) -> Result<(), IntegrationError> {
    match action {
        TransformAction::SetHeader { key, value } => {
            message.headers.insert(key.clone(), value.clone());
        }
        TransformAction::RemoveHeader(key) => {
            message.headers.remove(key);
        }
        TransformAction::RenameField { from, to } => {
            if let Value::Object(map) = &mut message.body {
                if let Some(val) = map.remove(from) {
                    map.insert(to.clone(), val);
                }
            }
        }
        TransformAction::SetField { key, value } => {
            if let Value::Object(map) = &mut message.body {
                map.insert(key.clone(), Value::String(value.clone()));
            }
        }
        TransformAction::RemoveField(key) => {
            if let Value::Object(map) = &mut message.body {
                map.remove(key);
            }
        }
        TransformAction::WrapBody(key) => {
            let body = message.body.clone();
            let mut wrapper = serde_json::Map::new();
            wrapper.insert(key.clone(), body);
            message.body = Value::Object(wrapper);
        }
    }
    Ok(())
}

// ── Enrichment Application ──────────────────────────────────────

fn apply_enrichment(message: &mut Message, enrichment: &EnrichmentDef) {
    // Look up value from message body or headers.
    let lookup_value = if let Value::Object(map) = &message.body {
        map.get(&enrichment.source_key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
    } else {
        message.headers.get(&enrichment.source_key).cloned()
    };

    if let Some(val) = lookup_value {
        if let Some(enriched) = enrichment.lookup_table.get(&val) {
            message
                .headers
                .insert(enrichment.target_header.clone(), enriched.clone());
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_route(id: &str, condition: RouteCondition) -> RouteDef {
        RouteDef {
            id: id.to_string(),
            name: format!("Route {id}"),
            condition,
            transforms: vec![],
            enrichments: vec![],
            destination: format!("dest-{id}"),
            enabled: true,
            error_channel: None,
        }
    }

    fn make_message(id: &str, msg_type: &str) -> Message {
        Message::new(id, msg_type, serde_json::json!({"key": "value"}))
    }

    fn setup() -> IntegrationHub {
        let mut hub = IntegrationHub::new();
        hub.add_route(make_route("r1", RouteCondition::MessageType("order".into())))
            .unwrap();
        hub.add_route(make_route("r2", RouteCondition::MessageType("invoice".into())))
            .unwrap();
        hub
    }

    #[test]
    fn test_add_route() {
        let hub = setup();
        assert!(hub.get_route("r1").is_some());
        assert_eq!(hub.list_routes().len(), 2);
    }

    #[test]
    fn test_duplicate_route() {
        let mut hub = setup();
        let err = hub
            .add_route(make_route("r1", RouteCondition::Always))
            .unwrap_err();
        assert!(matches!(err, IntegrationError::DuplicateRoute(_)));
    }

    #[test]
    fn test_remove_route() {
        let mut hub = setup();
        hub.remove_route("r1").unwrap();
        assert!(hub.get_route("r1").is_none());
    }

    #[test]
    fn test_content_based_routing() {
        let mut hub = setup();
        let msg = make_message("m1", "order");
        let results = hub.route(msg).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source_route.as_deref(), Some("r1"));
    }

    #[test]
    fn test_no_matching_route() {
        let mut hub = setup();
        let msg = make_message("m1", "unknown");
        let err = hub.route(msg).unwrap_err();
        assert!(matches!(err, IntegrationError::NoMatchingRoute { .. }));
    }

    #[test]
    fn test_message_transformation() {
        let mut hub = IntegrationHub::new();
        let mut route = make_route("r1", RouteCondition::Always);
        route.transforms = vec![
            TransformAction::SetHeader {
                key: "processed".into(),
                value: "true".into(),
            },
            TransformAction::SetField {
                key: "status".into(),
                value: "processed".into(),
            },
        ];
        hub.add_route(route).unwrap();
        let msg = make_message("m1", "any");
        let results = hub.route(msg).unwrap();
        assert_eq!(results[0].headers.get("processed"), Some(&"true".to_string()));
    }

    #[test]
    fn test_rename_field_transform() {
        let mut hub = IntegrationHub::new();
        let mut route = make_route("r1", RouteCondition::Always);
        route.transforms = vec![TransformAction::RenameField {
            from: "key".into(),
            to: "renamed_key".into(),
        }];
        hub.add_route(route).unwrap();
        let msg = make_message("m1", "any");
        let results = hub.route(msg).unwrap();
        let body = &results[0].body;
        assert!(body.get("renamed_key").is_some());
        assert!(body.get("key").is_none());
    }

    #[test]
    fn test_wrap_body_transform() {
        let mut hub = IntegrationHub::new();
        let mut route = make_route("r1", RouteCondition::Always);
        route.transforms = vec![TransformAction::WrapBody("data".into())];
        hub.add_route(route).unwrap();
        let msg = make_message("m1", "any");
        let results = hub.route(msg).unwrap();
        assert!(results[0].body.get("data").is_some());
    }

    #[test]
    fn test_enrichment() {
        let mut hub = IntegrationHub::new();
        let mut table = HashMap::new();
        table.insert("US".into(), "United States".into());
        table.insert("UK".into(), "United Kingdom".into());
        let mut route = make_route("r1", RouteCondition::Always);
        route.enrichments = vec![EnrichmentDef {
            source_key: "country".into(),
            target_header: "country_name".into(),
            lookup_table: table,
        }];
        hub.add_route(route).unwrap();

        let msg = Message::new("m1", "any", serde_json::json!({"country": "US"}));
        let results = hub.route(msg).unwrap();
        assert_eq!(
            results[0].headers.get("country_name"),
            Some(&"United States".to_string())
        );
    }

    #[test]
    fn test_split_message() {
        let hub = IntegrationHub::new();
        let msg = Message::new(
            "m1",
            "batch",
            serde_json::json!({"items": [1, 2, 3]}),
        );
        let parts = hub.split(&msg, "items");
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].body, serde_json::json!(1));
        assert_eq!(parts[2].body, serde_json::json!(3));
    }

    #[test]
    fn test_split_array_body() {
        let hub = IntegrationHub::new();
        let msg = Message::new("m1", "batch", serde_json::json!([10, 20]));
        let parts = hub.split(&msg, "");
        assert_eq!(parts.len(), 2);
    }

    #[test]
    fn test_aggregation() {
        let mut hub = IntegrationHub::new();
        let gid = hub.start_aggregation("g1", "corr-1", Some(2));

        let m1 = Message::new("m1", "part", serde_json::json!({"a": 1}));
        let complete1 = hub.aggregate(&gid, m1).unwrap();
        assert!(!complete1);

        let m2 = Message::new("m2", "part", serde_json::json!({"b": 2}));
        let complete2 = hub.aggregate(&gid, m2).unwrap();
        assert!(complete2);

        let result = hub.complete_aggregation(&gid).unwrap();
        assert_eq!(result.status, MessageStatus::Aggregated);
        if let Value::Array(arr) = &result.body {
            assert_eq!(arr.len(), 2);
        } else {
            panic!("expected array body");
        }
    }

    #[test]
    fn test_dead_letter_queue() {
        let mut hub = setup();
        let msg = make_message("m1", "unknown");
        let _ = hub.route(msg);
        assert_eq!(hub.total_dead_lettered(), 1);
        assert_eq!(hub.dead_letter_queue()[0].error, "no matching route");
    }

    #[test]
    fn test_dead_letter_capacity() {
        let mut hub = IntegrationHub::new().with_capacity(1);
        hub.add_route(make_route("r1", RouteCondition::MessageType("a".into())))
            .unwrap();
        let _ = hub.route(make_message("m1", "x")); // dead letter 1
        let result = hub.route(make_message("m2", "y")); // should fail
        assert!(result.is_err());
    }

    #[test]
    fn test_drain_dead_letter() {
        let mut hub = setup();
        let _ = hub.route(make_message("m1", "unknown"));
        assert_eq!(hub.total_dead_lettered(), 1);
        let drained = hub.drain_dead_letter();
        assert_eq!(drained.len(), 1);
        assert_eq!(hub.total_dead_lettered(), 0);
    }

    #[test]
    fn test_route_stats() {
        let mut hub = setup();
        hub.route(make_message("m1", "order")).unwrap();
        hub.route(make_message("m2", "order")).unwrap();
        let stats = hub.route_stats("r1").unwrap();
        assert_eq!(stats.messages_routed, 2);
    }

    #[test]
    fn test_header_based_condition() {
        let mut hub = IntegrationHub::new();
        hub.add_route(make_route(
            "r1",
            RouteCondition::HeaderEquals {
                key: "priority".into(),
                value: "high".into(),
            },
        ))
        .unwrap();

        let msg = make_message("m1", "any").with_header("priority", "high");
        let results = hub.route(msg).unwrap();
        assert_eq!(results.len(), 1);

        let msg2 = make_message("m2", "any").with_header("priority", "low");
        let err = hub.route(msg2);
        assert!(err.is_err());
    }

    #[test]
    fn test_body_contains_condition() {
        let cond = RouteCondition::BodyContains("key".into());
        let msg = make_message("m1", "any");
        assert!(evaluate_route_condition(&cond, &msg));

        let msg2 = Message::new("m2", "any", serde_json::json!({"other": 1}));
        assert!(!evaluate_route_condition(&cond, &msg2));
    }

    #[test]
    fn test_and_or_conditions() {
        let cond = RouteCondition::And(vec![
            RouteCondition::MessageType("order".into()),
            RouteCondition::HeaderPresent("auth".into()),
        ]);
        let msg = make_message("m1", "order").with_header("auth", "token");
        assert!(evaluate_route_condition(&cond, &msg));

        let msg2 = make_message("m2", "order");
        assert!(!evaluate_route_condition(&cond, &msg2));

        let or_cond = RouteCondition::Or(vec![
            RouteCondition::MessageType("order".into()),
            RouteCondition::MessageType("invoice".into()),
        ]);
        let msg3 = make_message("m3", "invoice");
        assert!(evaluate_route_condition(&or_cond, &msg3));
    }

    #[test]
    fn test_multicast_strategy() {
        let mut hub = IntegrationHub::new().with_strategy(RoutingStrategy::Multicast);
        hub.add_route(make_route("r1", RouteCondition::Always)).unwrap();
        hub.add_route(make_route("r2", RouteCondition::Always)).unwrap();
        let msg = make_message("m1", "any");
        let results = hub.route(msg).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_round_robin_strategy() {
        let mut hub = IntegrationHub::new().with_strategy(RoutingStrategy::RoundRobin);
        hub.add_route(make_route("r1", RouteCondition::Always)).unwrap();
        hub.add_route(make_route("r2", RouteCondition::Always)).unwrap();

        let r1 = hub.route(make_message("m1", "any")).unwrap();
        assert_eq!(r1.len(), 1);
        let first_route = r1[0].source_route.clone().unwrap();

        let r2 = hub.route(make_message("m2", "any")).unwrap();
        assert_eq!(r2.len(), 1);
        let second_route = r2[0].source_route.clone().unwrap();

        assert_ne!(first_route, second_route);
    }

    #[test]
    fn test_total_processed() {
        let mut hub = setup();
        assert_eq!(hub.total_processed(), 0);
        hub.route(make_message("m1", "order")).unwrap();
        assert_eq!(hub.total_processed(), 1);
    }

    #[test]
    fn test_messages_for_route() {
        let mut hub = setup();
        hub.route(make_message("m1", "order")).unwrap();
        hub.route(make_message("m2", "invoice")).unwrap();
        assert_eq!(hub.messages_for_route("r1").len(), 1);
        assert_eq!(hub.messages_for_route("r2").len(), 1);
    }

    #[test]
    fn test_message_with_correlation() {
        let msg = make_message("m1", "order").with_correlation("batch-1");
        assert_eq!(msg.correlation_id, Some("batch-1".to_string()));
    }

    #[test]
    fn test_remove_header_transform() {
        let mut msg = make_message("m1", "any").with_header("temp", "val");
        apply_transform(&mut msg, &TransformAction::RemoveHeader("temp".into())).unwrap();
        assert!(msg.headers.get("temp").is_none());
    }

    #[test]
    fn test_remove_field_transform() {
        let mut msg = Message::new("m1", "any", serde_json::json!({"a": 1, "b": 2}));
        apply_transform(&mut msg, &TransformAction::RemoveField("a".into())).unwrap();
        assert!(msg.body.get("a").is_none());
        assert!(msg.body.get("b").is_some());
    }
}
