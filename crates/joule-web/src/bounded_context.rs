//! Bounded context integration — anti-corruption layer (translator), context
//! mapping (shared kernel, customer-supplier, conformist), published language
//! definition, integration event contracts, and context boundary enforcement.
//!
//! Replaces ad-hoc microservice integration in JS/TS with a pure-Rust bounded
//! context framework that enforces DDD integration patterns.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Bounded context errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BoundedContextError {
    /// Context not found.
    ContextNotFound(String),
    /// Translation failed.
    TranslationFailed { from: String, to: String, reason: String },
    /// Contract violation.
    ContractViolation { context: String, reason: String },
    /// Mapping not found.
    MappingNotFound { upstream: String, downstream: String },
    /// Duplicate context.
    DuplicateContext(String),
    /// Invalid integration event.
    InvalidIntegrationEvent { event_type: String, reason: String },
    /// Boundary violation.
    BoundaryViolation { context: String, reason: String },
}

impl std::fmt::Display for BoundedContextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ContextNotFound(name) => write!(f, "context not found: {name}"),
            Self::TranslationFailed { from, to, reason } => {
                write!(f, "translation failed from {from} to {to}: {reason}")
            }
            Self::ContractViolation { context, reason } => {
                write!(f, "contract violation in {context}: {reason}")
            }
            Self::MappingNotFound { upstream, downstream } => {
                write!(f, "no mapping from {upstream} to {downstream}")
            }
            Self::DuplicateContext(name) => write!(f, "duplicate context: {name}"),
            Self::InvalidIntegrationEvent { event_type, reason } => {
                write!(f, "invalid integration event {event_type}: {reason}")
            }
            Self::BoundaryViolation { context, reason } => {
                write!(f, "boundary violation in {context}: {reason}")
            }
        }
    }
}

impl std::error::Error for BoundedContextError {}

// ── Context Mapping Type ────────────────────────────────────────

/// The type of relationship between two bounded contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MappingType {
    /// Shared kernel: both contexts share a subset of the model.
    SharedKernel,
    /// Customer-supplier: upstream provides, downstream consumes.
    CustomerSupplier,
    /// Conformist: downstream conforms to upstream model.
    Conformist,
    /// Anti-corruption layer: downstream translates upstream model.
    AntiCorruptionLayer,
    /// Open host service: upstream provides a well-defined protocol.
    OpenHostService,
    /// Published language: shared interchange format.
    PublishedLanguage,
    /// Separate ways: no integration.
    SeparateWays,
}

// ── BoundedContext ──────────────────────────────────────────────

/// A bounded context with its owned language and contracts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundedContext {
    pub name: String,
    pub description: String,
    pub owned_types: Vec<String>,
    pub published_events: Vec<String>,
    pub consumed_events: Vec<String>,
    pub created_at: DateTime<Utc>,
}

impl BoundedContext {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            owned_types: Vec::new(),
            published_events: Vec::new(),
            consumed_events: Vec::new(),
            created_at: Utc::now(),
        }
    }

    /// Register a type owned by this context.
    pub fn own_type(&mut self, type_name: impl Into<String>) {
        self.owned_types.push(type_name.into());
    }

    /// Register a published event.
    pub fn publish_event(&mut self, event_type: impl Into<String>) {
        self.published_events.push(event_type.into());
    }

    /// Register a consumed event.
    pub fn consume_event(&mut self, event_type: impl Into<String>) {
        self.consumed_events.push(event_type.into());
    }

    /// Whether this context owns a given type.
    pub fn owns_type(&self, type_name: &str) -> bool {
        self.owned_types.iter().any(|t| t == type_name)
    }

    /// Whether this context publishes a given event.
    pub fn publishes_event(&self, event_type: &str) -> bool {
        self.published_events.iter().any(|e| e == event_type)
    }

    /// Whether this context consumes a given event.
    pub fn consumes_event(&self, event_type: &str) -> bool {
        self.consumed_events.iter().any(|e| e == event_type)
    }
}

// ── ContextMapping ──────────────────────────────────────────────

/// A mapping relationship between two contexts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextMapping {
    pub upstream: String,
    pub downstream: String,
    pub mapping_type: MappingType,
    pub shared_events: Vec<String>,
    pub description: String,
}

impl ContextMapping {
    pub fn new(
        upstream: impl Into<String>,
        downstream: impl Into<String>,
        mapping_type: MappingType,
    ) -> Self {
        Self {
            upstream: upstream.into(),
            downstream: downstream.into(),
            mapping_type,
            shared_events: Vec::new(),
            description: String::new(),
        }
    }

    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    pub fn add_shared_event(&mut self, event_type: impl Into<String>) {
        self.shared_events.push(event_type.into());
    }
}

// ── Integration Event ───────────────────────────────────────────

/// An integration event published between bounded contexts.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntegrationEvent {
    pub event_id: String,
    pub event_type: String,
    pub source_context: String,
    pub version: u32,
    pub timestamp: DateTime<Utc>,
    pub data: HashMap<String, String>,
    pub correlation_id: Option<String>,
}

impl IntegrationEvent {
    pub fn new(
        event_type: impl Into<String>,
        source_context: impl Into<String>,
        data: HashMap<String, String>,
    ) -> Self {
        let ts = Utc::now();
        let et = event_type.into();
        let sc = source_context.into();
        Self {
            event_id: format!("{}-{}-{}", sc, et, ts.timestamp_nanos_opt().unwrap_or(0)),
            event_type: et,
            source_context: sc,
            version: 1,
            timestamp: ts,
            data,
            correlation_id: None,
        }
    }

    pub fn with_correlation(mut self, id: impl Into<String>) -> Self {
        self.correlation_id = Some(id.into());
        self
    }

    pub fn with_version(mut self, version: u32) -> Self {
        self.version = version;
        self
    }
}

// ── IntegrationEventContract ────────────────────────────────────

/// A contract defining the expected shape of an integration event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContract {
    pub event_type: String,
    pub version: u32,
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
    pub source_context: String,
}

impl EventContract {
    pub fn new(
        event_type: impl Into<String>,
        version: u32,
        source_context: impl Into<String>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            version,
            required_fields: Vec::new(),
            optional_fields: Vec::new(),
            source_context: source_context.into(),
        }
    }

    pub fn require_field(mut self, field: impl Into<String>) -> Self {
        self.required_fields.push(field.into());
        self
    }

    pub fn optional_field(mut self, field: impl Into<String>) -> Self {
        self.optional_fields.push(field.into());
        self
    }

    /// Validate an integration event against this contract.
    pub fn validate(&self, event: &IntegrationEvent) -> Result<(), BoundedContextError> {
        if event.event_type != self.event_type {
            return Err(BoundedContextError::ContractViolation {
                context: self.source_context.clone(),
                reason: format!(
                    "event type mismatch: expected {}, got {}",
                    self.event_type, event.event_type
                ),
            });
        }
        for field in &self.required_fields {
            if !event.data.contains_key(field) {
                return Err(BoundedContextError::ContractViolation {
                    context: self.source_context.clone(),
                    reason: format!("missing required field: {field}"),
                });
            }
        }
        Ok(())
    }
}

// ── Anti-Corruption Layer ───────────────────────────────────────

/// A translator function for the anti-corruption layer.
#[derive(Clone)]
pub struct Translator {
    pub name: String,
    pub from_context: String,
    pub to_context: String,
    translate_fn: fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>,
}

impl Translator {
    pub fn new(
        name: impl Into<String>,
        from_context: impl Into<String>,
        to_context: impl Into<String>,
        translate_fn: fn(&HashMap<String, String>) -> Result<HashMap<String, String>, String>,
    ) -> Self {
        Self {
            name: name.into(),
            from_context: from_context.into(),
            to_context: to_context.into(),
            translate_fn,
        }
    }

    /// Translate data from one context to another.
    pub fn translate(
        &self,
        data: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, BoundedContextError> {
        (self.translate_fn)(data).map_err(|reason| BoundedContextError::TranslationFailed {
            from: self.from_context.clone(),
            to: self.to_context.clone(),
            reason,
        })
    }
}

impl std::fmt::Debug for Translator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Translator")
            .field("name", &self.name)
            .field("from_context", &self.from_context)
            .field("to_context", &self.to_context)
            .finish()
    }
}

/// Anti-corruption layer with registered translators.
#[derive(Debug, Default)]
pub struct AntiCorruptionLayer {
    translators: Vec<Translator>,
}

impl AntiCorruptionLayer {
    pub fn new() -> Self {
        Self { translators: Vec::new() }
    }

    /// Register a translator.
    pub fn register(&mut self, translator: Translator) {
        self.translators.push(translator);
    }

    /// Translate data between contexts using a registered translator.
    pub fn translate(
        &self,
        from: &str,
        to: &str,
        data: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, BoundedContextError> {
        let translator = self.translators.iter()
            .find(|t| t.from_context == from && t.to_context == to)
            .ok_or_else(|| BoundedContextError::MappingNotFound {
                upstream: from.to_string(),
                downstream: to.to_string(),
            })?;
        translator.translate(data)
    }

    /// Registered translator count.
    pub fn translator_count(&self) -> usize {
        self.translators.len()
    }
}

// ── Published Language ──────────────────────────────────────────

/// A published language definition — shared interchange format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishedLanguage {
    pub name: String,
    pub version: u32,
    pub types: Vec<TypeDefinition>,
    pub events: Vec<EventContract>,
}

impl PublishedLanguage {
    pub fn new(name: impl Into<String>, version: u32) -> Self {
        Self {
            name: name.into(),
            version,
            types: Vec::new(),
            events: Vec::new(),
        }
    }

    pub fn add_type(&mut self, type_def: TypeDefinition) {
        self.types.push(type_def);
    }

    pub fn add_event_contract(&mut self, contract: EventContract) {
        self.events.push(contract);
    }

    /// Find a type definition by name.
    pub fn find_type(&self, name: &str) -> Option<&TypeDefinition> {
        self.types.iter().find(|t| t.name == name)
    }

    /// Find an event contract by type.
    pub fn find_event(&self, event_type: &str) -> Option<&EventContract> {
        self.events.iter().find(|e| e.event_type == event_type)
    }
}

/// A type definition in the published language.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDefinition {
    pub name: String,
    pub fields: Vec<FieldDefinition>,
    pub description: String,
}

impl TypeDefinition {
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            description: description.into(),
        }
    }

    pub fn add_field(mut self, field: FieldDefinition) -> Self {
        self.fields.push(field);
        self
    }
}

/// A field definition in a type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub field_type: String,
    pub required: bool,
    pub description: String,
}

impl FieldDefinition {
    pub fn new(
        name: impl Into<String>,
        field_type: impl Into<String>,
        required: bool,
    ) -> Self {
        Self {
            name: name.into(),
            field_type: field_type.into(),
            required,
            description: String::new(),
        }
    }

    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }
}

// ── ContextMap ──────────────────────────────────────────────────

/// A context map managing bounded contexts and their relationships.
#[derive(Debug, Default)]
pub struct ContextMap {
    contexts: HashMap<String, BoundedContext>,
    mappings: Vec<ContextMapping>,
    acl: AntiCorruptionLayer,
}

impl ContextMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a bounded context.
    pub fn register_context(
        &mut self,
        context: BoundedContext,
    ) -> Result<(), BoundedContextError> {
        if self.contexts.contains_key(&context.name) {
            return Err(BoundedContextError::DuplicateContext(context.name.clone()));
        }
        self.contexts.insert(context.name.clone(), context);
        Ok(())
    }

    /// Add a mapping between contexts.
    pub fn add_mapping(&mut self, mapping: ContextMapping) -> Result<(), BoundedContextError> {
        if !self.contexts.contains_key(&mapping.upstream) {
            return Err(BoundedContextError::ContextNotFound(mapping.upstream.clone()));
        }
        if !self.contexts.contains_key(&mapping.downstream) {
            return Err(BoundedContextError::ContextNotFound(mapping.downstream.clone()));
        }
        self.mappings.push(mapping);
        Ok(())
    }

    /// Get a context by name.
    pub fn get_context(&self, name: &str) -> Option<&BoundedContext> {
        self.contexts.get(name)
    }

    /// Get mappings for a context (as upstream).
    pub fn upstream_mappings(&self, context_name: &str) -> Vec<&ContextMapping> {
        self.mappings.iter().filter(|m| m.upstream == context_name).collect()
    }

    /// Get mappings for a context (as downstream).
    pub fn downstream_mappings(&self, context_name: &str) -> Vec<&ContextMapping> {
        self.mappings.iter().filter(|m| m.downstream == context_name).collect()
    }

    /// Find the mapping between two contexts.
    pub fn find_mapping(&self, upstream: &str, downstream: &str) -> Option<&ContextMapping> {
        self.mappings.iter().find(|m| m.upstream == upstream && m.downstream == downstream)
    }

    /// Register a translator in the ACL.
    pub fn register_translator(&mut self, translator: Translator) {
        self.acl.register(translator);
    }

    /// Translate data through the ACL.
    pub fn translate(
        &self,
        from: &str,
        to: &str,
        data: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, BoundedContextError> {
        self.acl.translate(from, to, data)
    }

    /// Validate that a context boundary is not violated.
    pub fn validate_boundary(
        &self,
        context_name: &str,
        type_name: &str,
    ) -> Result<(), BoundedContextError> {
        let ctx = self.contexts.get(context_name)
            .ok_or_else(|| BoundedContextError::ContextNotFound(context_name.to_string()))?;
        if !ctx.owns_type(type_name) {
            return Err(BoundedContextError::BoundaryViolation {
                context: context_name.to_string(),
                reason: format!("type {type_name} is not owned by this context"),
            });
        }
        Ok(())
    }

    /// Context count.
    pub fn context_count(&self) -> usize {
        self.contexts.len()
    }

    /// Mapping count.
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_contexts() -> (BoundedContext, BoundedContext) {
        let mut orders = BoundedContext::new("Orders", "Order management context");
        orders.own_type("Order");
        orders.own_type("LineItem");
        orders.publish_event("order_placed");
        orders.publish_event("order_shipped");

        let mut inventory = BoundedContext::new("Inventory", "Inventory management context");
        inventory.own_type("StockItem");
        inventory.consume_event("order_placed");

        (orders, inventory)
    }

    #[test]
    fn test_bounded_context_creation() {
        let (orders, _) = make_contexts();
        assert_eq!(orders.name, "Orders");
        assert!(orders.owns_type("Order"));
        assert!(!orders.owns_type("StockItem"));
    }

    #[test]
    fn test_publishes_and_consumes() {
        let (orders, inventory) = make_contexts();
        assert!(orders.publishes_event("order_placed"));
        assert!(inventory.consumes_event("order_placed"));
        assert!(!inventory.publishes_event("order_placed"));
    }

    #[test]
    fn test_context_map_register() {
        let mut map = ContextMap::new();
        let (orders, inventory) = make_contexts();
        map.register_context(orders).unwrap();
        map.register_context(inventory).unwrap();
        assert_eq!(map.context_count(), 2);
    }

    #[test]
    fn test_context_map_duplicate() {
        let mut map = ContextMap::new();
        let (orders, _) = make_contexts();
        map.register_context(orders.clone()).unwrap();
        let result = map.register_context(orders);
        assert!(matches!(result, Err(BoundedContextError::DuplicateContext(_))));
    }

    #[test]
    fn test_context_mapping() {
        let mut map = ContextMap::new();
        let (orders, inventory) = make_contexts();
        map.register_context(orders).unwrap();
        map.register_context(inventory).unwrap();
        let mapping = ContextMapping::new("Orders", "Inventory", MappingType::CustomerSupplier);
        map.add_mapping(mapping).unwrap();
        assert_eq!(map.mapping_count(), 1);
    }

    #[test]
    fn test_mapping_not_found_context() {
        let mut map = ContextMap::new();
        let (orders, _) = make_contexts();
        map.register_context(orders).unwrap();
        let mapping = ContextMapping::new("Orders", "Missing", MappingType::Conformist);
        let result = map.add_mapping(mapping);
        assert!(matches!(result, Err(BoundedContextError::ContextNotFound(_))));
    }

    #[test]
    fn test_upstream_downstream_mappings() {
        let mut map = ContextMap::new();
        let (orders, inventory) = make_contexts();
        map.register_context(orders).unwrap();
        map.register_context(inventory).unwrap();
        let mapping = ContextMapping::new("Orders", "Inventory", MappingType::CustomerSupplier);
        map.add_mapping(mapping).unwrap();
        assert_eq!(map.upstream_mappings("Orders").len(), 1);
        assert_eq!(map.downstream_mappings("Inventory").len(), 1);
        assert_eq!(map.upstream_mappings("Inventory").len(), 0);
    }

    #[test]
    fn test_anti_corruption_layer() {
        let mut acl = AntiCorruptionLayer::new();
        acl.register(Translator::new(
            "order_to_stock",
            "Orders",
            "Inventory",
            |data| {
                let mut result = HashMap::new();
                if let Some(item_id) = data.get("line_item_id") {
                    result.insert("stock_item_id".to_string(), item_id.clone());
                }
                if let Some(qty) = data.get("quantity") {
                    result.insert("reserve_quantity".to_string(), qty.clone());
                }
                Ok(result)
            },
        ));
        let mut input = HashMap::new();
        input.insert("line_item_id".to_string(), "LI-001".to_string());
        input.insert("quantity".to_string(), "5".to_string());
        let output = acl.translate("Orders", "Inventory", &input).unwrap();
        assert_eq!(output.get("stock_item_id").unwrap(), "LI-001");
        assert_eq!(output.get("reserve_quantity").unwrap(), "5");
    }

    #[test]
    fn test_acl_translation_not_found() {
        let acl = AntiCorruptionLayer::new();
        let result = acl.translate("A", "B", &HashMap::new());
        assert!(matches!(result, Err(BoundedContextError::MappingNotFound { .. })));
    }

    #[test]
    fn test_integration_event() {
        let mut data = HashMap::new();
        data.insert("order_id".to_string(), "ORD-001".to_string());
        let event = IntegrationEvent::new("order_placed", "Orders", data)
            .with_correlation("corr-123");
        assert_eq!(event.event_type, "order_placed");
        assert_eq!(event.source_context, "Orders");
        assert_eq!(event.correlation_id.as_deref(), Some("corr-123"));
    }

    #[test]
    fn test_event_contract_validation_success() {
        let contract = EventContract::new("order_placed", 1, "Orders")
            .require_field("order_id")
            .require_field("customer_id");
        let mut data = HashMap::new();
        data.insert("order_id".to_string(), "ORD-001".to_string());
        data.insert("customer_id".to_string(), "CUST-001".to_string());
        let event = IntegrationEvent::new("order_placed", "Orders", data);
        assert!(contract.validate(&event).is_ok());
    }

    #[test]
    fn test_event_contract_missing_field() {
        let contract = EventContract::new("order_placed", 1, "Orders")
            .require_field("order_id");
        let event = IntegrationEvent::new("order_placed", "Orders", HashMap::new());
        let result = contract.validate(&event);
        assert!(matches!(result, Err(BoundedContextError::ContractViolation { .. })));
    }

    #[test]
    fn test_event_contract_type_mismatch() {
        let contract = EventContract::new("order_placed", 1, "Orders");
        let event = IntegrationEvent::new("order_shipped", "Orders", HashMap::new());
        let result = contract.validate(&event);
        assert!(matches!(result, Err(BoundedContextError::ContractViolation { .. })));
    }

    #[test]
    fn test_published_language() {
        let mut lang = PublishedLanguage::new("OrderLanguage", 1);
        let order_type = TypeDefinition::new("Order", "An order in the system")
            .add_field(FieldDefinition::new("order_id", "string", true))
            .add_field(FieldDefinition::new("total", "decimal", true));
        lang.add_type(order_type);
        lang.add_event_contract(
            EventContract::new("order_placed", 1, "Orders")
                .require_field("order_id"),
        );
        assert!(lang.find_type("Order").is_some());
        assert!(lang.find_type("Missing").is_none());
        assert!(lang.find_event("order_placed").is_some());
    }

    #[test]
    fn test_validate_boundary_success() {
        let mut map = ContextMap::new();
        let (orders, _) = make_contexts();
        map.register_context(orders).unwrap();
        assert!(map.validate_boundary("Orders", "Order").is_ok());
    }

    #[test]
    fn test_validate_boundary_violation() {
        let mut map = ContextMap::new();
        let (orders, _) = make_contexts();
        map.register_context(orders).unwrap();
        let result = map.validate_boundary("Orders", "StockItem");
        assert!(matches!(result, Err(BoundedContextError::BoundaryViolation { .. })));
    }

    #[test]
    fn test_context_map_translate() {
        let mut map = ContextMap::new();
        let (orders, inventory) = make_contexts();
        map.register_context(orders).unwrap();
        map.register_context(inventory).unwrap();
        map.register_translator(Translator::new(
            "test",
            "Orders",
            "Inventory",
            |data| Ok(data.clone()),
        ));
        let data = HashMap::new();
        assert!(map.translate("Orders", "Inventory", &data).is_ok());
    }

    #[test]
    fn test_mapping_with_shared_events() {
        let mut mapping = ContextMapping::new("Orders", "Inventory", MappingType::SharedKernel)
            .with_description("shared product reference");
        mapping.add_shared_event("product_updated");
        assert_eq!(mapping.shared_events.len(), 1);
        assert_eq!(mapping.description, "shared product reference");
    }

    #[test]
    fn test_find_mapping() {
        let mut map = ContextMap::new();
        let (orders, inventory) = make_contexts();
        map.register_context(orders).unwrap();
        map.register_context(inventory).unwrap();
        map.add_mapping(ContextMapping::new("Orders", "Inventory", MappingType::CustomerSupplier)).unwrap();
        assert!(map.find_mapping("Orders", "Inventory").is_some());
        assert!(map.find_mapping("Inventory", "Orders").is_none());
    }
}
