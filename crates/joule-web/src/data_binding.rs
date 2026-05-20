//! Two-way data binding engine with computed properties, validation,
//! change notification, converter functions, and binding groups.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// ── BindingId ────────────────────────────────────────────────

/// Unique identifier for a binding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BindingId(usize);

// ── ValidationResult ─────────────────────────────────────────

/// Result of a validation check.
#[derive(Debug, Clone, PartialEq)]
pub enum ValidationResult {
    Valid,
    Invalid(String),
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool {
        matches!(self, ValidationResult::Valid)
    }

    pub fn error_message(&self) -> Option<&str> {
        match self {
            ValidationResult::Invalid(msg) => Some(msg),
            ValidationResult::Valid => None,
        }
    }
}

// ── Property ─────────────────────────────────────────────────

/// A bindable property with change notification and validation.
pub struct Property<T> {
    value: T,
    validators: Vec<Box<dyn Fn(&T) -> ValidationResult>>,
    listeners: Vec<Option<Box<dyn FnMut(&T, &T)>>>,
    next_listener_id: usize,
    last_validation: ValidationResult,
}

impl<T: Clone + PartialEq> Property<T> {
    pub fn new(initial: T) -> Self {
        Self {
            value: initial,
            validators: Vec::new(),
            listeners: Vec::new(),
            next_listener_id: 0,
            last_validation: ValidationResult::Valid,
        }
    }

    /// Get the current value.
    pub fn get(&self) -> &T {
        &self.value
    }

    /// Set the value. Validates first, then notifies listeners if changed.
    /// Returns the validation result.
    pub fn set(&mut self, new_value: T) -> ValidationResult {
        // Run validation
        for validator in &self.validators {
            let result = validator(&new_value);
            if !result.is_valid() {
                self.last_validation = result.clone();
                return result;
            }
        }

        self.last_validation = ValidationResult::Valid;

        if self.value != new_value {
            let old = self.value.clone();
            self.value = new_value;
            let new_ref = &self.value;
            for slot in &mut self.listeners {
                if let Some(f) = slot.as_mut() {
                    f(&old, new_ref);
                }
            }
        }

        ValidationResult::Valid
    }

    /// Add a validator.
    pub fn add_validator(&mut self, validator: impl Fn(&T) -> ValidationResult + 'static) {
        self.validators.push(Box::new(validator));
    }

    /// Subscribe to changes. Callback receives (old_value, new_value).
    /// Returns an ID for removal.
    pub fn on_change(&mut self, f: impl FnMut(&T, &T) + 'static) -> usize {
        let id = self.next_listener_id;
        while self.listeners.len() <= id {
            self.listeners.push(None);
        }
        self.listeners[id] = Some(Box::new(f));
        self.next_listener_id += 1;
        id
    }

    /// Remove a change listener.
    pub fn off_change(&mut self, id: usize) {
        if id < self.listeners.len() {
            self.listeners[id] = None;
        }
    }

    /// Get the last validation result.
    pub fn validation_state(&self) -> &ValidationResult {
        &self.last_validation
    }
}

// ── Converter ────────────────────────────────────────────────

/// A bidirectional converter between two types.
pub struct Converter<A, B> {
    forward: Box<dyn Fn(&A) -> B>,
    backward: Box<dyn Fn(&B) -> A>,
}

impl<A, B> Converter<A, B> {
    pub fn new(
        forward: impl Fn(&A) -> B + 'static,
        backward: impl Fn(&B) -> A + 'static,
    ) -> Self {
        Self {
            forward: Box::new(forward),
            backward: Box::new(backward),
        }
    }

    pub fn convert(&self, value: &A) -> B {
        (self.forward)(value)
    }

    pub fn convert_back(&self, value: &B) -> A {
        (self.backward)(value)
    }
}

// ── ComputedProperty ─────────────────────────────────────────

/// A read-only computed property derived from other values.
pub struct ComputedProperty<T> {
    compute: Box<dyn Fn() -> T>,
    cached: Option<T>,
    dirty: bool,
}

impl<T: Clone> ComputedProperty<T> {
    pub fn new(compute: impl Fn() -> T + 'static) -> Self {
        Self {
            compute: Box::new(compute),
            cached: None,
            dirty: true,
        }
    }

    /// Get the computed value. Uses cache if not dirty.
    pub fn get(&mut self) -> T {
        if self.dirty || self.cached.is_none() {
            let val = (self.compute)();
            self.cached = Some(val.clone());
            self.dirty = false;
            val
        } else {
            self.cached.as_ref().unwrap().clone()
        }
    }

    /// Mark the computed value as dirty (needs recomputation).
    pub fn invalidate(&mut self) {
        self.dirty = true;
    }

    /// Check if the value needs recomputation.
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }
}

// ── BindingExpression ────────────────────────────────────────

/// A binding expression that evaluates a string expression against
/// a context of named values.
pub struct BindingExpression {
    /// The expression template, e.g. "Hello, {name}!"
    template: String,
}

impl BindingExpression {
    pub fn new(template: impl Into<String>) -> Self {
        Self {
            template: template.into(),
        }
    }

    /// Evaluate the expression by substituting `{key}` placeholders.
    pub fn evaluate(&self, context: &HashMap<String, String>) -> String {
        let mut result = self.template.clone();
        for (key, value) in context {
            let placeholder = format!("{{{}}}", key);
            result = result.replace(&placeholder, value);
        }
        result
    }

    /// Get all placeholder keys in the template.
    pub fn keys(&self) -> Vec<String> {
        let mut keys = Vec::new();
        let chars: Vec<char> = self.template.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '{' {
                let start = i + 1;
                while i < chars.len() && chars[i] != '}' {
                    i += 1;
                }
                if i < chars.len() {
                    let key: String = chars[start..i].iter().collect();
                    if !key.is_empty() {
                        keys.push(key);
                    }
                }
            }
            i += 1;
        }
        keys
    }
}

// ── BindingGroup ─────────────────────────────────────────────

/// A group of related bindings that can be enabled/disabled together.
pub struct BindingGroup {
    name: String,
    bindings: Vec<BindingEntry>,
    enabled: bool,
}

struct BindingEntry {
    source_key: String,
    target_key: String,
    active: bool,
}

impl BindingGroup {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            bindings: Vec::new(),
            enabled: true,
        }
    }

    /// Add a binding from source to target.
    pub fn add_binding(
        &mut self,
        source_key: impl Into<String>,
        target_key: impl Into<String>,
    ) {
        self.bindings.push(BindingEntry {
            source_key: source_key.into(),
            target_key: target_key.into(),
            active: true,
        });
    }

    /// Enable all bindings in the group.
    pub fn enable(&mut self) {
        self.enabled = true;
        for b in &mut self.bindings {
            b.active = true;
        }
    }

    /// Disable all bindings in the group.
    pub fn disable(&mut self) {
        self.enabled = false;
        for b in &mut self.bindings {
            b.active = false;
        }
    }

    /// Check if the group is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get the group name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of bindings in the group.
    pub fn len(&self) -> usize {
        self.bindings.len()
    }

    /// Whether the group is empty.
    pub fn is_empty(&self) -> bool {
        self.bindings.is_empty()
    }

    /// Apply all active bindings: copy values from source to target in the given map.
    pub fn apply(&self, values: &mut HashMap<String, String>) {
        if !self.enabled {
            return;
        }
        let snapshot: HashMap<String, String> = values.clone();
        for binding in &self.bindings {
            if binding.active {
                if let Some(val) = snapshot.get(&binding.source_key) {
                    values.insert(binding.target_key.clone(), val.clone());
                }
            }
        }
    }

    /// Get binding pairs as (source, target) tuples.
    pub fn binding_pairs(&self) -> Vec<(String, String)> {
        self.bindings
            .iter()
            .map(|b| (b.source_key.clone(), b.target_key.clone()))
            .collect()
    }
}

// ── TwoWayBinding ────────────────────────────────────────────

/// Engine for managing two-way bindings between named properties.
pub struct BindingEngine {
    properties: HashMap<String, Rc<RefCell<Property<String>>>>,
    bindings: Vec<(String, String)>,
    /// Prevents infinite loops during synchronization.
    syncing: RefCell<bool>,
}

impl Default for BindingEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl BindingEngine {
    pub fn new() -> Self {
        Self {
            properties: HashMap::new(),
            bindings: Vec::new(),
            syncing: RefCell::new(false),
        }
    }

    /// Register a property with a name.
    pub fn register(&mut self, name: impl Into<String>, initial: impl Into<String>) {
        let name = name.into();
        let prop = Rc::new(RefCell::new(Property::new(initial.into())));
        self.properties.insert(name, prop);
    }

    /// Get a property by name.
    pub fn get_property(&self, name: &str) -> Option<Rc<RefCell<Property<String>>>> {
        self.properties.get(name).cloned()
    }

    /// Get a property value by name.
    pub fn get_value(&self, name: &str) -> Option<String> {
        self.properties
            .get(name)
            .map(|p| p.borrow().get().clone())
    }

    /// Set a property value by name. Propagates to bound properties.
    pub fn set_value(&self, name: &str, value: impl Into<String>) -> Option<ValidationResult> {
        let value = value.into();

        // Prevent re-entrant sync
        if *self.syncing.borrow() {
            if let Some(prop) = self.properties.get(name) {
                let result = prop.borrow_mut().set(value);
                return Some(result);
            }
            return None;
        }

        if let Some(prop) = self.properties.get(name) {
            let result = prop.borrow_mut().set(value.clone());
            if result.is_valid() {
                // Propagate to bound targets
                *self.syncing.borrow_mut() = true;
                for (source, target) in &self.bindings {
                    if source == name {
                        if let Some(target_prop) = self.properties.get(target) {
                            target_prop.borrow_mut().set(value.clone());
                        }
                    } else if target == name {
                        if let Some(source_prop) = self.properties.get(source) {
                            source_prop.borrow_mut().set(value.clone());
                        }
                    }
                }
                *self.syncing.borrow_mut() = false;
            }
            Some(result)
        } else {
            None
        }
    }

    /// Create a two-way binding between two properties.
    pub fn bind(&mut self, source: impl Into<String>, target: impl Into<String>) {
        let source = source.into();
        let target = target.into();

        // Sync initial value from source to target
        if let Some(val) = self.get_value(&source) {
            if let Some(target_prop) = self.properties.get(&target) {
                target_prop.borrow_mut().set(val);
            }
        }

        self.bindings.push((source, target));
    }

    /// Remove a binding between two properties.
    pub fn unbind(&mut self, source: &str, target: &str) {
        self.bindings
            .retain(|(s, t)| !(s == source && t == target));
    }

    /// Number of active bindings.
    pub fn binding_count(&self) -> usize {
        self.bindings.len()
    }

    /// Number of registered properties.
    pub fn property_count(&self) -> usize {
        self.properties.len()
    }
}

// ── Tests ────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn property_get_set() {
        let mut prop = Property::new(42);
        assert_eq!(*prop.get(), 42);
        prop.set(100);
        assert_eq!(*prop.get(), 100);
    }

    #[test]
    fn property_change_notification() {
        let changes = Rc::new(RefCell::new(Vec::new()));
        let c = changes.clone();
        let mut prop = Property::new(0);
        prop.on_change(move |old, new| {
            c.borrow_mut().push((*old, *new));
        });
        prop.set(5);
        prop.set(10);
        assert_eq!(*changes.borrow(), vec![(0, 5), (5, 10)]);
    }

    #[test]
    fn property_no_change_no_notification() {
        let count = Rc::new(RefCell::new(0));
        let c = count.clone();
        let mut prop = Property::new(42);
        prop.on_change(move |_, _| {
            *c.borrow_mut() += 1;
        });
        prop.set(42); // same value
        assert_eq!(*count.borrow(), 0);
    }

    #[test]
    fn property_validation_blocks_set() {
        let mut prop = Property::new(0);
        prop.add_validator(|v| {
            if *v < 0 {
                ValidationResult::Invalid("must be non-negative".into())
            } else {
                ValidationResult::Valid
            }
        });
        assert!(prop.set(5).is_valid());
        assert_eq!(*prop.get(), 5);

        assert!(!prop.set(-1).is_valid());
        assert_eq!(*prop.get(), 5); // unchanged
    }

    #[test]
    fn property_validation_state() {
        let mut prop = Property::new(0);
        prop.add_validator(|v| {
            if *v > 100 {
                ValidationResult::Invalid("too large".into())
            } else {
                ValidationResult::Valid
            }
        });
        prop.set(200);
        assert!(!prop.validation_state().is_valid());
        assert_eq!(
            prop.validation_state().error_message(),
            Some("too large")
        );
    }

    #[test]
    fn off_change_removes_listener() {
        let count = Rc::new(RefCell::new(0));
        let c = count.clone();
        let mut prop = Property::new(0);
        let id = prop.on_change(move |_, _| {
            *c.borrow_mut() += 1;
        });
        prop.set(1);
        prop.off_change(id);
        prop.set(2);
        assert_eq!(*count.borrow(), 1);
    }

    #[test]
    fn converter_forward_backward() {
        let conv = Converter::new(
            |v: &i32| v.to_string(),
            |s: &String| s.parse::<i32>().unwrap_or(0),
        );
        assert_eq!(conv.convert(&42), "42");
        assert_eq!(conv.convert_back(&"99".to_string()), 99);
    }

    #[test]
    fn computed_property() {
        let counter = Rc::new(RefCell::new(5));
        let c = counter.clone();
        let mut computed = ComputedProperty::new(move || *c.borrow() * 2);
        assert_eq!(computed.get(), 10);
        assert!(!computed.is_dirty());

        *counter.borrow_mut() = 10;
        computed.invalidate();
        assert!(computed.is_dirty());
        assert_eq!(computed.get(), 20);
    }

    #[test]
    fn computed_property_caches() {
        let call_count = Rc::new(RefCell::new(0));
        let c = call_count.clone();
        let mut computed = ComputedProperty::new(move || {
            *c.borrow_mut() += 1;
            42
        });
        computed.get();
        computed.get();
        assert_eq!(*call_count.borrow(), 1); // only computed once
    }

    #[test]
    fn binding_expression_evaluate() {
        let expr = BindingExpression::new("Hello, {name}! You are {age} years old.");
        let mut ctx = HashMap::new();
        ctx.insert("name".into(), "Alice".into());
        ctx.insert("age".into(), "30".into());
        assert_eq!(
            expr.evaluate(&ctx),
            "Hello, Alice! You are 30 years old."
        );
    }

    #[test]
    fn binding_expression_keys() {
        let expr = BindingExpression::new("{first} {last} ({email})");
        let mut keys = expr.keys();
        keys.sort();
        assert_eq!(keys, vec!["email", "first", "last"]);
    }

    #[test]
    fn binding_group_apply() {
        let mut group = BindingGroup::new("test");
        group.add_binding("source", "target");
        let mut values = HashMap::new();
        values.insert("source".into(), "hello".into());
        values.insert("target".into(), "".into());
        group.apply(&mut values);
        assert_eq!(values.get("target"), Some(&"hello".to_string()));
    }

    #[test]
    fn binding_group_disable() {
        let mut group = BindingGroup::new("test");
        group.add_binding("source", "target");
        group.disable();
        let mut values = HashMap::new();
        values.insert("source".into(), "hello".into());
        values.insert("target".into(), "original".into());
        group.apply(&mut values);
        assert_eq!(values.get("target"), Some(&"original".to_string()));
    }

    #[test]
    fn binding_group_metadata() {
        let mut group = BindingGroup::new("form");
        assert!(group.is_empty());
        group.add_binding("a", "b");
        assert_eq!(group.len(), 1);
        assert_eq!(group.name(), "form");
        let pairs = group.binding_pairs();
        assert_eq!(pairs, vec![("a".into(), "b".into())]);
    }

    #[test]
    fn binding_engine_two_way() {
        let mut engine = BindingEngine::new();
        engine.register("name", "Alice");
        engine.register("display", "");
        engine.bind("name", "display");

        // Source should have propagated to target
        assert_eq!(engine.get_value("display"), Some("Alice".to_string()));

        // Update source
        engine.set_value("name", "Bob");
        assert_eq!(engine.get_value("display"), Some("Bob".to_string()));

        // Update target (two-way)
        engine.set_value("display", "Charlie");
        assert_eq!(engine.get_value("name"), Some("Charlie".to_string()));
    }

    #[test]
    fn binding_engine_unbind() {
        let mut engine = BindingEngine::new();
        engine.register("a", "1");
        engine.register("b", "2");
        engine.bind("a", "b");
        assert_eq!(engine.binding_count(), 1);
        engine.unbind("a", "b");
        assert_eq!(engine.binding_count(), 0);

        // After unbind, changes should not propagate
        engine.set_value("a", "99");
        assert_eq!(engine.get_value("b"), Some("1".to_string()));
    }

    #[test]
    fn binding_engine_counts() {
        let mut engine = BindingEngine::new();
        engine.register("a", "");
        engine.register("b", "");
        engine.register("c", "");
        assert_eq!(engine.property_count(), 3);
        engine.bind("a", "b");
        engine.bind("b", "c");
        assert_eq!(engine.binding_count(), 2);
    }
}
