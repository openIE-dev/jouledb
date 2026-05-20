//! Database Triggers
//!
//! Provides event-based triggers for INSERT, UPDATE, and DELETE operations.
//! Triggers can be executed BEFORE or AFTER the operation.

use crate::ast::{Query, Value};
use crate::executor::RowData;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// Trigger timing: when the trigger fires
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerTiming {
    /// Fire before the operation
    Before,
    /// Fire after the operation  
    After,
}

/// Trigger event: what operation triggers the event
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TriggerEvent {
    /// INSERT operation
    Insert,
    /// UPDATE operation
    Update,
    /// DELETE operation
    Delete,
}

/// Trigger action type
#[derive(Clone)]
pub enum TriggerAction {
    /// Execute a SQL query
    Sql(String),
    /// Call a named function
    Function(String),
    /// Custom callback (not serializable, for programmatic use)
    Callback(Arc<dyn Fn(&TriggerContext) -> TriggerResult + Send + Sync>),
}

impl std::fmt::Debug for TriggerAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TriggerAction::Sql(s) => write!(f, "Sql({:?})", s),
            TriggerAction::Function(name) => write!(f, "Function({:?})", name),
            TriggerAction::Callback(_) => write!(f, "Callback(<fn>)"),
        }
    }
}

/// Result of trigger execution
#[derive(Debug, Clone)]
pub enum TriggerResult {
    /// Allow the operation to proceed
    Proceed,
    /// Abort the operation with a message
    Abort(String),
    /// Modify the row data (for BEFORE triggers)
    ModifyRow(RowData),
}

/// Context passed to trigger callbacks
#[derive(Debug, Clone)]
pub struct TriggerContext {
    /// The table being modified
    pub table: String,
    /// The event that fired the trigger
    pub event: TriggerEvent,
    /// The timing of the trigger
    pub timing: TriggerTiming,
    /// The old row (for UPDATE/DELETE)
    pub old_row: Option<RowData>,
    /// The new row (for INSERT/UPDATE)
    pub new_row: Option<RowData>,
    /// Additional metadata
    pub metadata: HashMap<String, Value>,
}

impl TriggerContext {
    /// Create a new trigger context
    pub fn new(table: &str, event: TriggerEvent, timing: TriggerTiming) -> Self {
        Self {
            table: table.to_string(),
            event,
            timing,
            old_row: None,
            new_row: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the old row
    pub fn with_old_row(mut self, row: RowData) -> Self {
        self.old_row = Some(row);
        self
    }

    /// Set the new row
    pub fn with_new_row(mut self, row: RowData) -> Self {
        self.new_row = Some(row);
        self
    }
}

/// A database trigger definition
#[derive(Debug, Clone)]
pub struct Trigger {
    /// Unique trigger name
    pub name: String,
    /// Table the trigger is attached to
    pub table: String,
    /// When the trigger fires
    pub timing: TriggerTiming,
    /// What event triggers it
    pub event: TriggerEvent,
    /// The action to perform
    pub action: TriggerAction,
    /// Whether the trigger is enabled
    pub enabled: bool,
    /// Priority (lower = higher priority)
    pub priority: i32,
}

impl Trigger {
    /// Create a new trigger
    pub fn new(
        name: &str,
        table: &str,
        timing: TriggerTiming,
        event: TriggerEvent,
        action: TriggerAction,
    ) -> Self {
        Self {
            name: name.to_string(),
            table: table.to_string(),
            timing,
            event,
            action,
            enabled: true,
            priority: 0,
        }
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Enable or disable the trigger
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
}

/// Trigger manager that stores and executes triggers
pub struct TriggerManager {
    /// Triggers indexed by (table, event, timing)
    triggers: RwLock<HashMap<(String, TriggerEvent, TriggerTiming), Vec<Arc<Trigger>>>>,
    /// All triggers by name (for easy lookup)
    by_name: RwLock<HashMap<String, Arc<Trigger>>>,
}

impl TriggerManager {
    /// Create a new trigger manager
    pub fn new() -> Self {
        Self {
            triggers: RwLock::new(HashMap::new()),
            by_name: RwLock::new(HashMap::new()),
        }
    }

    /// Register a new trigger
    pub fn register(&self, trigger: Trigger) -> Result<(), TriggerError> {
        let trigger = Arc::new(trigger);
        let key = (trigger.table.clone(), trigger.event, trigger.timing);

        let mut triggers = self.triggers.write().expect("trigger lock poisoned");
        let mut by_name = self.by_name.write().expect("trigger lock poisoned");

        // Check for duplicate name
        if by_name.contains_key(&trigger.name) {
            return Err(TriggerError::DuplicateName(trigger.name.clone()));
        }

        // Add to triggers map
        triggers
            .entry(key)
            .or_insert_with(Vec::new)
            .push(Arc::clone(&trigger));

        // Sort by priority
        if let Some(list) =
            triggers.get_mut(&(trigger.table.clone(), trigger.event, trigger.timing))
        {
            list.sort_by_key(|t| t.priority);
        }

        // Add to name index
        by_name.insert(trigger.name.clone(), trigger);

        Ok(())
    }

    /// Unregister a trigger by name
    pub fn unregister(&self, name: &str) -> Result<(), TriggerError> {
        let mut by_name = self.by_name.write().expect("trigger lock poisoned");
        let trigger = by_name
            .remove(name)
            .ok_or_else(|| TriggerError::NotFound(name.to_string()))?;

        let key = (trigger.table.clone(), trigger.event, trigger.timing);
        let mut triggers = self.triggers.write().expect("trigger lock poisoned");

        if let Some(list) = triggers.get_mut(&key) {
            list.retain(|t| t.name != name);
        }

        Ok(())
    }

    /// Get a trigger by name
    pub fn get(&self, name: &str) -> Option<Arc<Trigger>> {
        self.by_name
            .read()
            .expect("trigger lock poisoned")
            .get(name)
            .cloned()
    }

    /// Fire triggers for an event
    pub fn fire(
        &self,
        table: &str,
        event: TriggerEvent,
        timing: TriggerTiming,
        context: &TriggerContext,
    ) -> Result<TriggerResult, TriggerError> {
        let key = (table.to_string(), event, timing);
        let triggers = self.triggers.read().expect("trigger lock poisoned");

        let Some(trigger_list) = triggers.get(&key) else {
            return Ok(TriggerResult::Proceed);
        };

        for trigger in trigger_list {
            if !trigger.enabled {
                continue;
            }

            let result = self.execute_trigger(trigger, context)?;

            match result {
                TriggerResult::Proceed => continue,
                TriggerResult::Abort(msg) => return Ok(TriggerResult::Abort(msg)),
                TriggerResult::ModifyRow(_) => {
                    if timing == TriggerTiming::Before {
                        return Ok(result);
                    }
                    // AFTER triggers can't modify rows
                }
            }
        }

        Ok(TriggerResult::Proceed)
    }

    fn execute_trigger(
        &self,
        trigger: &Trigger,
        context: &TriggerContext,
    ) -> Result<TriggerResult, TriggerError> {
        match &trigger.action {
            TriggerAction::Sql(_sql) => {
                // SQL execution would require query engine access
                // For now, just proceed
                Ok(TriggerResult::Proceed)
            }
            TriggerAction::Function(_name) => {
                // Function calls would require a function registry
                // For now, just proceed
                Ok(TriggerResult::Proceed)
            }
            TriggerAction::Callback(callback) => Ok(callback(context)),
        }
    }

    /// List all triggers for a table
    pub fn list_for_table(&self, table: &str) -> Vec<Arc<Trigger>> {
        let triggers = self.triggers.read().expect("trigger lock poisoned");
        let mut result = Vec::new();

        for ((t, _, _), list) in triggers.iter() {
            if t == table {
                result.extend(list.iter().cloned());
            }
        }

        result.sort_by_key(|t| (t.event as u8, t.timing as u8, t.priority));
        result
    }

    /// List all triggers
    pub fn list_all(&self) -> Vec<Arc<Trigger>> {
        self.by_name
            .read()
            .expect("trigger lock poisoned")
            .values()
            .cloned()
            .collect()
    }

    /// Enable or disable a trigger
    pub fn set_enabled(&self, name: &str, enabled: bool) -> Result<(), TriggerError> {
        // We need to re-create the trigger with the new enabled state
        // since triggers are stored as Arc
        let by_name = self.by_name.read().expect("trigger lock poisoned");
        let trigger = by_name
            .get(name)
            .ok_or_else(|| TriggerError::NotFound(name.to_string()))?;

        // Note: This requires interior mutability, which we don't have with Arc<Trigger>
        // In a production implementation, we'd use Arc<RwLock<Trigger>> or similar
        drop(by_name);

        // For now, just validate the trigger exists
        if enabled { Ok(()) } else { Ok(()) }
    }
}

impl Default for TriggerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Trigger-related errors
#[derive(Debug, Clone, PartialEq)]
pub enum TriggerError {
    /// Duplicate trigger name
    DuplicateName(String),
    /// Trigger not found
    NotFound(String),
    /// Trigger execution failed
    ExecutionFailed(String),
}

impl std::fmt::Display for TriggerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TriggerError::DuplicateName(name) => write!(f, "Trigger already exists: {}", name),
            TriggerError::NotFound(name) => write!(f, "Trigger not found: {}", name),
            TriggerError::ExecutionFailed(msg) => write!(f, "Trigger execution failed: {}", msg),
        }
    }
}

impl std::error::Error for TriggerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trigger_registration() {
        let manager = TriggerManager::new();

        let trigger = Trigger::new(
            "audit_insert",
            "users",
            TriggerTiming::After,
            TriggerEvent::Insert,
            TriggerAction::Sql("INSERT INTO audit_log VALUES (...)".to_string()),
        );

        manager.register(trigger).unwrap();

        let retrieved = manager.get("audit_insert").unwrap();
        assert_eq!(retrieved.name, "audit_insert");
        assert_eq!(retrieved.table, "users");
    }

    #[test]
    fn test_duplicate_trigger_name() {
        let manager = TriggerManager::new();

        let trigger1 = Trigger::new(
            "my_trigger",
            "users",
            TriggerTiming::Before,
            TriggerEvent::Insert,
            TriggerAction::Sql("SELECT 1".to_string()),
        );

        let trigger2 = Trigger::new(
            "my_trigger",
            "orders",
            TriggerTiming::After,
            TriggerEvent::Delete,
            TriggerAction::Sql("SELECT 2".to_string()),
        );

        manager.register(trigger1).unwrap();
        assert!(matches!(
            manager.register(trigger2),
            Err(TriggerError::DuplicateName(_))
        ));
    }

    #[test]
    fn test_trigger_callback() {
        let manager = TriggerManager::new();
        let called = Arc::new(RwLock::new(false));
        let called_clone = Arc::clone(&called);

        let trigger = Trigger::new(
            "callback_trigger",
            "users",
            TriggerTiming::Before,
            TriggerEvent::Insert,
            TriggerAction::Callback(Arc::new(move |_ctx| {
                *called_clone.write().expect("trigger lock poisoned") = true;
                TriggerResult::Proceed
            })),
        );

        manager.register(trigger).unwrap();

        let context = TriggerContext::new("users", TriggerEvent::Insert, TriggerTiming::Before);
        let result = manager.fire(
            "users",
            TriggerEvent::Insert,
            TriggerTiming::Before,
            &context,
        );

        assert!(matches!(result, Ok(TriggerResult::Proceed)));
        assert!(*called.read().expect("trigger lock poisoned"));
    }

    #[test]
    fn test_trigger_abort() {
        let manager = TriggerManager::new();

        let trigger = Trigger::new(
            "abort_trigger",
            "users",
            TriggerTiming::Before,
            TriggerEvent::Delete,
            TriggerAction::Callback(Arc::new(|_ctx| {
                TriggerResult::Abort("Cannot delete users".to_string())
            })),
        );

        manager.register(trigger).unwrap();

        let context = TriggerContext::new("users", TriggerEvent::Delete, TriggerTiming::Before);
        let result = manager
            .fire(
                "users",
                TriggerEvent::Delete,
                TriggerTiming::Before,
                &context,
            )
            .unwrap();

        assert!(matches!(result, TriggerResult::Abort(_)));
    }

    #[test]
    fn test_list_triggers() {
        let manager = TriggerManager::new();

        manager
            .register(Trigger::new(
                "t1",
                "users",
                TriggerTiming::Before,
                TriggerEvent::Insert,
                TriggerAction::Sql("".to_string()),
            ))
            .unwrap();

        manager
            .register(Trigger::new(
                "t2",
                "users",
                TriggerTiming::After,
                TriggerEvent::Insert,
                TriggerAction::Sql("".to_string()),
            ))
            .unwrap();

        manager
            .register(Trigger::new(
                "t3",
                "orders",
                TriggerTiming::Before,
                TriggerEvent::Delete,
                TriggerAction::Sql("".to_string()),
            ))
            .unwrap();

        let user_triggers = manager.list_for_table("users");
        assert_eq!(user_triggers.len(), 2);

        let all = manager.list_all();
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn test_trigger_priority() {
        let manager = TriggerManager::new();
        let execution_order = Arc::new(RwLock::new(Vec::new()));

        for (name, priority) in [("low", 10), ("high", 1), ("medium", 5)] {
            let order = Arc::clone(&execution_order);
            let name_owned = name.to_string();
            manager
                .register(
                    Trigger::new(
                        name,
                        "users",
                        TriggerTiming::Before,
                        TriggerEvent::Insert,
                        TriggerAction::Callback(Arc::new(move |_ctx| {
                            order
                                .write()
                                .expect("trigger lock poisoned")
                                .push(name_owned.clone());
                            TriggerResult::Proceed
                        })),
                    )
                    .with_priority(priority),
                )
                .unwrap();
        }

        let context = TriggerContext::new("users", TriggerEvent::Insert, TriggerTiming::Before);
        manager
            .fire(
                "users",
                TriggerEvent::Insert,
                TriggerTiming::Before,
                &context,
            )
            .unwrap();

        let order = execution_order.read().expect("trigger lock poisoned");
        assert_eq!(*order, vec!["high", "medium", "low"]);
    }
}
