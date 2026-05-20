//! Database Triggers
//!
//! Implements database triggers for event-driven architectures.
//! Supports ON INSERT/UPDATE/DELETE triggers with:
//! - Wasm function execution (via wasmtime)
//! - Webhook dispatch
//!
//! # Wasm Trigger Functions
//! Wasm modules can export functions that receive trigger context and can:
//! - Read from the database (`db_get`)
//! - Write to the database (`db_put`)
//! - Log messages (`log_message`)

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

#[cfg(feature = "wasm-triggers")]
use wasmtime::*;

/// Trigger event type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerEventType {
    /// ON INSERT
    Insert,
    /// ON UPDATE
    Update,
    /// ON DELETE
    Delete,
    /// ON ALL (insert, update, delete)
    All,
}

/// Trigger execution context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerContext {
    /// Operation type
    pub operation: TriggerEventType,
    /// Table name
    pub table: String,
    /// Old row (for UPDATE/DELETE)
    pub old_row: Option<HashMap<String, serde_json::Value>>,
    /// New row (for INSERT/UPDATE)
    pub new_row: Option<HashMap<String, serde_json::Value>>,
    /// Trigger name
    pub trigger_name: String,
}

/// Trigger definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trigger {
    /// Trigger name
    pub name: String,
    /// Table name
    pub table: String,
    /// Event type
    pub event_type: TriggerEventType,
    /// Wasm module bytes (optional)
    pub wasm_module: Option<Vec<u8>>,
    /// Wasm function name
    pub wasm_function: Option<String>,
    /// Webhook URL (optional)
    pub webhook_url: Option<String>,
    /// Enabled
    pub enabled: bool,
}

impl Trigger {
    /// Create new trigger
    pub fn new(name: String, table: String, event_type: TriggerEventType) -> Self {
        Self {
            name,
            table,
            event_type,
            wasm_module: None,
            wasm_function: None,
            webhook_url: None,
            enabled: true,
        }
    }

    /// Set Wasm function
    pub fn with_wasm(mut self, module: Vec<u8>, function: String) -> Self {
        self.wasm_module = Some(module);
        self.wasm_function = Some(function);
        self
    }

    /// Set webhook URL
    pub fn with_webhook(mut self, url: String) -> Self {
        self.webhook_url = Some(url);
        self
    }

    /// Check if trigger matches event
    pub fn matches(&self, operation: TriggerEventType, table: &str) -> bool {
        if !self.enabled {
            return false;
        }

        if self.table != table {
            return false;
        }

        matches!(
            (self.event_type, operation),
            (TriggerEventType::All, _)
                | (TriggerEventType::Insert, TriggerEventType::Insert)
                | (TriggerEventType::Update, TriggerEventType::Update)
                | (TriggerEventType::Delete, TriggerEventType::Delete)
        )
    }
}

// ============================================================================
// Wasm Executor
// ============================================================================

/// Host state for Wasm trigger execution
#[cfg(feature = "wasm-triggers")]
pub struct WasmHostState {
    /// Trigger context (serialized JSON)
    pub context_json: String,
    /// Result from trigger execution
    pub result: Option<String>,
    /// Log messages
    pub logs: Vec<String>,
    /// Database operations requested by Wasm
    pub db_operations: Vec<DbOperation>,
}

/// Database operation requested by Wasm trigger
#[cfg(feature = "wasm-triggers")]
#[derive(Debug, Clone)]
pub enum DbOperation {
    Get { key: String },
    Put { key: String, value: String },
    Delete { key: String },
}

/// Wasm trigger executor
#[cfg(feature = "wasm-triggers")]
pub struct WasmExecutor {
    /// Wasmtime engine (shared for performance)
    engine: Engine,
    /// Module cache (keyed by trigger name)
    module_cache: Arc<RwLock<HashMap<String, Module>>>,
}

#[cfg(feature = "wasm-triggers")]
impl WasmExecutor {
    /// Create a new Wasm executor
    pub fn new() -> Result<Self, String> {
        let mut config = Config::new();
        config.async_support(true);
        config.consume_fuel(true); // Enable fuel for execution limits

        let engine =
            Engine::new(&config).map_err(|e| format!("Failed to create Wasm engine: {}", e))?;

        Ok(Self {
            engine,
            module_cache: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Compile and cache a Wasm module
    pub async fn compile_module(&self, name: &str, wasm_bytes: &[u8]) -> Result<(), String> {
        let module = Module::new(&self.engine, wasm_bytes)
            .map_err(|e| format!("Failed to compile Wasm module: {}", e))?;

        let mut cache = self.module_cache.write().await;
        cache.insert(name.to_string(), module);

        Ok(())
    }

    /// Execute a trigger function
    pub async fn execute(
        &self,
        module_name: &str,
        function_name: &str,
        context: &TriggerContext,
    ) -> Result<TriggerResult, String> {
        // Get cached module or compile
        let cache = self.module_cache.read().await;
        let module = cache
            .get(module_name)
            .ok_or_else(|| format!("Module '{}' not found in cache", module_name))?
            .clone();
        drop(cache);

        // Serialize context to JSON
        let context_json = serde_json::to_string(context)
            .map_err(|e| format!("Failed to serialize context: {}", e))?;

        // Create host state
        let host_state = WasmHostState {
            context_json,
            result: None,
            logs: Vec::new(),
            db_operations: Vec::new(),
        };

        // Create store with fuel limit
        let mut store = Store::new(&self.engine, host_state);
        store.set_fuel(1_000_000).unwrap(); // 1M instructions limit

        // Create linker with host functions
        let mut linker = Linker::new(&self.engine);
        Self::define_host_functions(&mut linker)?;

        // Instantiate module
        let instance = linker
            .instantiate_async(&mut store, &module)
            .await
            .map_err(|e| format!("Failed to instantiate module: {}", e))?;

        // Get exported function
        let trigger_fn = instance
            .get_typed_func::<(), i32>(&mut store, function_name)
            .map_err(|e| format!("Function '{}' not found: {}", function_name, e))?;

        // Execute trigger
        let result_code = trigger_fn
            .call_async(&mut store, ())
            .await
            .map_err(|e| format!("Trigger execution failed: {}", e))?;

        // Get result from store
        let state = store.data();

        Ok(TriggerResult {
            trigger_name: module_name.to_string(),
            success: result_code == 0,
            error: if result_code != 0 {
                Some(format!("Trigger returned error code: {}", result_code))
            } else {
                None
            },
        })
    }

    /// Define host functions available to Wasm modules
    fn define_host_functions(linker: &mut Linker<WasmHostState>) -> Result<(), String> {
        // log_message(ptr: i32, len: i32)
        linker
            .func_wrap(
                "env",
                "log_message",
                |mut caller: Caller<'_, WasmHostState>, ptr: i32, len: i32| {
                    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
                    if let Some(memory) = memory {
                        let msg = {
                            let data = memory.data(&caller);
                            data.get(ptr as usize..(ptr + len) as usize)
                                .and_then(|slice| std::str::from_utf8(slice).ok())
                                .map(|s| s.to_string())
                        };
                        if let Some(msg) = msg {
                            caller.data_mut().logs.push(msg.clone());
                            tracing::info!(target: "wasm_trigger", "{}", msg);
                        }
                    }
                },
            )
            .map_err(|e| format!("Failed to define log_message: {}", e))?;

        // get_context_len() -> i32
        linker
            .func_wrap(
                "env",
                "get_context_len",
                |caller: Caller<'_, WasmHostState>| -> i32 {
                    caller.data().context_json.len() as i32
                },
            )
            .map_err(|e| format!("Failed to define get_context_len: {}", e))?;

        // get_context(ptr: i32)
        linker
            .func_wrap(
                "env",
                "get_context",
                |mut caller: Caller<'_, WasmHostState>, ptr: i32| {
                    let json = caller.data().context_json.clone();
                    if let Some(memory) = caller.get_export("memory") {
                        if let Some(memory) = memory.into_memory() {
                            let data = memory.data_mut(&mut caller);
                            if let Some(slice) =
                                data.get_mut(ptr as usize..(ptr as usize + json.len()))
                            {
                                slice.copy_from_slice(json.as_bytes());
                            }
                        }
                    }
                },
            )
            .map_err(|e| format!("Failed to define get_context: {}", e))?;

        // set_result(ptr: i32, len: i32)
        linker
            .func_wrap(
                "env",
                "set_result",
                |mut caller: Caller<'_, WasmHostState>, ptr: i32, len: i32| {
                    if let Some(memory) = caller.get_export("memory") {
                        if let Some(memory) = memory.into_memory() {
                            let data = memory.data(&caller);
                            if let Some(slice) = data.get(ptr as usize..(ptr + len) as usize) {
                                if let Ok(result) = std::str::from_utf8(slice) {
                                    caller.data_mut().result = Some(result.to_string());
                                }
                            }
                        }
                    }
                },
            )
            .map_err(|e| format!("Failed to define set_result: {}", e))?;

        // db_get(key_ptr: i32, key_len: i32, value_ptr: i32, max_len: i32) -> i32
        linker
            .func_wrap(
                "env",
                "db_get",
                |mut caller: Caller<'_, WasmHostState>,
                 key_ptr: i32,
                 key_len: i32,
                 _value_ptr: i32,
                 _max_len: i32|
                 -> i32 {
                    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
                    if let Some(memory) = memory {
                        let key = {
                            let data = memory.data(&caller);
                            data.get(key_ptr as usize..(key_ptr + key_len) as usize)
                                .and_then(|slice| std::str::from_utf8(slice).ok())
                                .map(|s| s.to_string())
                        };
                        if let Some(key) = key {
                            caller
                                .data_mut()
                                .db_operations
                                .push(DbOperation::Get { key });
                            return 0;
                        }
                    }
                    -1
                },
            )
            .map_err(|e| format!("Failed to define db_get: {}", e))?;

        // db_put(key_ptr: i32, key_len: i32, value_ptr: i32, value_len: i32) -> i32
        linker
            .func_wrap(
                "env",
                "db_put",
                |mut caller: Caller<'_, WasmHostState>,
                 key_ptr: i32,
                 key_len: i32,
                 value_ptr: i32,
                 value_len: i32|
                 -> i32 {
                    if let Some(memory) = caller.get_export("memory") {
                        if let Some(memory) = memory.into_memory() {
                            let data = memory.data(&caller);
                            let key = data
                                .get(key_ptr as usize..(key_ptr + key_len) as usize)
                                .and_then(|s| std::str::from_utf8(s).ok())
                                .map(|s| s.to_string());
                            let value = data
                                .get(value_ptr as usize..(value_ptr + value_len) as usize)
                                .and_then(|s| std::str::from_utf8(s).ok())
                                .map(|s| s.to_string());

                            if let (Some(k), Some(v)) = (key, value) {
                                caller
                                    .data_mut()
                                    .db_operations
                                    .push(DbOperation::Put { key: k, value: v });
                                return 0;
                            }
                        }
                    }
                    -1
                },
            )
            .map_err(|e| format!("Failed to define db_put: {}", e))?;

        // db_delete(key_ptr: i32, key_len: i32) -> i32
        linker
            .func_wrap(
                "env",
                "db_delete",
                |mut caller: Caller<'_, WasmHostState>, key_ptr: i32, key_len: i32| -> i32 {
                    let memory = caller.get_export("memory").and_then(|e| e.into_memory());
                    if let Some(memory) = memory {
                        let key = {
                            let data = memory.data(&caller);
                            data.get(key_ptr as usize..(key_ptr + key_len) as usize)
                                .and_then(|slice| std::str::from_utf8(slice).ok())
                                .map(|s| s.to_string())
                        };
                        if let Some(key) = key {
                            caller
                                .data_mut()
                                .db_operations
                                .push(DbOperation::Delete { key });
                            return 0;
                        }
                    }
                    -1
                },
            )
            .map_err(|e| format!("Failed to define db_delete: {}", e))?;

        Ok(())
    }
}

#[cfg(feature = "wasm-triggers")]
impl Default for WasmExecutor {
    fn default() -> Self {
        Self::new().expect("Failed to create default WasmExecutor")
    }
}

// ============================================================================
// Trigger Manager
// ============================================================================

/// Trigger manager
pub struct TriggerManager {
    /// Triggers by table
    triggers: Arc<RwLock<HashMap<String, Vec<Trigger>>>>,
    /// Wasm executor
    #[cfg(feature = "wasm-triggers")]
    wasm_executor: Option<WasmExecutor>,
    #[cfg(not(feature = "wasm-triggers"))]
    #[allow(dead_code)]
    wasm_executor: Option<()>,
}

impl TriggerManager {
    /// Create new trigger manager
    pub fn new() -> Self {
        Self {
            triggers: Arc::new(RwLock::new(HashMap::new())),
            #[cfg(feature = "wasm-triggers")]
            wasm_executor: WasmExecutor::new().ok(),
            #[cfg(not(feature = "wasm-triggers"))]
            wasm_executor: None,
        }
    }

    /// Create trigger manager with Wasm support explicitly enabled/disabled
    #[cfg(feature = "wasm-triggers")]
    pub fn with_wasm(enable_wasm: bool) -> Self {
        Self {
            triggers: Arc::new(RwLock::new(HashMap::new())),
            wasm_executor: if enable_wasm {
                WasmExecutor::new().ok()
            } else {
                None
            },
        }
    }

    /// Register trigger
    pub async fn register_trigger(&self, trigger: Trigger) {
        // Compile Wasm module if present
        #[cfg(feature = "wasm-triggers")]
        if let (Some(executor), Some(wasm_bytes)) = (&self.wasm_executor, &trigger.wasm_module) {
            if let Err(e) = executor.compile_module(&trigger.name, wasm_bytes).await {
                tracing::error!("Failed to compile Wasm trigger '{}': {}", trigger.name, e);
            }
        }

        let mut triggers = self.triggers.write().await;
        triggers
            .entry(trigger.table.clone())
            .or_insert_with(Vec::new)
            .push(trigger);
    }

    /// Unregister trigger
    pub async fn unregister_trigger(&self, table: &str, trigger_name: &str) -> bool {
        let mut triggers = self.triggers.write().await;
        if let Some(table_triggers) = triggers.get_mut(table) {
            let initial_len = table_triggers.len();
            table_triggers.retain(|t| t.name != trigger_name);
            table_triggers.len() < initial_len
        } else {
            false
        }
    }

    /// Execute triggers for an event
    pub async fn execute_triggers(
        &self,
        operation: TriggerEventType,
        table: &str,
        old_row: Option<HashMap<String, serde_json::Value>>,
        new_row: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<TriggerResult>, String> {
        let triggers = self.triggers.read().await;
        let table_triggers = triggers.get(table).cloned().unwrap_or_default();

        let mut results = Vec::new();

        for trigger in table_triggers {
            if trigger.matches(operation, table) {
                let context = TriggerContext {
                    operation,
                    table: table.to_string(),
                    old_row: old_row.clone(),
                    new_row: new_row.clone(),
                    trigger_name: trigger.name.clone(),
                };

                let result = self.execute_trigger(&trigger, &context).await;
                results.push(TriggerResult {
                    trigger_name: trigger.name,
                    success: result.is_ok(),
                    error: result.err(),
                });
            }
        }

        Ok(results)
    }

    /// Execute a single trigger
    async fn execute_trigger(
        &self,
        trigger: &Trigger,
        context: &TriggerContext,
    ) -> Result<(), String> {
        // Execute Wasm function if provided
        #[cfg(feature = "wasm-triggers")]
        if let (Some(executor), Some(_module), Some(function)) = (
            &self.wasm_executor,
            &trigger.wasm_module,
            &trigger.wasm_function,
        ) {
            let result = executor.execute(&trigger.name, function, context).await?;
            if !result.success {
                return Err(result.error.unwrap_or_else(|| "Unknown error".to_string()));
            }
        }

        #[cfg(not(feature = "wasm-triggers"))]
        if trigger.wasm_module.is_some() {
            tracing::warn!(
                "Wasm trigger '{}' defined but wasm-triggers feature not enabled",
                trigger.name
            );
        }

        // Dispatch webhook if provided
        if let Some(ref url) = trigger.webhook_url {
            self.dispatch_webhook(url, context).await?;
        }

        Ok(())
    }

    /// Dispatch webhook
    async fn dispatch_webhook(&self, url: &str, context: &TriggerContext) -> Result<(), String> {
        let client = reqwest::Client::new();
        let payload = serde_json::to_value(context)
            .map_err(|e| format!("Failed to serialize context: {}", e))?;

        client
            .post(url)
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Webhook request failed: {}", e))?
            .error_for_status()
            .map_err(|e| format!("Webhook returned error: {}", e))?;

        Ok(())
    }

    /// Get triggers for table
    pub async fn get_triggers(&self, table: &str) -> Vec<Trigger> {
        let triggers = self.triggers.read().await;
        triggers.get(table).cloned().unwrap_or_default()
    }

    /// Check if Wasm support is available
    pub fn wasm_available(&self) -> bool {
        #[cfg(feature = "wasm-triggers")]
        {
            self.wasm_executor.is_some()
        }
        #[cfg(not(feature = "wasm-triggers"))]
        {
            false
        }
    }
}

impl Default for TriggerManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Trigger execution result
#[derive(Debug, Clone)]
pub struct TriggerResult {
    /// Trigger name
    pub trigger_name: String,
    /// Success
    pub success: bool,
    /// Error message if failed
    pub error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_trigger_matching() {
        let trigger = Trigger::new(
            "test_trigger".to_string(),
            "users".to_string(),
            TriggerEventType::Insert,
        );

        assert!(trigger.matches(TriggerEventType::Insert, "users"));
        assert!(!trigger.matches(TriggerEventType::Update, "users"));
        assert!(!trigger.matches(TriggerEventType::Insert, "orders"));
    }

    #[tokio::test]
    async fn test_trigger_all_events() {
        let trigger = Trigger::new(
            "all_trigger".to_string(),
            "users".to_string(),
            TriggerEventType::All,
        );

        assert!(trigger.matches(TriggerEventType::Insert, "users"));
        assert!(trigger.matches(TriggerEventType::Update, "users"));
        assert!(trigger.matches(TriggerEventType::Delete, "users"));
    }

    #[tokio::test]
    async fn test_trigger_manager() {
        let manager = TriggerManager::new();

        let trigger = Trigger::new(
            "test".to_string(),
            "users".to_string(),
            TriggerEventType::Insert,
        )
        .with_webhook("http://example.com/hook".to_string());

        manager.register_trigger(trigger).await;

        let triggers = manager.get_triggers("users").await;
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].name, "test");
    }

    #[tokio::test]
    async fn test_unregister_trigger() {
        let manager = TriggerManager::new();

        manager
            .register_trigger(Trigger::new(
                "test1".to_string(),
                "users".to_string(),
                TriggerEventType::Insert,
            ))
            .await;

        manager
            .register_trigger(Trigger::new(
                "test2".to_string(),
                "users".to_string(),
                TriggerEventType::Update,
            ))
            .await;

        assert!(manager.unregister_trigger("users", "test1").await);
        assert!(!manager.unregister_trigger("users", "nonexistent").await);

        let triggers = manager.get_triggers("users").await;
        assert_eq!(triggers.len(), 1);
        assert_eq!(triggers[0].name, "test2");
    }

    #[test]
    fn test_wasm_available_check() {
        let manager = TriggerManager::new();
        #[cfg(feature = "wasm-triggers")]
        assert!(manager.wasm_available());
        #[cfg(not(feature = "wasm-triggers"))]
        assert!(!manager.wasm_available());
    }
}
