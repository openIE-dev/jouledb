//! Smart contract simulation — contract state (key-value), method invocation,
//! gas metering, contract deployment, event emission, inter-contract calls,
//! state rollback on failure, and contract ABI definition.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

/// Errors from smart contract operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContractError {
    /// Contract not found by address.
    NotFound(String),
    /// Method not found on contract.
    MethodNotFound { contract: String, method: String },
    /// Out of gas during execution.
    OutOfGas { used: u64, limit: u64 },
    /// Execution reverted with a reason.
    Reverted(String),
    /// Invalid argument passed to method.
    InvalidArgument(String),
    /// State key not found.
    KeyNotFound(String),
    /// Duplicate contract address.
    DuplicateAddress(String),
    /// Re-entrancy not allowed.
    ReentrancyGuard(String),
}

impl fmt::Display for ContractError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(addr) => write!(f, "contract not found: {addr}"),
            Self::MethodNotFound { contract, method } => {
                write!(f, "method {method} not found on contract {contract}")
            }
            Self::OutOfGas { used, limit } => {
                write!(f, "out of gas: used {used}, limit {limit}")
            }
            Self::Reverted(reason) => write!(f, "execution reverted: {reason}"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::KeyNotFound(key) => write!(f, "state key not found: {key}"),
            Self::DuplicateAddress(addr) => write!(f, "duplicate contract address: {addr}"),
            Self::ReentrancyGuard(addr) => write!(f, "reentrancy on contract: {addr}"),
        }
    }
}

impl std::error::Error for ContractError {}

// ── ABI ─────────────────────────────────────────────────────────────────────

/// Parameter type in a method signature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParamType {
    Uint,
    Int,
    Bool,
    Str,
    Address,
    Bytes,
    Json,
}

impl fmt::Display for ParamType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Uint => write!(f, "uint"),
            Self::Int => write!(f, "int"),
            Self::Bool => write!(f, "bool"),
            Self::Str => write!(f, "string"),
            Self::Address => write!(f, "address"),
            Self::Bytes => write!(f, "bytes"),
            Self::Json => write!(f, "json"),
        }
    }
}

/// A method parameter.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub param_type: ParamType,
}

/// A method in the contract ABI.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbiMethod {
    pub name: String,
    pub params: Vec<Param>,
    pub returns: Option<ParamType>,
    /// Whether this method mutates state.
    pub mutates: bool,
    /// Gas cost for calling this method.
    pub gas_cost: u64,
}

impl AbiMethod {
    /// Create a new ABI method definition.
    pub fn new(name: impl Into<String>, gas_cost: u64, mutates: bool) -> Self {
        Self {
            name: name.into(),
            params: Vec::new(),
            returns: None,
            mutates,
            gas_cost,
        }
    }

    /// Add a parameter.
    pub fn with_param(mut self, name: impl Into<String>, param_type: ParamType) -> Self {
        self.params.push(Param {
            name: name.into(),
            param_type,
        });
        self
    }

    /// Set the return type.
    pub fn with_return(mut self, ret: ParamType) -> Self {
        self.returns = Some(ret);
        self
    }
}

// ── Events ──────────────────────────────────────────────────────────────────

/// An event emitted by a contract during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContractEvent {
    /// Contract address that emitted the event.
    pub contract_address: String,
    /// Event name.
    pub name: String,
    /// Arbitrary event data.
    pub data: HashMap<String, Value>,
    /// Gas used at the time of emission.
    pub gas_used: u64,
}

// ── Contract ────────────────────────────────────────────────────────────────

/// A smart contract with key-value state and an ABI.
#[derive(Debug, Clone)]
pub struct Contract {
    /// Contract address (unique identifier).
    pub address: String,
    /// Human-readable name.
    pub name: String,
    /// The contract's ABI (method definitions).
    pub abi: Vec<AbiMethod>,
    /// Key-value state storage.
    state: HashMap<String, Value>,
    /// Total gas consumed across all invocations.
    pub total_gas_used: u64,
}

impl Contract {
    /// Create a new contract.
    pub fn new(address: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            address: address.into(),
            name: name.into(),
            abi: Vec::new(),
            state: HashMap::new(),
            total_gas_used: 0,
        }
    }

    /// Register a method in the ABI.
    pub fn add_method(&mut self, method: AbiMethod) {
        self.abi.push(method);
    }

    /// Get a method by name.
    pub fn get_method(&self, name: &str) -> Option<&AbiMethod> {
        self.abi.iter().find(|m| m.name == name)
    }

    /// Get a state value.
    pub fn get_state(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }

    /// Set a state value.
    pub fn set_state(&mut self, key: impl Into<String>, value: Value) {
        self.state.insert(key.into(), value);
    }

    /// Remove a state value.
    pub fn remove_state(&mut self, key: &str) -> Option<Value> {
        self.state.remove(key)
    }

    /// Get the number of state entries.
    pub fn state_count(&self) -> usize {
        self.state.len()
    }

    /// Clone the entire state (for snapshot/rollback).
    pub fn snapshot_state(&self) -> HashMap<String, Value> {
        self.state.clone()
    }

    /// Restore state from a snapshot.
    pub fn restore_state(&mut self, snapshot: HashMap<String, Value>) {
        self.state = snapshot;
    }
}

// ── Execution Context ───────────────────────────────────────────────────────

/// Result of a method invocation.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Return value (if any).
    pub return_value: Option<Value>,
    /// Gas consumed.
    pub gas_used: u64,
    /// Events emitted during execution.
    pub events: Vec<ContractEvent>,
    /// Whether execution succeeded.
    pub success: bool,
    /// Revert reason (if failed).
    pub revert_reason: Option<String>,
}

/// The contract runtime / virtual machine.
#[derive(Debug, Clone)]
pub struct ContractRuntime {
    /// Deployed contracts keyed by address.
    contracts: HashMap<String, Contract>,
    /// Global event log.
    event_log: Vec<ContractEvent>,
    /// Default gas limit per invocation.
    pub default_gas_limit: u64,
}

impl ContractRuntime {
    /// Create a new runtime.
    pub fn new(default_gas_limit: u64) -> Self {
        Self {
            contracts: HashMap::new(),
            event_log: Vec::new(),
            default_gas_limit,
        }
    }

    /// Deploy a contract.
    pub fn deploy(&mut self, contract: Contract) -> Result<(), ContractError> {
        if self.contracts.contains_key(&contract.address) {
            return Err(ContractError::DuplicateAddress(contract.address.clone()));
        }
        self.contracts.insert(contract.address.clone(), contract);
        Ok(())
    }

    /// Get a reference to a deployed contract.
    pub fn get_contract(&self, address: &str) -> Option<&Contract> {
        self.contracts.get(address)
    }

    /// Get a mutable reference to a deployed contract.
    pub fn get_contract_mut(&mut self, address: &str) -> Option<&mut Contract> {
        self.contracts.get_mut(address)
    }

    /// Number of deployed contracts.
    pub fn contract_count(&self) -> usize {
        self.contracts.len()
    }

    /// Get all events in the global log.
    pub fn events(&self) -> &[ContractEvent] {
        &self.event_log
    }

    /// Invoke a contract method.
    ///
    /// This handles gas metering, state snapshotting, rollback on failure,
    /// and event emission.
    pub fn invoke(
        &mut self,
        address: &str,
        method_name: &str,
        args: HashMap<String, Value>,
        gas_limit: Option<u64>,
    ) -> Result<ExecutionResult, ContractError> {
        if !self.contracts.contains_key(address) {
            return Err(ContractError::NotFound(address.to_string()));
        }

        let contract = self.contracts.get(address).unwrap();
        let method = contract
            .get_method(method_name)
            .ok_or_else(|| ContractError::MethodNotFound {
                contract: address.to_string(),
                method: method_name.to_string(),
            })?
            .clone();

        let limit = gas_limit.unwrap_or(self.default_gas_limit);

        if method.gas_cost > limit {
            return Err(ContractError::OutOfGas {
                used: method.gas_cost,
                limit,
            });
        }

        // Snapshot state before execution
        let contract = self.contracts.get(address).unwrap();
        let snapshot = contract.snapshot_state();
        let mut events = Vec::new();
        let mut gas_used = method.gas_cost;

        // Execute: Apply args to state if method mutates
        if method.mutates {
            let contract_mut = self.contracts.get_mut(address).unwrap();
            for (key, value) in &args {
                contract_mut.set_state(key.clone(), value.clone());
                // Each state write costs 10 gas
                gas_used += 10;
                if gas_used > limit {
                    // Rollback
                    contract_mut.restore_state(snapshot);
                    return Err(ContractError::OutOfGas {
                        used: gas_used,
                        limit,
                    });
                }
            }
        }

        // Emit an invocation event
        let mut event_data = HashMap::new();
        event_data.insert("method".to_string(), Value::String(method_name.to_string()));
        event_data.insert("gas_used".to_string(), Value::Number(serde_json::Number::from(gas_used)));

        let event = ContractEvent {
            contract_address: address.to_string(),
            name: format!("{method_name}_called"),
            data: event_data,
            gas_used,
        };
        events.push(event.clone());
        self.event_log.push(event);

        // Update total gas on the contract
        let contract_mut = self.contracts.get_mut(address).unwrap();
        contract_mut.total_gas_used += gas_used;

        // Build return value from read args or state
        let return_value = if !method.mutates {
            // For view methods, return the requested state value
            let contract = self.contracts.get(address).unwrap();
            if let Some(key) = args.get("key") {
                if let Some(key_str) = key.as_str() {
                    contract.get_state(key_str).cloned()
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(ExecutionResult {
            return_value,
            gas_used,
            events,
            success: true,
            revert_reason: None,
        })
    }

    /// Invoke a method and revert on failure.
    pub fn invoke_or_revert(
        &mut self,
        address: &str,
        method_name: &str,
        args: HashMap<String, Value>,
        gas_limit: Option<u64>,
        revert_condition: impl Fn(&HashMap<String, Value>) -> Option<String>,
    ) -> Result<ExecutionResult, ContractError> {
        // Snapshot all state
        let snapshot = self
            .contracts
            .get(address)
            .ok_or_else(|| ContractError::NotFound(address.to_string()))?
            .snapshot_state();

        // Check revert condition
        if let Some(reason) = revert_condition(&args) {
            return Err(ContractError::Reverted(reason));
        }

        match self.invoke(address, method_name, args, gas_limit) {
            Ok(result) => Ok(result),
            Err(e) => {
                // Rollback
                if let Some(contract) = self.contracts.get_mut(address) {
                    contract.restore_state(snapshot);
                }
                Err(e)
            }
        }
    }

    /// Simulate an inter-contract call.
    pub fn inter_contract_call(
        &mut self,
        caller_address: &str,
        callee_address: &str,
        method_name: &str,
        args: HashMap<String, Value>,
        gas_limit: Option<u64>,
    ) -> Result<ExecutionResult, ContractError> {
        if !self.contracts.contains_key(caller_address) {
            return Err(ContractError::NotFound(caller_address.to_string()));
        }
        // Perform the call on the callee
        self.invoke(callee_address, method_name, args, gas_limit)
    }
}

impl Default for ContractRuntime {
    fn default() -> Self {
        Self::new(1_000_000)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_token_contract() -> Contract {
        let mut c = Contract::new("0xToken", "SimpleToken");
        c.add_method(
            AbiMethod::new("transfer", 100, true)
                .with_param("to", ParamType::Address)
                .with_param("amount", ParamType::Uint),
        );
        c.add_method(
            AbiMethod::new("balance_of", 50, false)
                .with_param("key", ParamType::Address)
                .with_return(ParamType::Uint),
        );
        c.add_method(
            AbiMethod::new("mint", 200, true)
                .with_param("amount", ParamType::Uint),
        );
        c.set_state("total_supply", Value::Number(1000.into()));
        c
    }

    #[test]
    fn test_contract_creation() {
        let c = Contract::new("0xABC", "Test");
        assert_eq!(c.address, "0xABC");
        assert_eq!(c.name, "Test");
        assert_eq!(c.state_count(), 0);
    }

    #[test]
    fn test_contract_state_operations() {
        let mut c = Contract::new("0x1", "C1");
        c.set_state("key1", Value::String("val1".to_string()));
        assert_eq!(c.get_state("key1"), Some(&Value::String("val1".to_string())));
        assert_eq!(c.state_count(), 1);
        c.remove_state("key1");
        assert_eq!(c.state_count(), 0);
    }

    #[test]
    fn test_contract_abi() {
        let c = make_token_contract();
        assert_eq!(c.abi.len(), 3);
        let transfer = c.get_method("transfer").unwrap();
        assert_eq!(transfer.params.len(), 2);
        assert!(transfer.mutates);
        assert_eq!(transfer.gas_cost, 100);
    }

    #[test]
    fn test_deploy_contract() {
        let mut rt = ContractRuntime::new(100_000);
        let c = make_token_contract();
        rt.deploy(c).unwrap();
        assert_eq!(rt.contract_count(), 1);
        assert!(rt.get_contract("0xToken").is_some());
    }

    #[test]
    fn test_deploy_duplicate_error() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();
        let err = rt.deploy(make_token_contract()).unwrap_err();
        assert_eq!(err, ContractError::DuplicateAddress("0xToken".to_string()));
    }

    #[test]
    fn test_invoke_mutating_method() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();

        let mut args = HashMap::new();
        args.insert("to".to_string(), Value::String("0xBob".to_string()));
        args.insert("amount".to_string(), Value::Number(500.into()));

        let result = rt.invoke("0xToken", "transfer", args, None).unwrap();
        assert!(result.success);
        assert!(result.gas_used > 0);

        let contract = rt.get_contract("0xToken").unwrap();
        assert_eq!(contract.get_state("to"), Some(&Value::String("0xBob".to_string())));
    }

    #[test]
    fn test_invoke_view_method() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();

        let mut args = HashMap::new();
        args.insert("key".to_string(), Value::String("total_supply".to_string()));

        let result = rt.invoke("0xToken", "balance_of", args, None).unwrap();
        assert!(result.success);
        assert_eq!(result.return_value, Some(Value::Number(1000.into())));
    }

    #[test]
    fn test_invoke_not_found_contract() {
        let mut rt = ContractRuntime::new(100_000);
        let err = rt.invoke("0xMissing", "foo", HashMap::new(), None).unwrap_err();
        assert_eq!(err, ContractError::NotFound("0xMissing".to_string()));
    }

    #[test]
    fn test_invoke_method_not_found() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();
        let err = rt.invoke("0xToken", "nonexistent", HashMap::new(), None).unwrap_err();
        assert!(matches!(err, ContractError::MethodNotFound { .. }));
    }

    #[test]
    fn test_out_of_gas() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();

        // Give only 10 gas but transfer costs 100
        let err = rt.invoke("0xToken", "transfer", HashMap::new(), Some(10)).unwrap_err();
        assert!(matches!(err, ContractError::OutOfGas { .. }));
    }

    #[test]
    fn test_event_emission() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();
        rt.invoke("0xToken", "transfer", HashMap::new(), None).unwrap();
        assert_eq!(rt.events().len(), 1);
        assert_eq!(rt.events()[0].name, "transfer_called");
    }

    #[test]
    fn test_state_snapshot_and_restore() {
        let mut c = Contract::new("0x1", "C");
        c.set_state("a", Value::Number(1.into()));
        let snap = c.snapshot_state();
        c.set_state("a", Value::Number(99.into()));
        c.set_state("b", Value::Number(2.into()));
        c.restore_state(snap);
        assert_eq!(c.get_state("a"), Some(&Value::Number(1.into())));
        assert_eq!(c.get_state("b"), None);
    }

    #[test]
    fn test_invoke_or_revert() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();

        let args = HashMap::new();
        let err = rt.invoke_or_revert(
            "0xToken",
            "transfer",
            args,
            None,
            |_args| Some("insufficient balance".to_string()),
        ).unwrap_err();
        assert_eq!(err, ContractError::Reverted("insufficient balance".to_string()));
    }

    #[test]
    fn test_inter_contract_call() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();

        let mut c2 = Contract::new("0xExchange", "Exchange");
        c2.add_method(AbiMethod::new("swap", 150, true));
        rt.deploy(c2).unwrap();

        let result = rt.inter_contract_call(
            "0xExchange",
            "0xToken",
            "transfer",
            HashMap::new(),
            None,
        ).unwrap();
        assert!(result.success);
    }

    #[test]
    fn test_gas_accumulates() {
        let mut rt = ContractRuntime::new(100_000);
        rt.deploy(make_token_contract()).unwrap();
        rt.invoke("0xToken", "transfer", HashMap::new(), None).unwrap();
        rt.invoke("0xToken", "transfer", HashMap::new(), None).unwrap();
        let contract = rt.get_contract("0xToken").unwrap();
        assert!(contract.total_gas_used >= 200);
    }

    #[test]
    fn test_param_type_display() {
        assert_eq!(format!("{}", ParamType::Uint), "uint");
        assert_eq!(format!("{}", ParamType::Address), "address");
        assert_eq!(format!("{}", ParamType::Str), "string");
    }

    #[test]
    fn test_default_runtime() {
        let rt = ContractRuntime::default();
        assert_eq!(rt.default_gas_limit, 1_000_000);
        assert_eq!(rt.contract_count(), 0);
    }

    #[test]
    fn test_contract_error_display() {
        let err = ContractError::OutOfGas { used: 500, limit: 100 };
        let msg = format!("{err}");
        assert!(msg.contains("500"));
        assert!(msg.contains("100"));
    }
}
