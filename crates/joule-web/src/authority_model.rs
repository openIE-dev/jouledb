//! Server/client authority model for multiplayer — ownership, prediction, delegation.
//!
//! Provides `AuthorityMode` enum (ServerAuthoritative, ClientAuthoritative, Shared),
//! `AuthorityManager` assigning authority per entity, authority transfer requests,
//! input validation on authoritative side, prediction on non-authoritative side,
//! authority conflict resolution, ownership table, and authority delegation with
//! timeout.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Authority model domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuthorityError {
    /// Entity not found.
    EntityNotFound(u64),
    /// Entity already registered.
    DuplicateEntity(u64),
    /// No authority over this entity.
    NoAuthority { entity_id: u64, requester: u64 },
    /// Transfer already pending.
    TransferPending(u64),
    /// Delegation expired.
    DelegationExpired { entity_id: u64, delegate: u64 },
    /// Invalid input rejected by authority.
    InputRejected { entity_id: u64, reason: String },
}

impl fmt::Display for AuthorityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EntityNotFound(id) => write!(f, "entity not found: {id}"),
            Self::DuplicateEntity(id) => write!(f, "duplicate entity: {id}"),
            Self::NoAuthority { entity_id, requester } => {
                write!(f, "no authority: entity={entity_id}, requester={requester}")
            }
            Self::TransferPending(id) => write!(f, "transfer already pending for entity {id}"),
            Self::DelegationExpired { entity_id, delegate } => {
                write!(f, "delegation expired: entity={entity_id}, delegate={delegate}")
            }
            Self::InputRejected { entity_id, reason } => {
                write!(f, "input rejected for entity {entity_id}: {reason}")
            }
        }
    }
}

impl std::error::Error for AuthorityError {}

// ── Authority Mode ──────────────────────────────────────────────

/// Authority model for an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorityMode {
    /// Server has full authority; clients predict and get corrected.
    ServerAuthoritative,
    /// Client has full authority; server accepts client state.
    ClientAuthoritative,
    /// Shared: both sides can modify, with reconciliation.
    Shared,
}

impl fmt::Display for AuthorityMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::ServerAuthoritative => "server",
            Self::ClientAuthoritative => "client",
            Self::Shared => "shared",
        };
        write!(f, "{s}")
    }
}

// ── Input Command ───────────────────────────────────────────────

/// An input command from a client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputCommand {
    pub entity_id: u64,
    pub sender: u64,
    pub sequence: u64,
    pub action: String,
    pub payload: Vec<u8>,
}

impl InputCommand {
    pub fn new(entity_id: u64, sender: u64, sequence: u64, action: impl Into<String>) -> Self {
        Self {
            entity_id,
            sender,
            sequence,
            action: action.into(),
            payload: Vec::new(),
        }
    }

    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }
}

impl fmt::Display for InputCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Input(entity={}, sender={}, seq={}, action={})",
            self.entity_id, self.sender, self.sequence, self.action,
        )
    }
}

// ── Prediction ──────────────────────────────────────────────────

/// A predicted state on the non-authoritative side.
#[derive(Debug, Clone)]
pub struct Prediction {
    pub entity_id: u64,
    pub sequence: u64,
    pub predicted_state: Vec<u8>,
    pub confirmed: bool,
}

impl Prediction {
    pub fn new(entity_id: u64, sequence: u64, state: Vec<u8>) -> Self {
        Self { entity_id, sequence, predicted_state: state, confirmed: false }
    }
}

// ── Delegation ──────────────────────────────────────────────────

/// Temporary delegation of authority to another node.
#[derive(Debug, Clone)]
struct Delegation {
    delegate: u64,
    granted_at_tick: u64,
    expires_at_tick: u64,
}

// ── Transfer Request ────────────────────────────────────────────

/// A pending authority transfer request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransferRequest {
    pub entity_id: u64,
    pub from: u64,
    pub to: u64,
    pub requested_at_tick: u64,
}

impl fmt::Display for TransferRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Transfer(entity={}, {}->{}, tick={})",
            self.entity_id, self.from, self.to, self.requested_at_tick,
        )
    }
}

// ── Authority Record ────────────────────────────────────────────

/// Per-entity authority record.
#[derive(Debug, Clone)]
struct AuthorityRecord {
    mode: AuthorityMode,
    owner: u64,
    delegation: Option<Delegation>,
    pending_transfer: Option<TransferRequest>,
    /// Input validation rules (action -> max payload size).
    validation_rules: HashMap<String, usize>,
}

// ── Authority Manager ───────────────────────────────────────────

/// Manages authority assignments, transfers, delegations, and input validation.
pub struct AuthorityManager {
    /// Server node ID (0 by convention).
    pub server_id: u64,
    entities: HashMap<u64, AuthorityRecord>,
    /// Prediction buffers per entity.
    predictions: HashMap<u64, Vec<Prediction>>,
    /// Current logical tick.
    current_tick: u64,
    /// Statistics.
    inputs_accepted: u64,
    inputs_rejected: u64,
    transfers_completed: u64,
}

impl AuthorityManager {
    pub fn new(server_id: u64) -> Self {
        Self {
            server_id,
            entities: HashMap::new(),
            predictions: HashMap::new(),
            current_tick: 0,
            inputs_accepted: 0,
            inputs_rejected: 0,
            transfers_completed: 0,
        }
    }

    /// Advance the tick counter.
    pub fn tick(&mut self) {
        self.current_tick += 1;
        // Expire delegations.
        for record in self.entities.values_mut() {
            if let Some(del) = &record.delegation {
                if del.expires_at_tick <= self.current_tick {
                    record.delegation = None;
                }
            }
        }
    }

    /// Register an entity with an authority mode and owner.
    pub fn register(
        &mut self,
        entity_id: u64,
        mode: AuthorityMode,
        owner: u64,
    ) -> Result<(), AuthorityError> {
        if self.entities.contains_key(&entity_id) {
            return Err(AuthorityError::DuplicateEntity(entity_id));
        }
        self.entities.insert(entity_id, AuthorityRecord {
            mode,
            owner,
            delegation: None,
            pending_transfer: None,
            validation_rules: HashMap::new(),
        });
        Ok(())
    }

    /// Unregister an entity.
    pub fn unregister(&mut self, entity_id: u64) -> Result<(), AuthorityError> {
        self.entities.remove(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?;
        self.predictions.remove(&entity_id);
        Ok(())
    }

    /// Get the authority mode for an entity.
    pub fn mode(&self, entity_id: u64) -> Result<AuthorityMode, AuthorityError> {
        Ok(self.entities.get(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?.mode)
    }

    /// Get the owner of an entity.
    pub fn owner(&self, entity_id: u64) -> Result<u64, AuthorityError> {
        Ok(self.entities.get(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?.owner)
    }

    /// Check if a node has authority over an entity (owner or delegate).
    pub fn has_authority(&self, entity_id: u64, node_id: u64) -> Result<bool, AuthorityError> {
        let record = self.entities.get(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?;
        if record.owner == node_id {
            return Ok(true);
        }
        if let Some(del) = &record.delegation {
            if del.delegate == node_id && del.expires_at_tick > self.current_tick {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Set a validation rule for an entity (action -> max payload size).
    pub fn set_validation_rule(
        &mut self,
        entity_id: u64,
        action: impl Into<String>,
        max_payload: usize,
    ) -> Result<(), AuthorityError> {
        let record = self.entities.get_mut(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?;
        record.validation_rules.insert(action.into(), max_payload);
        Ok(())
    }

    /// Validate an input command on the authoritative side.
    pub fn validate_input(&mut self, cmd: &InputCommand) -> Result<(), AuthorityError> {
        let record = self.entities.get(&cmd.entity_id)
            .ok_or(AuthorityError::EntityNotFound(cmd.entity_id))?;

        // In server-authoritative mode, only the server or delegatee can issue commands.
        if record.mode == AuthorityMode::ServerAuthoritative {
            let authorized = record.owner == cmd.sender
                || record.delegation.as_ref()
                    .map(|d| d.delegate == cmd.sender && d.expires_at_tick > self.current_tick)
                    .unwrap_or(false);
            if !authorized {
                self.inputs_rejected += 1;
                return Err(AuthorityError::NoAuthority {
                    entity_id: cmd.entity_id,
                    requester: cmd.sender,
                });
            }
        }

        // Check validation rules.
        if let Some(&max_size) = record.validation_rules.get(&cmd.action) {
            if cmd.payload.len() > max_size {
                self.inputs_rejected += 1;
                return Err(AuthorityError::InputRejected {
                    entity_id: cmd.entity_id,
                    reason: format!("payload too large: {} > {max_size}", cmd.payload.len()),
                });
            }
        }

        self.inputs_accepted += 1;
        Ok(())
    }

    /// Request authority transfer.
    pub fn request_transfer(
        &mut self,
        entity_id: u64,
        from: u64,
        to: u64,
    ) -> Result<TransferRequest, AuthorityError> {
        let record = self.entities.get_mut(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?;
        if record.owner != from {
            return Err(AuthorityError::NoAuthority { entity_id, requester: from });
        }
        if record.pending_transfer.is_some() {
            return Err(AuthorityError::TransferPending(entity_id));
        }
        let req = TransferRequest {
            entity_id,
            from,
            to,
            requested_at_tick: self.current_tick,
        };
        record.pending_transfer = Some(req.clone());
        Ok(req)
    }

    /// Complete a pending authority transfer.
    pub fn complete_transfer(&mut self, entity_id: u64) -> Result<(), AuthorityError> {
        let record = self.entities.get_mut(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?;
        let transfer = record.pending_transfer.take()
            .ok_or(AuthorityError::EntityNotFound(entity_id))?;
        record.owner = transfer.to;
        record.delegation = None;
        self.transfers_completed += 1;
        Ok(())
    }

    /// Delegate authority temporarily to another node.
    pub fn delegate(
        &mut self,
        entity_id: u64,
        owner: u64,
        delegate: u64,
        duration_ticks: u64,
    ) -> Result<(), AuthorityError> {
        let record = self.entities.get_mut(&entity_id)
            .ok_or(AuthorityError::EntityNotFound(entity_id))?;
        if record.owner != owner {
            return Err(AuthorityError::NoAuthority { entity_id, requester: owner });
        }
        record.delegation = Some(Delegation {
            delegate,
            granted_at_tick: self.current_tick,
            expires_at_tick: self.current_tick + duration_ticks,
        });
        Ok(())
    }

    /// Store a prediction for an entity.
    pub fn add_prediction(&mut self, prediction: Prediction) {
        self.predictions.entry(prediction.entity_id).or_default().push(prediction);
    }

    /// Confirm predictions up to a given sequence number.
    pub fn confirm_predictions(&mut self, entity_id: u64, up_to_seq: u64) -> usize {
        let preds = self.predictions.entry(entity_id).or_default();
        let mut confirmed = 0;
        for p in preds.iter_mut() {
            if p.sequence <= up_to_seq && !p.confirmed {
                p.confirmed = true;
                confirmed += 1;
            }
        }
        // Remove confirmed predictions.
        preds.retain(|p| !p.confirmed);
        confirmed
    }

    /// Get unconfirmed predictions for an entity.
    pub fn pending_predictions(&self, entity_id: u64) -> usize {
        self.predictions.get(&entity_id).map(|v| v.len()).unwrap_or(0)
    }

    /// Entity count.
    pub fn entity_count(&self) -> usize {
        self.entities.len()
    }

    /// Current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Inputs accepted.
    pub fn inputs_accepted(&self) -> u64 {
        self.inputs_accepted
    }

    /// Inputs rejected.
    pub fn inputs_rejected(&self) -> u64 {
        self.inputs_rejected
    }

    /// Transfers completed.
    pub fn transfers_completed(&self) -> u64 {
        self.transfers_completed
    }

    /// Get all entity IDs owned by a node.
    pub fn entities_owned_by(&self, node_id: u64) -> Vec<u64> {
        self.entities.iter()
            .filter(|(_, r)| r.owner == node_id)
            .map(|(&id, _)| id)
            .collect()
    }
}

impl fmt::Display for AuthorityManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AuthorityManager(server={}, entities={}, tick={})",
            self.server_id,
            self.entities.len(),
            self.current_tick,
        )
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authority_mode_display() {
        assert_eq!(format!("{}", AuthorityMode::ServerAuthoritative), "server");
        assert_eq!(format!("{}", AuthorityMode::ClientAuthoritative), "client");
        assert_eq!(format!("{}", AuthorityMode::Shared), "shared");
    }

    #[test]
    fn register_and_query() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        assert_eq!(mgr.mode(1).unwrap(), AuthorityMode::ServerAuthoritative);
        assert_eq!(mgr.owner(1).unwrap(), 0);
    }

    #[test]
    fn duplicate_register_error() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        assert!(matches!(
            mgr.register(1, AuthorityMode::ClientAuthoritative, 1),
            Err(AuthorityError::DuplicateEntity(1))
        ));
    }

    #[test]
    fn unregister() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        mgr.unregister(1).unwrap();
        assert!(matches!(mgr.mode(1), Err(AuthorityError::EntityNotFound(1))));
    }

    #[test]
    fn has_authority_owner() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        assert!(mgr.has_authority(1, 0).unwrap());
        assert!(!mgr.has_authority(1, 99).unwrap());
    }

    #[test]
    fn delegate_authority() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        mgr.delegate(1, 0, 5, 10).unwrap();
        assert!(mgr.has_authority(1, 5).unwrap());
    }

    #[test]
    fn delegation_expires() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        mgr.delegate(1, 0, 5, 2).unwrap();
        mgr.tick();
        mgr.tick();
        assert!(!mgr.has_authority(1, 5).unwrap());
    }

    #[test]
    fn validate_input_server_auth() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        let cmd = InputCommand::new(1, 0, 1, "move");
        mgr.validate_input(&cmd).unwrap();

        let bad_cmd = InputCommand::new(1, 99, 1, "move");
        assert!(matches!(mgr.validate_input(&bad_cmd), Err(AuthorityError::NoAuthority { .. })));
    }

    #[test]
    fn validate_input_payload_size() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ClientAuthoritative, 5).unwrap();
        mgr.set_validation_rule(1, "move", 10).unwrap();
        let cmd = InputCommand::new(1, 5, 1, "move").with_payload(vec![0; 20]);
        assert!(matches!(mgr.validate_input(&cmd), Err(AuthorityError::InputRejected { .. })));
    }

    #[test]
    fn request_and_complete_transfer() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        mgr.request_transfer(1, 0, 5).unwrap();
        mgr.complete_transfer(1).unwrap();
        assert_eq!(mgr.owner(1).unwrap(), 5);
        assert_eq!(mgr.transfers_completed(), 1);
    }

    #[test]
    fn transfer_pending_error() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        mgr.request_transfer(1, 0, 5).unwrap();
        assert!(matches!(mgr.request_transfer(1, 0, 6), Err(AuthorityError::TransferPending(1))));
    }

    #[test]
    fn transfer_not_owner_error() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        assert!(matches!(mgr.request_transfer(1, 99, 5), Err(AuthorityError::NoAuthority { .. })));
    }

    #[test]
    fn predictions() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        mgr.add_prediction(Prediction::new(1, 1, vec![10]));
        mgr.add_prediction(Prediction::new(1, 2, vec![20]));
        assert_eq!(mgr.pending_predictions(1), 2);
        let confirmed = mgr.confirm_predictions(1, 1);
        assert_eq!(confirmed, 1);
        assert_eq!(mgr.pending_predictions(1), 1);
    }

    #[test]
    fn entities_owned_by() {
        let mut mgr = AuthorityManager::new(0);
        mgr.register(1, AuthorityMode::ServerAuthoritative, 0).unwrap();
        mgr.register(2, AuthorityMode::ClientAuthoritative, 5).unwrap();
        mgr.register(3, AuthorityMode::ServerAuthoritative, 0).unwrap();
        assert_eq!(mgr.entities_owned_by(0).len(), 2);
    }

    #[test]
    fn input_command_display() {
        let cmd = InputCommand::new(1, 2, 3, "jump");
        let d = format!("{cmd}");
        assert!(d.contains("Input"));
    }

    #[test]
    fn transfer_request_display() {
        let req = TransferRequest { entity_id: 1, from: 0, to: 5, requested_at_tick: 10 };
        let d = format!("{req}");
        assert!(d.contains("Transfer"));
    }

    #[test]
    fn manager_display() {
        let mgr = AuthorityManager::new(0);
        let d = format!("{mgr}");
        assert!(d.contains("AuthorityManager"));
    }

    #[test]
    fn tick_advances() {
        let mut mgr = AuthorityManager::new(0);
        mgr.tick();
        mgr.tick();
        assert_eq!(mgr.current_tick(), 2);
    }
}
