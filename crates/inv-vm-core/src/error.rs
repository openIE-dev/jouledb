use thiserror::Error;

/// Top-level error type for Invisible Infrastructure.
#[derive(Error, Debug)]
pub enum InvError {
    // --- Identity & Auth ---
    #[error("identity error: {0}")]
    Identity(#[from] IdentityError),

    #[error("auth error: {0}")]
    Auth(#[from] AuthError),

    // --- Crypto ---
    #[error("crypto error: {0}")]
    Crypto(#[from] CryptoError),

    // --- Network ---
    #[error("mesh error: {0}")]
    Mesh(#[from] MeshError),

    // --- Energy ---
    #[error("energy error: {0}")]
    Energy(#[from] EnergyError),

    // --- Runtime ---
    #[error("runtime error: {0}")]
    Runtime(#[from] RuntimeError),

    // --- Storage ---
    #[error("storage error: {0}")]
    Storage(#[from] StorageError),

    // --- Consensus ---
    #[error("consensus error: {0}")]
    Consensus(#[from] ConsensusError),

    // --- Policy ---
    #[error("policy violation: {0}")]
    PolicyViolation(String),

    // --- Generic ---
    #[error("internal error: {0}")]
    Internal(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error(transparent)]
    SerdeJson(#[from] serde_json::Error),
}

#[derive(Error, Debug)]
pub enum IdentityError {
    #[error("invalid node ID: {0}")]
    InvalidNodeId(String),

    #[error("keypair generation failed: {0}")]
    KeypairGeneration(String),

    #[error("certificate error: {0}")]
    Certificate(String),
}

#[derive(Error, Debug)]
pub enum AuthError {
    #[error("authentication failed: {0}")]
    AuthenticationFailed(String),

    #[error("permission denied: {0}")]
    PermissionDenied(String),

    #[error("token expired")]
    TokenExpired,

    #[error("invalid token: {0}")]
    InvalidToken(String),

    #[error("capability not granted: {0}")]
    CapabilityNotGranted(String),
}

#[derive(Error, Debug)]
pub enum CryptoError {
    #[error("encryption failed: {0}")]
    EncryptionFailed(String),

    #[error("decryption failed: {0}")]
    DecryptionFailed(String),

    #[error("signature verification failed")]
    SignatureVerificationFailed,

    #[error("key error: {0}")]
    KeyError(String),

    #[error("TLS error: {0}")]
    TlsError(String),
}

#[derive(Error, Debug)]
pub enum MeshError {
    #[error("connection failed to {node}: {reason}")]
    ConnectionFailed { node: String, reason: String },

    #[error("node not found: {0}")]
    NodeNotFound(String),

    #[error("discovery failed: {0}")]
    DiscoveryFailed(String),

    #[error("gossip error: {0}")]
    GossipError(String),

    #[error("transport error: {0}")]
    TransportError(String),
}

#[derive(Error, Debug)]
pub enum EnergyError {
    #[error("energy budget exceeded: consumed {consumed:.2}J of {budget:.2}J")]
    BudgetExceeded { consumed: f64, budget: f64 },

    #[error("energy meter unavailable: {0}")]
    MeterUnavailable(String),

    #[error("measurement error: {0}")]
    MeasurementError(String),
}

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("WASM execution failed: {0}")]
    WasmExecutionFailed(String),

    #[error("WASM module load failed: {0}")]
    WasmLoadFailed(String),

    #[error("container execution failed: {0}")]
    ContainerFailed(String),

    #[error("host function error: {0}")]
    HostFunctionError(String),

    #[error("capability denied: function={function}, resource={resource}")]
    CapabilityDenied { function: String, resource: String },
}

#[derive(Error, Debug)]
pub enum StorageError {
    #[error("key not found: {0}")]
    KeyNotFound(String),

    #[error("storage backend error: {0}")]
    BackendError(String),

    #[error("replication error: {0}")]
    ReplicationError(String),
}

#[derive(Error, Debug)]
pub enum ConsensusError {
    #[error("not leader")]
    NotLeader,

    #[error("quorum not reached")]
    QuorumNotReached,

    #[error("raft error: {0}")]
    RaftError(String),

    #[error("state machine error: {0}")]
    StateMachineError(String),
}

/// Result type alias for Invisible Infrastructure.
pub type InvResult<T> = Result<T, InvError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let err = InvError::Energy(EnergyError::BudgetExceeded {
            consumed: 5.5,
            budget: 5.0,
        });
        let msg = format!("{err}");
        assert!(msg.contains("energy budget exceeded"));
        assert!(msg.contains("5.50J"));
    }

    #[test]
    fn error_from_io() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: InvError = io_err.into();
        assert!(format!("{err}").contains("file not found"));
    }

    #[test]
    fn nested_error_display() {
        let err = InvError::Runtime(RuntimeError::CapabilityDenied {
            function: "inv_storage_get".into(),
            resource: "bucket/secrets/*".into(),
        });
        let msg = format!("{err}");
        assert!(msg.contains("capability denied"));
        assert!(msg.contains("inv_storage_get"));
    }
}
