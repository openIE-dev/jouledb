use crate::error::JwpError;
use crate::frame::FrameType;

/// Protocol states for the JWP state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProtocolState {
    /// Initial state — awaiting handshake.
    Init,
    /// Handshake complete — ready for queries.
    Ready,
    /// Actively streaming results for a query.
    Streaming,
    /// v2: Mid-connection capability renegotiation in progress.
    Negotiating,
    /// Auth: in-band authentication in progress (between Handshake and Ready).
    Authenticating,
    /// Agent contract negotiation in progress (propose/accept/counter).
    Contracting,
    /// Agent is engaged — working under a signed contract.
    AgentEngaged,
}

impl std::fmt::Display for ProtocolState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Init => write!(f, "Init"),
            Self::Ready => write!(f, "Ready"),
            Self::Streaming => write!(f, "Streaming"),
            Self::Negotiating => write!(f, "Negotiating"),
            Self::Authenticating => write!(f, "Authenticating"),
            Self::Contracting => write!(f, "Contracting"),
            Self::AgentEngaged => write!(f, "AgentEngaged"),
        }
    }
}

/// Tracks protocol state and validates transitions.
///
/// ```text
/// [Init] --Handshake--> [Ready] --Query--> [Streaming]
/// [Streaming] --Result--> [Streaming]
/// [Streaming] --Done--> [Ready]
/// [Streaming] --Cancel--> [Ready]
/// [Ready] --Query--> [Streaming]
/// [*] --Error--> [Ready]  (if past Init)
/// [*] --Heartbeat--> [same state]
/// ```
pub struct ProtocolStateMachine {
    state: ProtocolState,
}

impl ProtocolStateMachine {
    pub fn new() -> Self {
        Self {
            state: ProtocolState::Init,
        }
    }

    pub fn state(&self) -> ProtocolState {
        self.state
    }

    /// Validate and apply a state transition for the given frame type.
    /// Returns the new state on success.
    pub fn transition(&mut self, frame_type: FrameType) -> Result<ProtocolState, JwpError> {
        let new_state = match (self.state, frame_type) {
            // Heartbeat is always valid, state unchanged
            (s, FrameType::Heartbeat) => s,

            // Error transitions: Init stays Init, Authenticating → Init (must re-handshake),
            // all others → Ready.
            (ProtocolState::Init, FrameType::Error) => ProtocolState::Init,
            (ProtocolState::Authenticating, FrameType::Error) => ProtocolState::Init,
            (_, FrameType::Error) => ProtocolState::Ready,

            // Init → Ready via Handshake
            (ProtocolState::Init, FrameType::Handshake) => ProtocolState::Ready,

            // Ready → Streaming via Query
            (ProtocolState::Ready, FrameType::Query) => ProtocolState::Streaming,

            // Streaming: Result stays Streaming
            (ProtocolState::Streaming, FrameType::Result) => ProtocolState::Streaming,

            // Streaming: Receipt stays Streaming (intermediate receipt)
            (ProtocolState::Streaming, FrameType::Receipt) => ProtocolState::Streaming,

            // Streaming → Ready via Done or Cancel
            (ProtocolState::Streaming, FrameType::Done) => ProtocolState::Ready,
            (ProtocolState::Streaming, FrameType::Cancel) => ProtocolState::Ready,

            // Ready: Meta starts streaming (server sends Meta right after Query)
            (ProtocolState::Ready, FrameType::Meta) => ProtocolState::Streaming,

            // Streaming: Meta is valid (re-sent for context)
            (ProtocolState::Streaming, FrameType::Meta) => ProtocolState::Streaming,

            // ── v2 frame types ──────────────────────────────────

            // Negotiate: Ready → Negotiating (request), Negotiating → Ready (response)
            (ProtocolState::Ready, FrameType::Negotiate) => ProtocolState::Negotiating,
            (ProtocolState::Negotiating, FrameType::Negotiate) => ProtocolState::Ready,

            // ProfileUpdate: informational, no state change (valid from Ready or Streaming)
            (ProtocolState::Ready, FrameType::ProfileUpdate) => ProtocolState::Ready,
            (ProtocolState::Streaming, FrameType::ProfileUpdate) => ProtocolState::Streaming,

            // EnergyGradient: informational, no state change (valid from Ready or Streaming)
            (ProtocolState::Ready, FrameType::EnergyGradient) => ProtocolState::Ready,
            (ProtocolState::Streaming, FrameType::EnergyGradient) => ProtocolState::Streaming,

            // Batch: only valid while streaming (contains packed result sub-frames)
            (ProtocolState::Streaming, FrameType::Batch) => ProtocolState::Streaming,

            // StreamChunk: token-by-token text output, only valid while streaming
            (ProtocolState::Streaming, FrameType::StreamChunk) => ProtocolState::Streaming,

            // RateLimit: advisory, no state change (valid from Ready or Streaming)
            (ProtocolState::Ready, FrameType::RateLimit) => ProtocolState::Ready,
            (ProtocolState::Streaming, FrameType::RateLimit) => ProtocolState::Streaming,

            // Session lifecycle: revoke/extend only valid from Ready
            (ProtocolState::Ready, FrameType::SessionRevoke) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::SessionExtend) => ProtocolState::Ready,

            // Command/CommandResponse: stateless request-response, only from Ready
            (ProtocolState::Ready, FrameType::Command) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::CommandResponse) => ProtocolState::Ready,

            // ── Auth frame types ─────────────────────────────────

            // Authenticating: challenge/response stay in Authenticating
            (ProtocolState::Authenticating, FrameType::AuthChallenge) => {
                ProtocolState::Authenticating
            }
            (ProtocolState::Authenticating, FrameType::AuthResponse) => {
                ProtocolState::Authenticating
            }
            // Authenticating → Ready via AuthSuccess
            (ProtocolState::Authenticating, FrameType::AuthSuccess) => ProtocolState::Ready,

            // ── Passkey ceremony frames ──────────────────────────

            // Passkey register: begin enters Authenticating from Ready or Init
            (ProtocolState::Ready, FrameType::PasskeyRegisterBegin) => {
                ProtocolState::Authenticating
            }
            (ProtocolState::Init, FrameType::PasskeyRegisterBegin) => {
                ProtocolState::Authenticating
            }
            // Challenge/complete stay in Authenticating
            (ProtocolState::Authenticating, FrameType::PasskeyRegisterChallenge) => {
                ProtocolState::Authenticating
            }
            (ProtocolState::Authenticating, FrameType::PasskeyRegisterComplete) => {
                ProtocolState::Authenticating
            }

            // Passkey login: begin enters Authenticating from Ready or Init
            (ProtocolState::Ready, FrameType::PasskeyLoginBegin) => {
                ProtocolState::Authenticating
            }
            (ProtocolState::Init, FrameType::PasskeyLoginBegin) => {
                ProtocolState::Authenticating
            }
            // Challenge/complete stay in Authenticating
            (ProtocolState::Authenticating, FrameType::PasskeyLoginChallenge) => {
                ProtocolState::Authenticating
            }
            (ProtocolState::Authenticating, FrameType::PasskeyLoginComplete) => {
                ProtocolState::Authenticating
            }

            // ── Billing frames (request-response, only from Ready) ──

            (ProtocolState::Ready, FrameType::BalanceQuery) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::BalanceResponse) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::TopupBegin) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::TopupResponse) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::UsageQuery) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::UsageResponse) => ProtocolState::Ready,

            // ── Device-code frames ───────────────────────────────

            // Device-code bootstrap: valid from Init (before auth)
            (ProtocolState::Init, FrameType::DeviceCodeRequest) => ProtocolState::Init,
            (ProtocolState::Init, FrameType::DeviceCodeResponse) => ProtocolState::Init,
            (ProtocolState::Init, FrameType::DeviceCodePoll) => ProtocolState::Init,
            (ProtocolState::Init, FrameType::DeviceCodeResult) => ProtocolState::Init,
            // Also valid from Ready (re-auth)
            (ProtocolState::Ready, FrameType::DeviceCodeRequest) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::DeviceCodeResponse) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::DeviceCodePoll) => ProtocolState::Ready,
            (ProtocolState::Ready, FrameType::DeviceCodeResult) => ProtocolState::Ready,

            // ── Agent contract lifecycle frames ──────────────────

            // ContractPropose: Ready → Contracting (host proposes to agent)
            (ProtocolState::Ready, FrameType::ContractPropose) => ProtocolState::Contracting,

            // ContractRespond: Contracting → Contracting (counter-propose stays in negotiation)
            (ProtocolState::Contracting, FrameType::ContractRespond) => {
                ProtocolState::Contracting
            }

            // ContractPropose again: Contracting → Contracting (revised proposal after counter)
            (ProtocolState::Contracting, FrameType::ContractPropose) => {
                ProtocolState::Contracting
            }

            // ContractSigned: Contracting → AgentEngaged (mutual agreement reached)
            (ProtocolState::Contracting, FrameType::ContractSigned) => {
                ProtocolState::AgentEngaged
            }

            // ExtensionRequest/Response: only valid while AgentEngaged
            (ProtocolState::AgentEngaged, FrameType::ExtensionRequest) => {
                ProtocolState::AgentEngaged
            }
            (ProtocolState::AgentEngaged, FrameType::ExtensionResponse) => {
                ProtocolState::AgentEngaged
            }

            // AgentReturn: AgentEngaged → Ready (agent voluntarily returns)
            (ProtocolState::AgentEngaged, FrameType::AgentReturn) => ProtocolState::Ready,

            // AgentRecall: AgentEngaged → Ready (host forces return)
            (ProtocolState::AgentEngaged, FrameType::AgentRecall) => ProtocolState::Ready,

            // Receipt during engagement (intermediate energy receipt)
            (ProtocolState::AgentEngaged, FrameType::Receipt) => ProtocolState::AgentEngaged,

            // Note: Error during Contracting and AgentEngaged is handled by
            // the (_, FrameType::Error) => Ready wildcard above.

            // Everything else is invalid
            (state, ft) => {
                return Err(JwpError::InvalidTransition {
                    from: state.to_string(),
                    event: format!("{ft:?}"),
                });
            }
        };

        self.state = new_state;
        Ok(new_state)
    }

    /// Force the state machine into a specific state.
    /// Used by auth layer to enter `Authenticating` after a Handshake
    /// carries a credential.
    pub fn force_state(&mut self, state: ProtocolState) {
        self.state = state;
    }

    /// Reset to Init state (e.g., on connection reset).
    pub fn reset(&mut self) {
        self.state = ProtocolState::Init;
    }
}

impl Default for ProtocolStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_handshake_query_results_done() {
        let mut sm = ProtocolStateMachine::new();
        assert_eq!(sm.state(), ProtocolState::Init);

        sm.transition(FrameType::Handshake).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);

        sm.transition(FrameType::Meta).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);

        sm.transition(FrameType::Result).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);

        sm.transition(FrameType::Result).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);

        sm.transition(FrameType::Done).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn cancel_returns_to_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::Result).unwrap();

        sm.transition(FrameType::Cancel).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn heartbeat_always_valid() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Heartbeat).unwrap();
        assert_eq!(sm.state(), ProtocolState::Init);

        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Heartbeat).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::Heartbeat).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    #[test]
    fn error_transitions_to_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Query).unwrap();

        sm.transition(FrameType::Error).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn invalid_query_from_init() {
        let mut sm = ProtocolStateMachine::new();
        assert!(sm.transition(FrameType::Query).is_err());
    }

    #[test]
    fn invalid_handshake_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        assert!(sm.transition(FrameType::Handshake).is_err());
    }

    #[test]
    fn multiple_queries() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // First query
        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::Done).unwrap();

        // Second query
        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::Result).unwrap();
        sm.transition(FrameType::Done).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn reset() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Query).unwrap();

        sm.reset();
        assert_eq!(sm.state(), ProtocolState::Init);
    }

    // ── v2 state machine tests ───────────────────────────────────

    #[test]
    fn negotiate_ready_to_negotiating_and_back() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Request negotiation
        sm.transition(FrameType::Negotiate).unwrap();
        assert_eq!(sm.state(), ProtocolState::Negotiating);

        // Response completes negotiation
        sm.transition(FrameType::Negotiate).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn negotiate_error_returns_to_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Negotiate).unwrap();
        assert_eq!(sm.state(), ProtocolState::Negotiating);

        sm.transition(FrameType::Error).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn heartbeat_valid_in_negotiating() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Negotiate).unwrap();

        sm.transition(FrameType::Heartbeat).unwrap();
        assert_eq!(sm.state(), ProtocolState::Negotiating);
    }

    #[test]
    fn profile_update_no_state_change() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        sm.transition(FrameType::ProfileUpdate).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::ProfileUpdate).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    #[test]
    fn energy_gradient_no_state_change() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        sm.transition(FrameType::EnergyGradient).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::EnergyGradient).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    #[test]
    fn batch_only_valid_in_streaming() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Batch from Ready should fail
        assert!(sm.transition(FrameType::Batch).is_err());

        // Batch from Streaming should succeed
        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::Batch).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    #[test]
    fn rate_limit_no_state_change() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // From Ready
        sm.transition(FrameType::RateLimit).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // From Streaming
        sm.transition(FrameType::Query).unwrap();
        sm.transition(FrameType::RateLimit).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    #[test]
    fn rate_limit_invalid_from_init() {
        let mut sm = ProtocolStateMachine::new();
        assert!(sm.transition(FrameType::RateLimit).is_err());
    }

    #[test]
    fn session_revoke_only_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Valid from Ready
        sm.transition(FrameType::SessionRevoke).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Invalid from Streaming
        sm.transition(FrameType::Query).unwrap();
        assert!(sm.transition(FrameType::SessionRevoke).is_err());
    }

    #[test]
    fn session_extend_only_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Valid from Ready
        sm.transition(FrameType::SessionExtend).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Invalid from Init
        let mut sm2 = ProtocolStateMachine::new();
        assert!(sm2.transition(FrameType::SessionExtend).is_err());
    }

    // ── Auth state machine tests ──────────────────────────────────

    #[test]
    fn auth_happy_path_challenge_response_success() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Server detects credential in handshake → force into Authenticating
        sm.force_state(ProtocolState::Authenticating);
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        // Server sends challenge
        sm.transition(FrameType::AuthChallenge).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        // Client sends response
        sm.transition(FrameType::AuthResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        // Server sends success → Ready
        sm.transition(FrameType::AuthSuccess).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Can now proceed with normal queries
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    #[test]
    fn auth_failure_returns_to_init() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.force_state(ProtocolState::Authenticating);

        sm.transition(FrameType::AuthChallenge).unwrap();
        sm.transition(FrameType::AuthResponse).unwrap();

        // Auth fails → Error → Init (must re-handshake)
        sm.transition(FrameType::Error).unwrap();
        assert_eq!(sm.state(), ProtocolState::Init);
    }

    #[test]
    fn auth_heartbeat_valid_during_authenticating() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.force_state(ProtocolState::Authenticating);

        sm.transition(FrameType::Heartbeat).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);
    }

    #[test]
    fn auth_query_invalid_during_authenticating() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.force_state(ProtocolState::Authenticating);

        // Can't query while authenticating
        assert!(sm.transition(FrameType::Query).is_err());
    }

    #[test]
    fn auth_challenge_invalid_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // AuthChallenge only valid in Authenticating state
        assert!(sm.transition(FrameType::AuthChallenge).is_err());
    }

    #[test]
    fn unauthenticated_v1_compat_skips_auth() {
        // v1 path: Init → Handshake → Ready (no auth)
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Goes straight to queries without touching Authenticating
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    // ── Command/CommandResponse state machine tests ───────────────

    #[test]
    fn command_valid_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::Command).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::CommandResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn command_invalid_from_init() {
        let mut sm = ProtocolStateMachine::new();
        assert!(sm.transition(FrameType::Command).is_err());
    }

    #[test]
    fn command_invalid_from_streaming() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);

        assert!(sm.transition(FrameType::Command).is_err());
    }

    #[test]
    fn command_then_query_works() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Command stays Ready
        sm.transition(FrameType::Command).unwrap();
        sm.transition(FrameType::CommandResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Can still do queries
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
        sm.transition(FrameType::Done).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Can do another command
        sm.transition(FrameType::Command).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    // ── Passkey state machine tests ─────────────────────────────

    #[test]
    fn passkey_register_happy_path() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Begin registration → Authenticating
        sm.transition(FrameType::PasskeyRegisterBegin).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        // Server sends challenge
        sm.transition(FrameType::PasskeyRegisterChallenge).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        // Client completes
        sm.transition(FrameType::PasskeyRegisterComplete).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        // Server sends AuthSuccess → Ready
        sm.transition(FrameType::AuthSuccess).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Can now query
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    #[test]
    fn passkey_login_happy_path() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        sm.transition(FrameType::PasskeyLoginBegin).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        sm.transition(FrameType::PasskeyLoginChallenge).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        sm.transition(FrameType::PasskeyLoginComplete).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        sm.transition(FrameType::AuthSuccess).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn passkey_register_failure_returns_to_init() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::PasskeyRegisterBegin).unwrap();
        sm.transition(FrameType::PasskeyRegisterChallenge).unwrap();

        // Error during auth → Init
        sm.transition(FrameType::Error).unwrap();
        assert_eq!(sm.state(), ProtocolState::Init);
    }

    #[test]
    fn passkey_register_invalid_from_streaming() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);

        // Can't start passkey from Streaming
        assert!(sm.transition(FrameType::PasskeyRegisterBegin).is_err());
    }

    #[test]
    fn passkey_login_from_init() {
        // Passkey login directly from Init (no prior handshake credential)
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::PasskeyLoginBegin).unwrap();
        assert_eq!(sm.state(), ProtocolState::Authenticating);

        sm.transition(FrameType::PasskeyLoginChallenge).unwrap();
        sm.transition(FrameType::PasskeyLoginComplete).unwrap();
        sm.transition(FrameType::AuthSuccess).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    // ── Billing state machine tests ──────────────────────────────

    #[test]
    fn balance_query_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        sm.transition(FrameType::BalanceQuery).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::BalanceResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn topup_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        sm.transition(FrameType::TopupBegin).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::TopupResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn usage_query_from_ready() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        sm.transition(FrameType::UsageQuery).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        sm.transition(FrameType::UsageResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn billing_invalid_from_init() {
        let mut sm = ProtocolStateMachine::new();
        assert!(sm.transition(FrameType::BalanceQuery).is_err());
        assert!(sm.transition(FrameType::TopupBegin).is_err());
        assert!(sm.transition(FrameType::UsageQuery).is_err());
    }

    #[test]
    fn billing_invalid_from_streaming() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::Query).unwrap();

        assert!(sm.transition(FrameType::BalanceQuery).is_err());
    }

    #[test]
    fn billing_after_passkey_auth() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Passkey login
        sm.transition(FrameType::PasskeyLoginBegin).unwrap();
        sm.transition(FrameType::PasskeyLoginChallenge).unwrap();
        sm.transition(FrameType::PasskeyLoginComplete).unwrap();
        sm.transition(FrameType::AuthSuccess).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Now billing works
        sm.transition(FrameType::BalanceQuery).unwrap();
        sm.transition(FrameType::BalanceResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // And queries work
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
    }

    // ── Agent contract lifecycle state machine tests ─────────────

    #[test]
    fn contract_happy_path_propose_accept_return() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Host proposes contract
        sm.transition(FrameType::ContractPropose).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);

        // Agent accepts
        sm.transition(FrameType::ContractRespond).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);

        // Host signs contract
        sm.transition(FrameType::ContractSigned).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);

        // Agent works... then returns
        sm.transition(FrameType::AgentReturn).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn contract_counter_propose_then_accept() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Host proposes
        sm.transition(FrameType::ContractPropose).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);

        // Agent counter-proposes
        sm.transition(FrameType::ContractRespond).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);

        // Host sends revised proposal
        sm.transition(FrameType::ContractPropose).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);

        // Agent accepts revised
        sm.transition(FrameType::ContractRespond).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);

        // Host signs
        sm.transition(FrameType::ContractSigned).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);
    }

    #[test]
    fn contract_rejection_via_error() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);

        // Error during negotiation → back to Ready
        sm.transition(FrameType::Error).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn contract_extension_during_engagement() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();
        sm.transition(FrameType::ContractRespond).unwrap();
        sm.transition(FrameType::ContractSigned).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);

        // Agent requests extension
        sm.transition(FrameType::ExtensionRequest).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);

        // Host grants
        sm.transition(FrameType::ExtensionResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);

        // Multiple extensions
        sm.transition(FrameType::ExtensionRequest).unwrap();
        sm.transition(FrameType::ExtensionResponse).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);
    }

    #[test]
    fn contract_recall_during_engagement() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();
        sm.transition(FrameType::ContractRespond).unwrap();
        sm.transition(FrameType::ContractSigned).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);

        // Host recalls agent
        sm.transition(FrameType::AgentRecall).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn contract_receipt_during_engagement() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();
        sm.transition(FrameType::ContractRespond).unwrap();
        sm.transition(FrameType::ContractSigned).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);

        // Energy receipts during engagement
        sm.transition(FrameType::Receipt).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);
    }

    #[test]
    fn contract_error_during_engagement() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();
        sm.transition(FrameType::ContractRespond).unwrap();
        sm.transition(FrameType::ContractSigned).unwrap();

        sm.transition(FrameType::Error).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);
    }

    #[test]
    fn contract_heartbeat_during_contracting() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();

        sm.transition(FrameType::Heartbeat).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);
    }

    #[test]
    fn contract_heartbeat_during_engaged() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();
        sm.transition(FrameType::ContractRespond).unwrap();
        sm.transition(FrameType::ContractSigned).unwrap();

        sm.transition(FrameType::Heartbeat).unwrap();
        assert_eq!(sm.state(), ProtocolState::AgentEngaged);
    }

    #[test]
    fn contract_invalid_from_init() {
        let mut sm = ProtocolStateMachine::new();
        assert!(sm.transition(FrameType::ContractPropose).is_err());
    }

    #[test]
    fn contract_query_invalid_during_engaged() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();
        sm.transition(FrameType::ContractPropose).unwrap();
        sm.transition(FrameType::ContractRespond).unwrap();
        sm.transition(FrameType::ContractSigned).unwrap();

        // Can't run queries while an agent is engaged
        assert!(sm.transition(FrameType::Query).is_err());
    }

    #[test]
    fn contract_then_query_after_return() {
        let mut sm = ProtocolStateMachine::new();
        sm.transition(FrameType::Handshake).unwrap();

        // Full contract lifecycle
        sm.transition(FrameType::ContractPropose).unwrap();
        sm.transition(FrameType::ContractRespond).unwrap();
        sm.transition(FrameType::ContractSigned).unwrap();
        sm.transition(FrameType::AgentReturn).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Now can do normal queries
        sm.transition(FrameType::Query).unwrap();
        assert_eq!(sm.state(), ProtocolState::Streaming);
        sm.transition(FrameType::Done).unwrap();
        assert_eq!(sm.state(), ProtocolState::Ready);

        // Or another contract
        sm.transition(FrameType::ContractPropose).unwrap();
        assert_eq!(sm.state(), ProtocolState::Contracting);
    }
}
