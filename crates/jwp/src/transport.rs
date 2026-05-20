use crate::error::JwpError;
use crate::frame::JwpFrame;

/// Statistics tracked per transport connection.
///
/// Updated by each `send_frame` / `recv_frame` call. The handler layer
/// can read these to feed into `ConnectionProfile` for adaptive decisions.
#[derive(Debug, Clone, Default)]
pub struct TransportStats {
    /// Estimated round-trip time in microseconds (if measurable).
    pub estimated_rtt_us: Option<u64>,
    /// Estimated bandwidth in bytes per second (if measurable).
    pub estimated_bandwidth_bps: Option<u64>,
    /// Total bytes sent over this transport.
    pub bytes_sent: u64,
    /// Total bytes received over this transport.
    pub bytes_received: u64,
    /// Total frames sent.
    pub frames_sent: u64,
    /// Total frames received.
    pub frames_received: u64,
}

/// Transport-agnostic JWP frame I/O.
///
/// Implementations handle the byte-level reads/writes for a specific
/// transport (TCP, QUIC, Unix socket) and maintain per-connection statistics.
///
/// ```text
/// [Handler] ──send_frame──▶ [Transport] ──bytes──▶ [Network]
/// [Handler] ◀──recv_frame── [Transport] ◀──bytes── [Network]
/// ```
///
/// The transport layer is strictly I/O. Frame construction, sequence
/// management, energy tracking, and state machine logic belong in the
/// handler layer above.
pub trait Transport: Send {
    /// Send a frame over this transport.
    fn send_frame(
        &mut self,
        frame: JwpFrame,
    ) -> impl std::future::Future<Output = Result<(), JwpError>> + Send;

    /// Receive the next frame. Returns `None` on clean shutdown.
    fn recv_frame(
        &mut self,
    ) -> impl std::future::Future<Output = Result<Option<JwpFrame>, JwpError>> + Send;

    /// Transport identifier (e.g., `"tcp"`, `"quic"`, `"unix"`).
    fn transport_id(&self) -> &str;

    /// Current transport statistics.
    fn stats(&self) -> &TransportStats;

    /// Mutable access to transport statistics.
    fn stats_mut(&mut self) -> &mut TransportStats;
}
