//! NAT traversal concepts — STUN-like binding, NAT type detection, candidates.
//!
//! Pure Rust simulation of NAT traversal techniques. Models STUN-like
//! binding requests/responses, mapped address extraction, NAT type
//! detection (full cone, restricted, port-restricted, symmetric),
//! keep-alive, and connection candidate pair management.

use std::collections::HashMap;
use std::fmt;

// ── Address Types ─────────────────────────────────────────────

/// A transport address (IP + port) represented as strings for portability.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TransportAddr {
    pub ip: String,
    pub port: u16,
}

impl TransportAddr {
    pub fn new(ip: impl Into<String>, port: u16) -> Self {
        Self { ip: ip.into(), port }
    }
}

impl fmt::Display for TransportAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.ip, self.port)
    }
}

// ── NAT Types ─────────────────────────────────────────────────

/// Classification of NAT behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NatType {
    /// No NAT — public address.
    None,
    /// All requests from same internal address get same external mapping.
    FullCone,
    /// External host must first receive a packet from the internal host.
    RestrictedCone,
    /// Like restricted, but also requires matching port.
    PortRestricted,
    /// Different external mapping for each destination.
    Symmetric,
}

impl fmt::Display for NatType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::None => "No NAT",
            Self::FullCone => "Full Cone",
            Self::RestrictedCone => "Restricted Cone",
            Self::PortRestricted => "Port Restricted",
            Self::Symmetric => "Symmetric",
        };
        f.write_str(s)
    }
}

// ── STUN Messages ─────────────────────────────────────────────

/// STUN message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunMessageType {
    BindingRequest,
    BindingResponse,
    BindingErrorResponse,
}

/// A STUN attribute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StunAttribute {
    MappedAddress(TransportAddr),
    XorMappedAddress(TransportAddr),
    ChangeRequest { change_ip: bool, change_port: bool },
    ResponseOrigin(TransportAddr),
    OtherAddress(TransportAddr),
    Software(String),
    ErrorCode { code: u16, reason: String },
    Unknown { attr_type: u16, data: Vec<u8> },
}

/// A STUN message.
#[derive(Debug, Clone)]
pub struct StunMessage {
    pub msg_type: StunMessageType,
    pub transaction_id: [u8; 12],
    pub attributes: Vec<StunAttribute>,
}

impl StunMessage {
    pub fn binding_request(transaction_id: [u8; 12]) -> Self {
        Self {
            msg_type: StunMessageType::BindingRequest,
            transaction_id,
            attributes: Vec::new(),
        }
    }

    pub fn binding_response(transaction_id: [u8; 12], mapped_addr: TransportAddr) -> Self {
        Self {
            msg_type: StunMessageType::BindingResponse,
            transaction_id,
            attributes: vec![StunAttribute::MappedAddress(mapped_addr)],
        }
    }

    pub fn error_response(transaction_id: [u8; 12], code: u16, reason: String) -> Self {
        Self {
            msg_type: StunMessageType::BindingErrorResponse,
            transaction_id,
            attributes: vec![StunAttribute::ErrorCode { code, reason }],
        }
    }

    pub fn add_attribute(&mut self, attr: StunAttribute) {
        self.attributes.push(attr);
    }

    /// Extract the mapped address from the response.
    pub fn mapped_address(&self) -> Option<&TransportAddr> {
        for attr in &self.attributes {
            match attr {
                StunAttribute::MappedAddress(addr) => return Some(addr),
                StunAttribute::XorMappedAddress(addr) => return Some(addr),
                _ => {}
            }
        }
        None
    }

    /// Extract the error code if present.
    pub fn error_code(&self) -> Option<(u16, &str)> {
        for attr in &self.attributes {
            if let StunAttribute::ErrorCode { code, reason } = attr {
                return Some((*code, reason.as_str()));
            }
        }
        None
    }
}

// ── NAT Type Detector ─────────────────────────────────────────

/// Result of a single binding test.
#[derive(Debug, Clone)]
pub struct BindingResult {
    /// The server we sent the request to.
    pub server: TransportAddr,
    /// The mapped address returned.
    pub mapped: Option<TransportAddr>,
    /// Whether the response was received.
    pub received: bool,
}

/// Determines NAT type from a series of binding test results.
/// Implements the classic STUN NAT type detection algorithm (RFC 3489 style).
pub struct NatDetector {
    /// Our local address.
    pub local_addr: TransportAddr,
    /// Results from test I (primary server, primary port).
    pub test1: Option<BindingResult>,
    /// Results from test II (primary server, change IP+port flag).
    pub test2: Option<BindingResult>,
    /// Results from test III (alternate server, primary port).
    pub test3: Option<BindingResult>,
    /// Results from test I sent to alternate server.
    pub test1_alt: Option<BindingResult>,
}

impl NatDetector {
    pub fn new(local_addr: TransportAddr) -> Self {
        Self {
            local_addr,
            test1: None,
            test2: None,
            test3: None,
            test1_alt: None,
        }
    }

    pub fn set_test1(&mut self, result: BindingResult) {
        self.test1 = Some(result);
    }

    pub fn set_test2(&mut self, result: BindingResult) {
        self.test2 = Some(result);
    }

    pub fn set_test3(&mut self, result: BindingResult) {
        self.test3 = Some(result);
    }

    pub fn set_test1_alt(&mut self, result: BindingResult) {
        self.test1_alt = Some(result);
    }

    /// Detect the NAT type based on the collected test results.
    pub fn detect(&self) -> Option<NatType> {
        // Test I must succeed.
        let t1 = self.test1.as_ref()?;
        if !t1.received {
            return None; // UDP blocked.
        }
        let mapped1 = t1.mapped.as_ref()?;

        // Check if mapped address equals local address (no NAT).
        if mapped1.ip == self.local_addr.ip && mapped1.port == self.local_addr.port {
            // No NAT — check Test II for firewall.
            let t2 = self.test2.as_ref()?;
            if t2.received {
                return Some(NatType::None);
            } else {
                // Symmetric firewall, but no NAT.
                return Some(NatType::None);
            }
        }

        // Behind NAT. Check Test II.
        let t2 = self.test2.as_ref()?;
        if t2.received {
            return Some(NatType::FullCone);
        }

        // Test I to alternate server.
        let t1_alt = self.test1_alt.as_ref()?;
        if !t1_alt.received {
            return None;
        }
        let mapped_alt = t1_alt.mapped.as_ref()?;

        // If mapped addresses differ, it's symmetric.
        if mapped1.ip != mapped_alt.ip || mapped1.port != mapped_alt.port {
            return Some(NatType::Symmetric);
        }

        // Same mapping — check Test III.
        let t3 = self.test3.as_ref()?;
        if t3.received {
            Some(NatType::RestrictedCone)
        } else {
            Some(NatType::PortRestricted)
        }
    }
}

// ── Candidate Types ───────────────────────────────────────────

/// ICE candidate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateType {
    Host,
    ServerReflexive,
    PeerReflexive,
    Relay,
}

impl fmt::Display for CandidateType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Host => "host",
            Self::ServerReflexive => "srflx",
            Self::PeerReflexive => "prflx",
            Self::Relay => "relay",
        };
        f.write_str(s)
    }
}

/// An ICE candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub addr: TransportAddr,
    pub candidate_type: CandidateType,
    pub priority: u32,
    pub foundation: String,
    pub component_id: u8,
}

impl Candidate {
    pub fn new(
        addr: TransportAddr,
        candidate_type: CandidateType,
        priority: u32,
        foundation: impl Into<String>,
        component_id: u8,
    ) -> Self {
        Self {
            addr,
            candidate_type,
            priority,
            foundation: foundation.into(),
            component_id,
        }
    }

    /// Compute priority per RFC 5245 formula.
    /// priority = (2^24)*(type_pref) + (2^8)*(local_pref) + (256 - component_id)
    pub fn compute_priority(type_pref: u32, local_pref: u32, component_id: u8) -> u32 {
        (type_pref << 24)
            .saturating_add(local_pref << 8)
            .saturating_add(256u32.saturating_sub(component_id as u32))
    }
}

// ── Candidate Pair ────────────────────────────────────────────

/// State of a candidate pair check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairState {
    Frozen,
    Waiting,
    InProgress,
    Succeeded,
    Failed,
}

/// A pair of local and remote candidates.
#[derive(Debug, Clone)]
pub struct CandidatePair {
    pub local: Candidate,
    pub remote: Candidate,
    pub state: PairState,
    pub priority: u64,
    pub nominated: bool,
}

impl CandidatePair {
    pub fn new(local: Candidate, remote: Candidate, controlling: bool) -> Self {
        let priority = Self::pair_priority(local.priority, remote.priority, controlling);
        Self {
            local,
            remote,
            state: PairState::Frozen,
            priority,
            nominated: false,
        }
    }

    /// Compute pair priority per RFC 5245.
    fn pair_priority(g: u32, d: u32, controlling: bool) -> u64 {
        let (max_val, min_val) = if controlling {
            (g as u64, d as u64)
        } else {
            (d as u64, g as u64)
        };
        // pair_priority = 2^32 * min(G,D) + 2 * max(G,D) + (G>D ? 1 : 0)
        let m = max_val.min(min_val);
        let mx = max_val.max(min_val);
        let tie = if g > d { 1u64 } else { 0u64 };
        (1u64 << 32).saturating_mul(m).saturating_add(2 * mx).saturating_add(tie)
    }
}

// ── Candidate Pair List ───────────────────────────────────────

/// Manages candidate pairs for connectivity checks.
#[derive(Debug)]
pub struct CandidatePairList {
    pairs: Vec<CandidatePair>,
}

impl CandidatePairList {
    pub fn new() -> Self {
        Self { pairs: Vec::new() }
    }

    /// Add a candidate pair.
    pub fn add(&mut self, pair: CandidatePair) {
        self.pairs.push(pair);
    }

    /// Sort pairs by priority (highest first).
    pub fn sort_by_priority(&mut self) {
        self.pairs.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get the next pair to check (first Waiting pair).
    pub fn next_to_check(&mut self) -> Option<&mut CandidatePair> {
        self.pairs.iter_mut().find(|p| p.state == PairState::Waiting)
    }

    /// Unfreeze all Frozen pairs (set to Waiting).
    pub fn unfreeze_all(&mut self) {
        for pair in &mut self.pairs {
            if pair.state == PairState::Frozen {
                pair.state = PairState::Waiting;
            }
        }
    }

    /// Get the best succeeded pair (highest priority).
    pub fn best_succeeded(&self) -> Option<&CandidatePair> {
        self.pairs
            .iter()
            .filter(|p| p.state == PairState::Succeeded)
            .max_by_key(|p| p.priority)
    }

    /// Get the nominated pair.
    pub fn nominated(&self) -> Option<&CandidatePair> {
        self.pairs.iter().find(|p| p.nominated && p.state == PairState::Succeeded)
    }

    /// Count pairs in a given state.
    pub fn count_in_state(&self, state: PairState) -> usize {
        self.pairs.iter().filter(|p| p.state == state).count()
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn pairs(&self) -> &[CandidatePair] {
        &self.pairs
    }

    pub fn pairs_mut(&mut self) -> &mut [CandidatePair] {
        &mut self.pairs
    }
}

impl Default for CandidatePairList {
    fn default() -> Self {
        Self::new()
    }
}

// ── Keep-Alive Tracker ────────────────────────────────────────

/// Tracks keep-alive state for NAT binding refresh.
#[derive(Debug, Clone)]
pub struct KeepAlive {
    /// Interval between keep-alive probes in milliseconds.
    pub interval_ms: u64,
    /// Timestamp of last keep-alive sent.
    pub last_sent_ms: u64,
    /// Timestamp of last keep-alive response received.
    pub last_recv_ms: u64,
    /// Number of consecutive keep-alives without response.
    pub missed_count: u32,
    /// Maximum misses before declaring binding lost.
    pub max_misses: u32,
    /// Whether the binding is considered alive.
    pub alive: bool,
}

impl KeepAlive {
    pub fn new(interval_ms: u64, max_misses: u32) -> Self {
        Self {
            interval_ms,
            last_sent_ms: 0,
            last_recv_ms: 0,
            missed_count: 0,
            max_misses,
            alive: true,
        }
    }

    /// Check if a keep-alive should be sent now.
    pub fn should_send(&self, now_ms: u64) -> bool {
        self.alive && now_ms.saturating_sub(self.last_sent_ms) >= self.interval_ms
    }

    /// Record that a keep-alive was sent.
    pub fn record_sent(&mut self, now_ms: u64) {
        self.last_sent_ms = now_ms;
        self.missed_count += 1;
        if self.missed_count > self.max_misses {
            self.alive = false;
        }
    }

    /// Record that a keep-alive response was received.
    pub fn record_recv(&mut self, now_ms: u64) {
        self.last_recv_ms = now_ms;
        self.missed_count = 0;
        self.alive = true;
    }

    /// Reset the keep-alive tracker.
    pub fn reset(&mut self) {
        self.last_sent_ms = 0;
        self.last_recv_ms = 0;
        self.missed_count = 0;
        self.alive = true;
    }
}

// ── NAT Mapping Table ─────────────────────────────────────────

/// Simulates a NAT mapping table (for testing NAT traversal logic).
#[derive(Debug)]
pub struct NatMappingTable {
    /// Maps (internal_addr) -> external_addr for full-cone style.
    mappings: HashMap<String, TransportAddr>,
    /// Next external port to allocate.
    next_port: u16,
    /// External IP address.
    external_ip: String,
}

impl NatMappingTable {
    pub fn new(external_ip: impl Into<String>, start_port: u16) -> Self {
        Self {
            mappings: HashMap::new(),
            next_port: start_port,
            external_ip: external_ip.into(),
        }
    }

    /// Get or create a mapping for the given internal address.
    pub fn get_or_create(&mut self, internal: &TransportAddr) -> TransportAddr {
        let key = format!("{}", internal);
        if let Some(ext) = self.mappings.get(&key) {
            return ext.clone();
        }
        let ext = TransportAddr::new(self.external_ip.clone(), self.next_port);
        self.next_port = self.next_port.wrapping_add(1);
        self.mappings.insert(key, ext.clone());
        ext
    }

    /// Look up the external mapping for an internal address.
    pub fn lookup(&self, internal: &TransportAddr) -> Option<&TransportAddr> {
        let key = format!("{}", internal);
        self.mappings.get(&key)
    }

    /// Remove a mapping.
    pub fn remove(&mut self, internal: &TransportAddr) -> bool {
        let key = format!("{}", internal);
        self.mappings.remove(&key).is_some()
    }

    /// Number of active mappings.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn addr(ip: &str, port: u16) -> TransportAddr {
        TransportAddr::new(ip, port)
    }

    #[test]
    fn test_transport_addr_display() {
        let a = addr("192.168.1.1", 8080);
        assert_eq!(format!("{}", a), "192.168.1.1:8080");
    }

    #[test]
    fn test_nat_type_display() {
        assert_eq!(format!("{}", NatType::FullCone), "Full Cone");
        assert_eq!(format!("{}", NatType::Symmetric), "Symmetric");
    }

    #[test]
    fn test_stun_binding_request() {
        let msg = StunMessage::binding_request([1; 12]);
        assert_eq!(msg.msg_type, StunMessageType::BindingRequest);
        assert!(msg.mapped_address().is_none());
    }

    #[test]
    fn test_stun_binding_response() {
        let mapped = addr("203.0.113.1", 5060);
        let msg = StunMessage::binding_response([2; 12], mapped.clone());
        assert_eq!(msg.mapped_address().unwrap(), &mapped);
    }

    #[test]
    fn test_stun_error_response() {
        let msg = StunMessage::error_response([3; 12], 401, "Unauthorized".into());
        let (code, reason) = msg.error_code().unwrap();
        assert_eq!(code, 401);
        assert_eq!(reason, "Unauthorized");
    }

    #[test]
    fn test_stun_add_attribute() {
        let mut msg = StunMessage::binding_response([4; 12], addr("1.2.3.4", 100));
        msg.add_attribute(StunAttribute::Software("test".into()));
        assert_eq!(msg.attributes.len(), 2);
    }

    #[test]
    fn test_nat_detect_no_nat() {
        let local = addr("203.0.113.50", 5000);
        let mut det = NatDetector::new(local.clone());
        det.set_test1(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: Some(local.clone()),
            received: true,
        });
        det.set_test2(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: None,
            received: true,
        });
        assert_eq!(det.detect(), Some(NatType::None));
    }

    #[test]
    fn test_nat_detect_full_cone() {
        let local = addr("192.168.1.100", 5000);
        let mapped = addr("203.0.113.50", 12345);
        let mut det = NatDetector::new(local);
        det.set_test1(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: Some(mapped.clone()),
            received: true,
        });
        det.set_test2(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: Some(mapped),
            received: true,
        });
        assert_eq!(det.detect(), Some(NatType::FullCone));
    }

    #[test]
    fn test_nat_detect_symmetric() {
        let local = addr("192.168.1.100", 5000);
        let mapped1 = addr("203.0.113.50", 12345);
        let mapped2 = addr("203.0.113.50", 12346);
        let mut det = NatDetector::new(local);
        det.set_test1(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: Some(mapped1),
            received: true,
        });
        det.set_test2(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: None,
            received: false,
        });
        det.set_test1_alt(BindingResult {
            server: addr("198.51.100.2", 3478),
            mapped: Some(mapped2),
            received: true,
        });
        assert_eq!(det.detect(), Some(NatType::Symmetric));
    }

    #[test]
    fn test_nat_detect_restricted() {
        let local = addr("192.168.1.100", 5000);
        let mapped = addr("203.0.113.50", 12345);
        let mut det = NatDetector::new(local);
        det.set_test1(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: Some(mapped.clone()),
            received: true,
        });
        det.set_test2(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: None,
            received: false,
        });
        det.set_test1_alt(BindingResult {
            server: addr("198.51.100.2", 3478),
            mapped: Some(mapped),
            received: true,
        });
        det.set_test3(BindingResult {
            server: addr("198.51.100.2", 3478),
            mapped: None,
            received: true,
        });
        assert_eq!(det.detect(), Some(NatType::RestrictedCone));
    }

    #[test]
    fn test_nat_detect_port_restricted() {
        let local = addr("192.168.1.100", 5000);
        let mapped = addr("203.0.113.50", 12345);
        let mut det = NatDetector::new(local);
        det.set_test1(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: Some(mapped.clone()),
            received: true,
        });
        det.set_test2(BindingResult {
            server: addr("198.51.100.1", 3478),
            mapped: None,
            received: false,
        });
        det.set_test1_alt(BindingResult {
            server: addr("198.51.100.2", 3478),
            mapped: Some(mapped),
            received: true,
        });
        det.set_test3(BindingResult {
            server: addr("198.51.100.2", 3478),
            mapped: None,
            received: false,
        });
        assert_eq!(det.detect(), Some(NatType::PortRestricted));
    }

    #[test]
    fn test_candidate_priority() {
        let prio = Candidate::compute_priority(126, 65535, 1);
        assert!(prio > 0);
        // host candidates (type_pref=126) > srflx (type_pref=100)
        let host = Candidate::compute_priority(126, 65535, 1);
        let srflx = Candidate::compute_priority(100, 65535, 1);
        assert!(host > srflx);
    }

    #[test]
    fn test_candidate_pair_list() {
        let local = Candidate::new(addr("192.168.1.1", 5000), CandidateType::Host, 1000, "f1", 1);
        let remote = Candidate::new(addr("10.0.0.1", 5000), CandidateType::Host, 900, "f2", 1);
        let pair = CandidatePair::new(local, remote, true);

        let mut list = CandidatePairList::new();
        list.add(pair);
        assert_eq!(list.len(), 1);
        assert_eq!(list.count_in_state(PairState::Frozen), 1);
    }

    #[test]
    fn test_candidate_pair_unfreeze() {
        let local = Candidate::new(addr("192.168.1.1", 5000), CandidateType::Host, 1000, "f1", 1);
        let remote = Candidate::new(addr("10.0.0.1", 5000), CandidateType::Host, 900, "f2", 1);
        let pair = CandidatePair::new(local, remote, true);

        let mut list = CandidatePairList::new();
        list.add(pair);
        list.unfreeze_all();
        assert_eq!(list.count_in_state(PairState::Waiting), 1);
        assert!(list.next_to_check().is_some());
    }

    #[test]
    fn test_candidate_pair_succeeded() {
        let local = Candidate::new(addr("192.168.1.1", 5000), CandidateType::Host, 1000, "f1", 1);
        let remote = Candidate::new(addr("10.0.0.1", 5000), CandidateType::Host, 900, "f2", 1);
        let mut pair = CandidatePair::new(local, remote, true);
        pair.state = PairState::Succeeded;
        pair.nominated = true;

        let mut list = CandidatePairList::new();
        list.add(pair);
        assert!(list.best_succeeded().is_some());
        assert!(list.nominated().is_some());
    }

    #[test]
    fn test_keep_alive_should_send() {
        let ka = KeepAlive::new(5000, 3);
        assert!(ka.should_send(5000));
        assert!(!ka.should_send(3000));
    }

    #[test]
    fn test_keep_alive_missed() {
        let mut ka = KeepAlive::new(5000, 2);
        ka.record_sent(5000);
        assert!(ka.alive);
        ka.record_sent(10000);
        assert!(ka.alive);
        ka.record_sent(15000);
        assert!(!ka.alive);
    }

    #[test]
    fn test_keep_alive_recv_resets() {
        let mut ka = KeepAlive::new(5000, 2);
        ka.record_sent(5000);
        ka.record_sent(10000);
        assert_eq!(ka.missed_count, 2);
        ka.record_recv(11000);
        assert_eq!(ka.missed_count, 0);
        assert!(ka.alive);
    }

    #[test]
    fn test_nat_mapping_table() {
        let mut table = NatMappingTable::new("203.0.113.1", 10000);
        let internal = addr("192.168.1.100", 5000);
        let ext1 = table.get_or_create(&internal);
        assert_eq!(ext1.port, 10000);
        let ext2 = table.get_or_create(&internal);
        assert_eq!(ext1, ext2); // Same mapping.
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn test_nat_mapping_table_remove() {
        let mut table = NatMappingTable::new("203.0.113.1", 10000);
        let internal = addr("192.168.1.100", 5000);
        table.get_or_create(&internal);
        assert!(table.remove(&internal));
        assert!(table.is_empty());
    }

    #[test]
    fn test_candidate_type_display() {
        assert_eq!(format!("{}", CandidateType::Host), "host");
        assert_eq!(format!("{}", CandidateType::ServerReflexive), "srflx");
        assert_eq!(format!("{}", CandidateType::Relay), "relay");
    }
}
