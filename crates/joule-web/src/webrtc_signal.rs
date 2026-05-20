//! WebRTC signaling — SDP offer/answer modeling, ICE candidate exchange,
//! session description parsing, codec negotiation, and STUN message format.
//!
//! Pure-Rust signaling layer that models the WebRTC handshake without any
//! actual network I/O. Callers wire signaling messages to their own transport.

use std::collections::HashMap;
use std::fmt;

// ── SDP types ──────────────────────────────────────────────────────

/// Session Description Protocol type (offer vs answer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SdpType {
    Offer,
    Answer,
    Pranswer,
    Rollback,
}

impl fmt::Display for SdpType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Offer => write!(f, "offer"),
            Self::Answer => write!(f, "answer"),
            Self::Pranswer => write!(f, "pranswer"),
            Self::Rollback => write!(f, "rollback"),
        }
    }
}

/// A session description (SDP blob + type).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SessionDescription {
    pub sdp_type: SdpType,
    pub sdp: String,
}

impl SessionDescription {
    pub fn new(sdp_type: SdpType, sdp: impl Into<String>) -> Self {
        Self { sdp_type, sdp: sdp.into() }
    }
}

// ── Media description ──────────────────────────────────────────────

/// Media type within an SDP description.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaKind {
    Audio,
    Video,
    Application,
}

impl fmt::Display for MediaKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Audio => write!(f, "audio"),
            Self::Video => write!(f, "video"),
            Self::Application => write!(f, "application"),
        }
    }
}

/// Direction attribute for a media line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MediaDirection {
    SendRecv,
    SendOnly,
    RecvOnly,
    Inactive,
}

/// A codec description with payload type and parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Codec {
    pub payload_type: u8,
    pub name: String,
    pub clock_rate: u32,
    pub channels: Option<u8>,
    pub fmtp: HashMap<String, String>,
}

impl Codec {
    pub fn new(payload_type: u8, name: impl Into<String>, clock_rate: u32) -> Self {
        Self {
            payload_type,
            name: name.into(),
            clock_rate,
            channels: None,
            fmtp: HashMap::new(),
        }
    }

    pub fn with_channels(mut self, ch: u8) -> Self {
        self.channels = Some(ch);
        self
    }

    pub fn with_fmtp(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.fmtp.insert(key.into(), val.into());
        self
    }

    /// Format the codec as an SDP rtpmap line value: `<name>/<clock>[/<ch>]`.
    pub fn rtpmap_value(&self) -> String {
        match self.channels {
            Some(ch) => format!("{}/{}/{}", self.name, self.clock_rate, ch),
            None => format!("{}/{}", self.name, self.clock_rate),
        }
    }
}

/// A media section within an SDP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSection {
    pub kind: MediaKind,
    pub port: u16,
    pub protocol: String,
    pub direction: MediaDirection,
    pub codecs: Vec<Codec>,
    pub mid: Option<String>,
}

impl MediaSection {
    pub fn new(kind: MediaKind) -> Self {
        Self {
            kind,
            port: 9,
            protocol: "UDP/TLS/RTP/SAVPF".to_string(),
            direction: MediaDirection::SendRecv,
            codecs: Vec::new(),
            mid: None,
        }
    }

    pub fn add_codec(&mut self, codec: Codec) {
        self.codecs.push(codec);
    }

    pub fn payload_types(&self) -> Vec<u8> {
        self.codecs.iter().map(|c| c.payload_type).collect()
    }
}

// ── SDP parser (minimal) ──────────────────────────────────────────

/// Parsed SDP contents.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedSdp {
    pub version: u8,
    pub session_name: String,
    pub origin_username: String,
    pub origin_session_id: String,
    pub media_sections: Vec<MediaSection>,
    pub ice_ufrag: Option<String>,
    pub ice_pwd: Option<String>,
    pub fingerprint: Option<String>,
    pub attributes: HashMap<String, String>,
}

impl ParsedSdp {
    /// Parse a raw SDP string into structured data.
    pub fn parse(sdp: &str) -> Result<Self, SdpParseError> {
        let mut version = 0u8;
        let mut session_name = String::new();
        let mut origin_username = String::new();
        let mut origin_session_id = String::new();
        let mut ice_ufrag: Option<String> = None;
        let mut ice_pwd: Option<String> = None;
        let mut fingerprint: Option<String> = None;
        let mut attributes = HashMap::new();
        let mut media_sections: Vec<MediaSection> = Vec::new();

        for line in sdp.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if line.len() < 2 || line.as_bytes()[1] != b'=' {
                continue; // skip malformed
            }
            let kind_byte = line.as_bytes()[0];
            let value = &line[2..];

            match kind_byte {
                b'v' => {
                    version = value.parse::<u8>().map_err(|_| SdpParseError::InvalidVersion)?;
                }
                b's' => {
                    session_name = value.to_string();
                }
                b'o' => {
                    let parts: Vec<&str> = value.split_whitespace().collect();
                    if parts.len() >= 2 {
                        origin_username = parts[0].to_string();
                        origin_session_id = parts[1].to_string();
                    }
                }
                b'm' => {
                    let parts: Vec<&str> = value.split_whitespace().collect();
                    if parts.len() < 3 {
                        return Err(SdpParseError::InvalidMediaLine);
                    }
                    let mk = match parts[0] {
                        "audio" => MediaKind::Audio,
                        "video" => MediaKind::Video,
                        _ => MediaKind::Application,
                    };
                    let port = parts[1].parse::<u16>().unwrap_or(9);
                    let proto = parts[2].to_string();
                    let mut section = MediaSection::new(mk);
                    section.port = port;
                    section.protocol = proto;
                    // Remaining tokens are payload types
                    for pt_str in &parts[3..] {
                        if let Ok(pt) = pt_str.parse::<u8>() {
                            section.add_codec(Codec::new(pt, "", 0));
                        }
                    }
                    media_sections.push(section);
                }
                b'a' => {
                    if let Some((attr_name, attr_val)) = value.split_once(':') {
                        match attr_name {
                            "ice-ufrag" => ice_ufrag = Some(attr_val.to_string()),
                            "ice-pwd" => ice_pwd = Some(attr_val.to_string()),
                            "fingerprint" => fingerprint = Some(attr_val.to_string()),
                            "rtpmap" => {
                                // e.g. "111 opus/48000/2"
                                if let Some(section) = media_sections.last_mut() {
                                    if let Some((pt_str, codec_info)) = attr_val.split_once(' ') {
                                        if let Ok(pt) = pt_str.parse::<u8>() {
                                            let parts: Vec<&str> = codec_info.split('/').collect();
                                            let name = parts.first().unwrap_or(&"").to_string();
                                            let clock = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);
                                            let ch = parts.get(2).and_then(|s| s.parse().ok());
                                            // Update existing codec stub or add new
                                            if let Some(c) = section.codecs.iter_mut().find(|c| c.payload_type == pt) {
                                                c.name = name;
                                                c.clock_rate = clock;
                                                c.channels = ch;
                                            } else {
                                                let mut codec = Codec::new(pt, name, clock);
                                                codec.channels = ch;
                                                section.add_codec(codec);
                                            }
                                        }
                                    }
                                }
                            }
                            "mid" => {
                                if let Some(section) = media_sections.last_mut() {
                                    section.mid = Some(attr_val.to_string());
                                }
                            }
                            "sendrecv" | "sendonly" | "recvonly" | "inactive" => {}
                            _ => { attributes.insert(attr_name.to_string(), attr_val.to_string()); }
                        }
                    } else {
                        // Attribute-only flags like a=sendrecv
                        if let Some(section) = media_sections.last_mut() {
                            match value {
                                "sendrecv" => section.direction = MediaDirection::SendRecv,
                                "sendonly" => section.direction = MediaDirection::SendOnly,
                                "recvonly" => section.direction = MediaDirection::RecvOnly,
                                "inactive" => section.direction = MediaDirection::Inactive,
                                _ => {}
                            }
                        }
                    }
                }
                _ => {} // ignore c=, t=, etc.
            }
        }

        Ok(Self {
            version,
            session_name,
            origin_username,
            origin_session_id,
            media_sections,
            ice_ufrag,
            ice_pwd,
            fingerprint,
            attributes,
        })
    }
}

/// Errors from parsing SDP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SdpParseError {
    InvalidVersion,
    InvalidMediaLine,
}

impl fmt::Display for SdpParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidVersion => write!(f, "invalid SDP version"),
            Self::InvalidMediaLine => write!(f, "invalid SDP media line"),
        }
    }
}

// ── ICE candidate ──────────────────────────────────────────────────

/// ICE candidate transport protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceProtocol {
    Udp,
    Tcp,
}

/// ICE candidate type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceCandidateType {
    Host,
    Srflx,
    Prflx,
    Relay,
}

/// An ICE candidate for connectivity checking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IceCandidate {
    pub foundation: String,
    pub component: u8,
    pub protocol: IceProtocol,
    pub priority: u32,
    pub address: String,
    pub port: u16,
    pub candidate_type: IceCandidateType,
    pub related_address: Option<String>,
    pub related_port: Option<u16>,
    pub sdp_mid: Option<String>,
    pub sdp_mline_index: Option<u16>,
}

impl IceCandidate {
    pub fn host(address: impl Into<String>, port: u16, protocol: IceProtocol) -> Self {
        Self {
            foundation: "1".to_string(),
            component: 1,
            protocol,
            priority: Self::compute_priority(IceCandidateType::Host, 1, 1),
            address: address.into(),
            port,
            candidate_type: IceCandidateType::Host,
            related_address: None,
            related_port: None,
            sdp_mid: None,
            sdp_mline_index: None,
        }
    }

    /// Compute ICE priority per RFC 5245: (2^24)*type + (2^8)*local + (2^0)*(256-component).
    pub fn compute_priority(ctype: IceCandidateType, local_pref: u32, component: u8) -> u32 {
        let type_pref: u32 = match ctype {
            IceCandidateType::Host => 126,
            IceCandidateType::Srflx => 100,
            IceCandidateType::Prflx => 110,
            IceCandidateType::Relay => 0,
        };
        (type_pref << 24) | (local_pref << 8) | (256 - component as u32)
    }

    /// Serialize to SDP candidate attribute format.
    pub fn to_sdp_attribute(&self) -> String {
        let proto_str = match self.protocol {
            IceProtocol::Udp => "udp",
            IceProtocol::Tcp => "tcp",
        };
        let typ_str = match self.candidate_type {
            IceCandidateType::Host => "host",
            IceCandidateType::Srflx => "srflx",
            IceCandidateType::Prflx => "prflx",
            IceCandidateType::Relay => "relay",
        };
        format!(
            "candidate:{} {} {} {} {} {} typ {}",
            self.foundation, self.component, proto_str, self.priority,
            self.address, self.port, typ_str
        )
    }
}

// ── STUN message ───────────────────────────────────────────────────

/// STUN message type (RFC 5389).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunMessageType {
    BindingRequest,
    BindingResponse,
    BindingErrorResponse,
}

impl StunMessageType {
    pub fn to_u16(self) -> u16 {
        match self {
            Self::BindingRequest => 0x0001,
            Self::BindingResponse => 0x0101,
            Self::BindingErrorResponse => 0x0111,
        }
    }

    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0x0001 => Some(Self::BindingRequest),
            0x0101 => Some(Self::BindingResponse),
            0x0111 => Some(Self::BindingErrorResponse),
            _ => None,
        }
    }
}

/// STUN attribute types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StunAttrType {
    MappedAddress,
    XorMappedAddress,
    Username,
    MessageIntegrity,
    Fingerprint,
    Software,
    Priority,
    UseCandidate,
    IceControlled,
    IceControlling,
}

impl StunAttrType {
    pub fn to_u16(self) -> u16 {
        match self {
            Self::MappedAddress => 0x0001,
            Self::XorMappedAddress => 0x0020,
            Self::Username => 0x0006,
            Self::MessageIntegrity => 0x0008,
            Self::Fingerprint => 0x8028,
            Self::Software => 0x8022,
            Self::Priority => 0x0024,
            Self::UseCandidate => 0x0025,
            Self::IceControlled => 0x8029,
            Self::IceControlling => 0x802A,
        }
    }
}

/// A STUN attribute (type + raw value bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StunAttribute {
    pub attr_type: u16,
    pub value: Vec<u8>,
}

/// STUN magic cookie constant (RFC 5389).
pub const STUN_MAGIC_COOKIE: u32 = 0x2112A442;

/// A STUN message with header and attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StunMessage {
    pub msg_type: StunMessageType,
    pub transaction_id: [u8; 12],
    pub attributes: Vec<StunAttribute>,
}

impl StunMessage {
    pub fn new_binding_request(transaction_id: [u8; 12]) -> Self {
        Self {
            msg_type: StunMessageType::BindingRequest,
            transaction_id,
            attributes: Vec::new(),
        }
    }

    pub fn add_attribute(&mut self, attr_type: u16, value: Vec<u8>) {
        self.attributes.push(StunAttribute { attr_type, value });
    }

    /// Serialize to bytes (STUN header + attributes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut attr_bytes = Vec::new();
        for attr in &self.attributes {
            let len = attr.value.len() as u16;
            attr_bytes.extend_from_slice(&attr.attr_type.to_be_bytes());
            attr_bytes.extend_from_slice(&len.to_be_bytes());
            attr_bytes.extend_from_slice(&attr.value);
            // STUN attributes are padded to 4-byte boundaries
            let pad = (4 - (attr.value.len() % 4)) % 4;
            attr_bytes.extend(std::iter::repeat(0u8).take(pad));
        }

        let mut buf = Vec::with_capacity(20 + attr_bytes.len());
        buf.extend_from_slice(&self.msg_type.to_u16().to_be_bytes());
        buf.extend_from_slice(&(attr_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(&STUN_MAGIC_COOKIE.to_be_bytes());
        buf.extend_from_slice(&self.transaction_id);
        buf.extend_from_slice(&attr_bytes);
        buf
    }

    /// Parse from bytes. Returns None on invalid data.
    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 20 {
            return None;
        }
        let msg_type_val = u16::from_be_bytes([data[0], data[1]]);
        let msg_type = StunMessageType::from_u16(msg_type_val)?;
        let attr_len = u16::from_be_bytes([data[2], data[3]]) as usize;
        let cookie = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        if cookie != STUN_MAGIC_COOKIE {
            return None;
        }
        let mut tid = [0u8; 12];
        tid.copy_from_slice(&data[8..20]);

        if data.len() < 20 + attr_len {
            return None;
        }

        let mut attributes = Vec::new();
        let mut offset = 20;
        let end = 20 + attr_len;
        while offset + 4 <= end {
            let at = u16::from_be_bytes([data[offset], data[offset + 1]]);
            let al = u16::from_be_bytes([data[offset + 2], data[offset + 3]]) as usize;
            offset += 4;
            if offset + al > end {
                break;
            }
            let value = data[offset..offset + al].to_vec();
            attributes.push(StunAttribute { attr_type: at, value });
            offset += al;
            offset += (4 - (al % 4)) % 4; // skip padding
        }

        Some(Self { msg_type, transaction_id: tid, attributes })
    }
}

// ── Codec negotiation ──────────────────────────────────────────────

/// Result of codec negotiation between local and remote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NegotiatedCodec {
    pub payload_type: u8,
    pub name: String,
    pub clock_rate: u32,
    pub channels: Option<u8>,
}

/// Negotiate codecs by matching local preferences against remote offered codecs.
/// Returns codecs common to both, in local preference order.
pub fn negotiate_codecs(local: &[Codec], remote: &[Codec]) -> Vec<NegotiatedCodec> {
    let mut result = Vec::new();
    for lc in local {
        for rc in remote {
            if lc.name.eq_ignore_ascii_case(&rc.name) && lc.clock_rate == rc.clock_rate {
                let ch = lc.channels.or(rc.channels);
                result.push(NegotiatedCodec {
                    payload_type: rc.payload_type,
                    name: lc.name.clone(),
                    clock_rate: lc.clock_rate,
                    channels: ch,
                });
                break;
            }
        }
    }
    result
}

// ── Signaling state machine ────────────────────────────────────────

/// WebRTC signaling state (mirrors RTCSignalingState).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalingState {
    Stable,
    HaveLocalOffer,
    HaveRemoteOffer,
    HaveLocalPranswer,
    HaveRemotePranswer,
    Closed,
}

/// Events emitted by the signaling state machine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignalEvent {
    SendOffer(SessionDescription),
    SendAnswer(SessionDescription),
    AddIceCandidate(IceCandidate),
    NegotiationComplete,
    Error(String),
}

/// Signaling state machine that tracks offer/answer exchange.
#[derive(Debug)]
pub struct SignalingMachine {
    pub state: SignalingState,
    pub local_description: Option<SessionDescription>,
    pub remote_description: Option<SessionDescription>,
    pub local_candidates: Vec<IceCandidate>,
    pub remote_candidates: Vec<IceCandidate>,
    events: Vec<SignalEvent>,
}

impl SignalingMachine {
    pub fn new() -> Self {
        Self {
            state: SignalingState::Stable,
            local_description: None,
            remote_description: None,
            local_candidates: Vec::new(),
            remote_candidates: Vec::new(),
            events: Vec::new(),
        }
    }

    /// Drain pending events.
    pub fn take_events(&mut self) -> Vec<SignalEvent> {
        std::mem::take(&mut self.events)
    }

    /// Set local description (offer or answer).
    pub fn set_local_description(&mut self, desc: SessionDescription) {
        match (self.state, desc.sdp_type) {
            (SignalingState::Stable, SdpType::Offer) => {
                self.state = SignalingState::HaveLocalOffer;
                self.events.push(SignalEvent::SendOffer(desc.clone()));
                self.local_description = Some(desc);
            }
            (SignalingState::HaveRemoteOffer, SdpType::Answer) => {
                self.state = SignalingState::Stable;
                self.events.push(SignalEvent::SendAnswer(desc.clone()));
                self.events.push(SignalEvent::NegotiationComplete);
                self.local_description = Some(desc);
            }
            (SignalingState::HaveRemoteOffer, SdpType::Pranswer) => {
                self.state = SignalingState::HaveLocalPranswer;
                self.local_description = Some(desc);
            }
            (_, SdpType::Rollback) => {
                self.state = SignalingState::Stable;
                self.local_description = None;
            }
            _ => {
                self.events.push(SignalEvent::Error(format!(
                    "invalid state transition: {:?} + local {:?}",
                    self.state, desc.sdp_type
                )));
            }
        }
    }

    /// Set remote description (offer or answer).
    pub fn set_remote_description(&mut self, desc: SessionDescription) {
        match (self.state, desc.sdp_type) {
            (SignalingState::Stable, SdpType::Offer) => {
                self.state = SignalingState::HaveRemoteOffer;
                self.remote_description = Some(desc);
            }
            (SignalingState::HaveLocalOffer, SdpType::Answer) => {
                self.state = SignalingState::Stable;
                self.events.push(SignalEvent::NegotiationComplete);
                self.remote_description = Some(desc);
            }
            (SignalingState::HaveLocalOffer, SdpType::Pranswer) => {
                self.state = SignalingState::HaveRemotePranswer;
                self.remote_description = Some(desc);
            }
            (_, SdpType::Rollback) => {
                self.state = SignalingState::Stable;
                self.remote_description = None;
            }
            _ => {
                self.events.push(SignalEvent::Error(format!(
                    "invalid state transition: {:?} + remote {:?}",
                    self.state, desc.sdp_type
                )));
            }
        }
    }

    /// Add a local ICE candidate.
    pub fn add_local_candidate(&mut self, candidate: IceCandidate) {
        self.events.push(SignalEvent::AddIceCandidate(candidate.clone()));
        self.local_candidates.push(candidate);
    }

    /// Add a remote ICE candidate.
    pub fn add_remote_candidate(&mut self, candidate: IceCandidate) {
        self.remote_candidates.push(candidate);
    }

    /// Close the signaling session.
    pub fn close(&mut self) {
        self.state = SignalingState::Closed;
    }
}

impl Default for SignalingMachine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── SDP type display ───────────────────────────────────────────

    #[test]
    fn sdp_type_display() {
        assert_eq!(SdpType::Offer.to_string(), "offer");
        assert_eq!(SdpType::Answer.to_string(), "answer");
        assert_eq!(SdpType::Pranswer.to_string(), "pranswer");
        assert_eq!(SdpType::Rollback.to_string(), "rollback");
    }

    // ── Session description ────────────────────────────────────────

    #[test]
    fn session_description_creation() {
        let sd = SessionDescription::new(SdpType::Offer, "v=0\r\n");
        assert_eq!(sd.sdp_type, SdpType::Offer);
        assert_eq!(sd.sdp, "v=0\r\n");
    }

    // ── Codec ──────────────────────────────────────────────────────

    #[test]
    fn codec_rtpmap_value() {
        let c = Codec::new(111, "opus", 48000).with_channels(2);
        assert_eq!(c.rtpmap_value(), "opus/48000/2");

        let c2 = Codec::new(96, "VP8", 90000);
        assert_eq!(c2.rtpmap_value(), "VP8/90000");
    }

    #[test]
    fn codec_fmtp() {
        let c = Codec::new(111, "opus", 48000)
            .with_fmtp("minptime", "10")
            .with_fmtp("useinbandfec", "1");
        assert_eq!(c.fmtp.get("minptime").unwrap(), "10");
        assert_eq!(c.fmtp.get("useinbandfec").unwrap(), "1");
    }

    // ── Media section ──────────────────────────────────────────────

    #[test]
    fn media_section_payload_types() {
        let mut section = MediaSection::new(MediaKind::Audio);
        section.add_codec(Codec::new(111, "opus", 48000));
        section.add_codec(Codec::new(0, "PCMU", 8000));
        assert_eq!(section.payload_types(), vec![111, 0]);
    }

    #[test]
    fn media_kind_display() {
        assert_eq!(MediaKind::Audio.to_string(), "audio");
        assert_eq!(MediaKind::Video.to_string(), "video");
        assert_eq!(MediaKind::Application.to_string(), "application");
    }

    // ── SDP parsing ────────────────────────────────────────────────

    fn sample_sdp() -> String {
        [
            "v=0",
            "o=- 123456 2 IN IP4 127.0.0.1",
            "s=-",
            "t=0 0",
            "a=ice-ufrag:abcd",
            "a=ice-pwd:efghijklmnop",
            "a=fingerprint:sha-256 AA:BB:CC",
            "m=audio 9 UDP/TLS/RTP/SAVPF 111 0",
            "a=mid:0",
            "a=rtpmap:111 opus/48000/2",
            "a=rtpmap:0 PCMU/8000",
            "a=sendrecv",
            "m=video 9 UDP/TLS/RTP/SAVPF 96",
            "a=mid:1",
            "a=rtpmap:96 VP8/90000",
            "a=recvonly",
        ]
        .join("\r\n")
    }

    #[test]
    fn parse_sdp_version_and_session() {
        let parsed = ParsedSdp::parse(&sample_sdp()).unwrap();
        assert_eq!(parsed.version, 0);
        assert_eq!(parsed.session_name, "-");
        assert_eq!(parsed.origin_username, "-");
        assert_eq!(parsed.origin_session_id, "123456");
    }

    #[test]
    fn parse_sdp_ice_credentials() {
        let parsed = ParsedSdp::parse(&sample_sdp()).unwrap();
        assert_eq!(parsed.ice_ufrag.as_deref(), Some("abcd"));
        assert_eq!(parsed.ice_pwd.as_deref(), Some("efghijklmnop"));
        assert_eq!(parsed.fingerprint.as_deref(), Some("sha-256 AA:BB:CC"));
    }

    #[test]
    fn parse_sdp_media_sections() {
        let parsed = ParsedSdp::parse(&sample_sdp()).unwrap();
        assert_eq!(parsed.media_sections.len(), 2);

        let audio = &parsed.media_sections[0];
        assert_eq!(audio.kind, MediaKind::Audio);
        assert_eq!(audio.mid.as_deref(), Some("0"));
        assert_eq!(audio.direction, MediaDirection::SendRecv);
        assert_eq!(audio.codecs.len(), 2);
        assert_eq!(audio.codecs[0].name, "opus");
        assert_eq!(audio.codecs[0].clock_rate, 48000);
        assert_eq!(audio.codecs[0].channels, Some(2));
        assert_eq!(audio.codecs[1].name, "PCMU");

        let video = &parsed.media_sections[1];
        assert_eq!(video.kind, MediaKind::Video);
        assert_eq!(video.mid.as_deref(), Some("1"));
        assert_eq!(video.direction, MediaDirection::RecvOnly);
        assert_eq!(video.codecs[0].name, "VP8");
    }

    #[test]
    fn parse_invalid_version() {
        let sdp = "v=xyz\r\n";
        assert_eq!(ParsedSdp::parse(sdp), Err(SdpParseError::InvalidVersion));
    }

    #[test]
    fn parse_invalid_media_line() {
        let sdp = "v=0\r\nm=audio 9\r\n";
        assert_eq!(ParsedSdp::parse(sdp), Err(SdpParseError::InvalidMediaLine));
    }

    // ── ICE candidate ──────────────────────────────────────────────

    #[test]
    fn ice_candidate_host() {
        let c = IceCandidate::host("192.168.1.1", 54321, IceProtocol::Udp);
        assert_eq!(c.candidate_type, IceCandidateType::Host);
        assert_eq!(c.address, "192.168.1.1");
        assert_eq!(c.port, 54321);
        assert_eq!(c.component, 1);
    }

    #[test]
    fn ice_priority_ordering() {
        let host = IceCandidate::compute_priority(IceCandidateType::Host, 1, 1);
        let srflx = IceCandidate::compute_priority(IceCandidateType::Srflx, 1, 1);
        let relay = IceCandidate::compute_priority(IceCandidateType::Relay, 1, 1);
        assert!(host > srflx);
        assert!(srflx > relay);
    }

    #[test]
    fn ice_candidate_sdp_attribute() {
        let c = IceCandidate::host("10.0.0.1", 8080, IceProtocol::Udp);
        let attr = c.to_sdp_attribute();
        assert!(attr.starts_with("candidate:1 1 udp"));
        assert!(attr.contains("10.0.0.1"));
        assert!(attr.contains("8080"));
        assert!(attr.ends_with("typ host"));
    }

    // ── STUN message ───────────────────────────────────────────────

    #[test]
    fn stun_message_type_roundtrip() {
        for mt in [StunMessageType::BindingRequest, StunMessageType::BindingResponse, StunMessageType::BindingErrorResponse] {
            assert_eq!(StunMessageType::from_u16(mt.to_u16()), Some(mt));
        }
    }

    #[test]
    fn stun_message_type_unknown() {
        assert_eq!(StunMessageType::from_u16(0xFFFF), None);
    }

    #[test]
    fn stun_message_roundtrip() {
        let tid = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut msg = StunMessage::new_binding_request(tid);
        msg.add_attribute(StunAttrType::Username.to_u16(), b"user:pass".to_vec());
        msg.add_attribute(StunAttrType::Priority.to_u16(), 100u32.to_be_bytes().to_vec());

        let bytes = msg.to_bytes();
        assert!(bytes.len() >= 20);

        let parsed = StunMessage::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.msg_type, StunMessageType::BindingRequest);
        assert_eq!(parsed.transaction_id, tid);
        assert_eq!(parsed.attributes.len(), 2);
        assert_eq!(parsed.attributes[0].value, b"user:pass");
    }

    #[test]
    fn stun_message_too_short() {
        assert!(StunMessage::from_bytes(&[0; 10]).is_none());
    }

    #[test]
    fn stun_bad_magic_cookie() {
        let mut bytes = StunMessage::new_binding_request([0; 12]).to_bytes();
        bytes[4] = 0xFF; // corrupt cookie
        assert!(StunMessage::from_bytes(&bytes).is_none());
    }

    #[test]
    fn stun_attribute_padding() {
        let tid = [0u8; 12];
        let mut msg = StunMessage::new_binding_request(tid);
        // Value with length not divisible by 4
        msg.add_attribute(0x0006, vec![1, 2, 3]);
        let bytes = msg.to_bytes();
        let parsed = StunMessage::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.attributes[0].value, vec![1, 2, 3]);
    }

    // ── Codec negotiation ──────────────────────────────────────────

    #[test]
    fn negotiate_common_codecs() {
        let local = vec![
            Codec::new(111, "opus", 48000).with_channels(2),
            Codec::new(96, "VP8", 90000),
            Codec::new(97, "H264", 90000),
        ];
        let remote = vec![
            Codec::new(100, "VP8", 90000),
            Codec::new(102, "opus", 48000).with_channels(2),
        ];
        let result = negotiate_codecs(&local, &remote);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].name, "opus");
        assert_eq!(result[0].payload_type, 102); // remote PT
        assert_eq!(result[1].name, "VP8");
        assert_eq!(result[1].payload_type, 100);
    }

    #[test]
    fn negotiate_no_common() {
        let local = vec![Codec::new(111, "opus", 48000)];
        let remote = vec![Codec::new(96, "VP8", 90000)];
        assert!(negotiate_codecs(&local, &remote).is_empty());
    }

    #[test]
    fn negotiate_case_insensitive() {
        let local = vec![Codec::new(0, "pcmu", 8000)];
        let remote = vec![Codec::new(0, "PCMU", 8000)];
        let result = negotiate_codecs(&local, &remote);
        assert_eq!(result.len(), 1);
    }

    // ── Signaling state machine ────────────────────────────────────

    #[test]
    fn signaling_offer_answer_flow() {
        let mut sm = SignalingMachine::new();
        assert_eq!(sm.state, SignalingState::Stable);

        // Local creates offer
        sm.set_local_description(SessionDescription::new(SdpType::Offer, "offer-sdp"));
        assert_eq!(sm.state, SignalingState::HaveLocalOffer);
        let events = sm.take_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SignalEvent::SendOffer(_)));

        // Remote sends answer
        sm.set_remote_description(SessionDescription::new(SdpType::Answer, "answer-sdp"));
        assert_eq!(sm.state, SignalingState::Stable);
        let events = sm.take_events();
        assert!(events.iter().any(|e| matches!(e, SignalEvent::NegotiationComplete)));
    }

    #[test]
    fn signaling_remote_offer_local_answer() {
        let mut sm = SignalingMachine::new();

        sm.set_remote_description(SessionDescription::new(SdpType::Offer, "remote-offer"));
        assert_eq!(sm.state, SignalingState::HaveRemoteOffer);

        sm.set_local_description(SessionDescription::new(SdpType::Answer, "local-answer"));
        assert_eq!(sm.state, SignalingState::Stable);
        let events = sm.take_events();
        assert!(events.iter().any(|e| matches!(e, SignalEvent::NegotiationComplete)));
        assert!(events.iter().any(|e| matches!(e, SignalEvent::SendAnswer(_))));
    }

    #[test]
    fn signaling_pranswer() {
        let mut sm = SignalingMachine::new();
        sm.set_remote_description(SessionDescription::new(SdpType::Offer, "offer"));
        assert_eq!(sm.state, SignalingState::HaveRemoteOffer);

        sm.set_local_description(SessionDescription::new(SdpType::Pranswer, "pranswer"));
        assert_eq!(sm.state, SignalingState::HaveLocalPranswer);
    }

    #[test]
    fn signaling_rollback() {
        let mut sm = SignalingMachine::new();
        sm.set_local_description(SessionDescription::new(SdpType::Offer, "offer"));
        assert_eq!(sm.state, SignalingState::HaveLocalOffer);

        sm.set_local_description(SessionDescription::new(SdpType::Rollback, ""));
        assert_eq!(sm.state, SignalingState::Stable);
        assert!(sm.local_description.is_none());
    }

    #[test]
    fn signaling_invalid_transition() {
        let mut sm = SignalingMachine::new();
        // Can't set answer from stable
        sm.set_local_description(SessionDescription::new(SdpType::Answer, "answer"));
        let events = sm.take_events();
        assert!(events.iter().any(|e| matches!(e, SignalEvent::Error(_))));
    }

    #[test]
    fn signaling_ice_candidates() {
        let mut sm = SignalingMachine::new();
        let c1 = IceCandidate::host("10.0.0.1", 5000, IceProtocol::Udp);
        let c2 = IceCandidate::host("10.0.0.2", 5001, IceProtocol::Tcp);

        sm.add_local_candidate(c1);
        sm.add_remote_candidate(c2);
        assert_eq!(sm.local_candidates.len(), 1);
        assert_eq!(sm.remote_candidates.len(), 1);

        let events = sm.take_events();
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], SignalEvent::AddIceCandidate(_)));
    }

    #[test]
    fn signaling_close() {
        let mut sm = SignalingMachine::new();
        sm.close();
        assert_eq!(sm.state, SignalingState::Closed);
    }

    #[test]
    fn signaling_default() {
        let sm = SignalingMachine::default();
        assert_eq!(sm.state, SignalingState::Stable);
    }

    #[test]
    fn sdp_parse_error_display() {
        assert_eq!(SdpParseError::InvalidVersion.to_string(), "invalid SDP version");
        assert_eq!(SdpParseError::InvalidMediaLine.to_string(), "invalid SDP media line");
    }

    #[test]
    fn parse_empty_sdp() {
        let parsed = ParsedSdp::parse("").unwrap();
        assert_eq!(parsed.version, 0);
        assert!(parsed.media_sections.is_empty());
    }

    #[test]
    fn stun_attr_type_values() {
        assert_eq!(StunAttrType::MappedAddress.to_u16(), 0x0001);
        assert_eq!(StunAttrType::XorMappedAddress.to_u16(), 0x0020);
        assert_eq!(StunAttrType::Fingerprint.to_u16(), 0x8028);
    }
}
