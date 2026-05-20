//! WebRTC state machine: peer connections, ICE, signaling, data channels.

use std::collections::VecDeque;

// ── ICE ─────────────────────────────────────────────────────────

/// ICE connection state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IceConnectionState {
    New,
    Checking,
    Connected,
    Completed,
    Failed,
    Disconnected,
    Closed,
}

/// An ICE candidate.
#[derive(Debug, Clone, PartialEq)]
pub struct IceCandidate {
    pub candidate: String,
    pub sdp_mid: Option<String>,
    pub sdp_m_line_index: Option<u32>,
}

// ── Signaling ───────────────────────────────────────────────────

/// WebRTC signaling state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalingState {
    Stable,
    HaveLocalOffer,
    HaveRemoteOffer,
    HaveLocalPrAnswer,
    HaveRemotePrAnswer,
    Closed,
}

/// SDP type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SdpType {
    Offer,
    Answer,
    Pranswer,
    Rollback,
}

/// Session description (SDP).
#[derive(Debug, Clone, PartialEq)]
pub struct SessionDescription {
    pub sdp_type: SdpType,
    pub sdp: String,
}

// ── Data Channel ────────────────────────────────────────────────

/// Data channel state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataChannelState {
    Connecting,
    Open,
    Closing,
    Closed,
}

/// A WebRTC data channel.
#[derive(Debug, Clone)]
pub struct DataChannel {
    pub id: u16,
    pub label: String,
    pub ordered: bool,
    pub max_retransmits: Option<u16>,
    pub protocol: String,
    pub state: DataChannelState,
    pub buffered: VecDeque<Vec<u8>>,
}

// ── Peer Connection ─────────────────────────────────────────────

/// A WebRTC peer connection state machine.
#[derive(Debug)]
pub struct PeerConnection {
    pub ice_state: IceConnectionState,
    pub signaling_state: SignalingState,
    pub local_description: Option<SessionDescription>,
    pub remote_description: Option<SessionDescription>,
    pub ice_candidates: Vec<IceCandidate>,
    pub remote_candidates: Vec<IceCandidate>,
    pub data_channels: Vec<DataChannel>,
    next_channel_id: u16,
}

impl PeerConnection {
    pub fn new() -> Self {
        Self {
            ice_state: IceConnectionState::New,
            signaling_state: SignalingState::Stable,
            local_description: None,
            remote_description: None,
            ice_candidates: Vec::new(),
            remote_candidates: Vec::new(),
            data_channels: Vec::new(),
            next_channel_id: 0,
        }
    }

    /// Create an SDP offer. Connection must be in `Stable` signaling state.
    pub fn create_offer(&self) -> SessionDescription {
        SessionDescription {
            sdp_type: SdpType::Offer,
            sdp: format!("v=0\r\no=- {} 0 IN IP4 0.0.0.0\r\n", self.next_channel_id),
        }
    }

    /// Create an SDP answer. Connection must have a remote offer.
    pub fn create_answer(&self) -> SessionDescription {
        SessionDescription {
            sdp_type: SdpType::Answer,
            sdp: format!("v=0\r\no=- {} 0 IN IP4 0.0.0.0\r\n", self.next_channel_id),
        }
    }

    /// Set local description and update signaling state per WebRTC spec.
    pub fn set_local_description(&mut self, desc: SessionDescription) {
        match (&self.signaling_state, desc.sdp_type) {
            (SignalingState::Stable, SdpType::Offer) => {
                self.signaling_state = SignalingState::HaveLocalOffer;
            }
            (SignalingState::HaveRemoteOffer, SdpType::Answer) => {
                self.signaling_state = SignalingState::Stable;
            }
            (SignalingState::HaveRemoteOffer, SdpType::Pranswer) => {
                self.signaling_state = SignalingState::HaveLocalPrAnswer;
            }
            (SignalingState::Stable, SdpType::Rollback) => {
                // No-op: already stable.
            }
            (SignalingState::HaveLocalOffer, SdpType::Rollback) => {
                self.signaling_state = SignalingState::Stable;
            }
            _ => {}
        }
        self.local_description = Some(desc);
    }

    /// Set remote description and update signaling state per WebRTC spec.
    pub fn set_remote_description(&mut self, desc: SessionDescription) {
        match (&self.signaling_state, desc.sdp_type) {
            (SignalingState::Stable, SdpType::Offer) => {
                self.signaling_state = SignalingState::HaveRemoteOffer;
            }
            (SignalingState::HaveLocalOffer, SdpType::Answer) => {
                self.signaling_state = SignalingState::Stable;
            }
            (SignalingState::HaveLocalOffer, SdpType::Pranswer) => {
                self.signaling_state = SignalingState::HaveRemotePrAnswer;
            }
            (SignalingState::Stable, SdpType::Rollback) => {
                // No-op.
            }
            (SignalingState::HaveRemoteOffer, SdpType::Rollback) => {
                self.signaling_state = SignalingState::Stable;
            }
            _ => {}
        }
        self.remote_description = Some(desc);
    }

    /// Add a local ICE candidate.
    pub fn add_ice_candidate(&mut self, candidate: IceCandidate) {
        if self.ice_state == IceConnectionState::New {
            self.ice_state = IceConnectionState::Checking;
        }
        self.ice_candidates.push(candidate);
    }

    /// Create a data channel. Returns a reference to the newly created channel.
    pub fn create_data_channel(&mut self, label: impl Into<String>, ordered: bool) -> &DataChannel {
        let id = self.next_channel_id;
        self.next_channel_id += 1;
        let channel = DataChannel {
            id,
            label: label.into(),
            ordered,
            max_retransmits: None,
            protocol: String::new(),
            state: DataChannelState::Connecting,
            buffered: VecDeque::new(),
        };
        self.data_channels.push(channel);
        self.data_channels.last().unwrap()
    }

    /// Close the connection.
    pub fn close(&mut self) {
        self.signaling_state = SignalingState::Closed;
        self.ice_state = IceConnectionState::Closed;
        for ch in &mut self.data_channels {
            ch.state = DataChannelState::Closed;
        }
    }
}

impl Default for PeerConnection {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state() {
        let pc = PeerConnection::new();
        assert_eq!(pc.ice_state, IceConnectionState::New);
        assert_eq!(pc.signaling_state, SignalingState::Stable);
        assert!(pc.local_description.is_none());
        assert!(pc.remote_description.is_none());
    }

    #[test]
    fn offer_answer_exchange() {
        let mut offerer = PeerConnection::new();
        let mut answerer = PeerConnection::new();

        let offer = offerer.create_offer();
        assert_eq!(offer.sdp_type, SdpType::Offer);
        offerer.set_local_description(offer.clone());
        assert_eq!(offerer.signaling_state, SignalingState::HaveLocalOffer);

        answerer.set_remote_description(offer);
        assert_eq!(answerer.signaling_state, SignalingState::HaveRemoteOffer);

        let answer = answerer.create_answer();
        assert_eq!(answer.sdp_type, SdpType::Answer);
        answerer.set_local_description(answer.clone());
        assert_eq!(answerer.signaling_state, SignalingState::Stable);

        offerer.set_remote_description(answer);
        assert_eq!(offerer.signaling_state, SignalingState::Stable);
    }

    #[test]
    fn signaling_state_pranswer() {
        let mut pc = PeerConnection::new();
        let offer = SessionDescription { sdp_type: SdpType::Offer, sdp: "v=0\r\n".into() };
        pc.set_remote_description(offer);
        assert_eq!(pc.signaling_state, SignalingState::HaveRemoteOffer);

        let pranswer = SessionDescription { sdp_type: SdpType::Pranswer, sdp: "v=0\r\n".into() };
        pc.set_local_description(pranswer);
        assert_eq!(pc.signaling_state, SignalingState::HaveLocalPrAnswer);
    }

    #[test]
    fn ice_candidate_addition() {
        let mut pc = PeerConnection::new();
        assert_eq!(pc.ice_state, IceConnectionState::New);

        pc.add_ice_candidate(IceCandidate {
            candidate: "candidate:1 1 UDP 2122252543 192.168.1.1 12345 typ host".into(),
            sdp_mid: Some("0".into()),
            sdp_m_line_index: Some(0),
        });

        assert_eq!(pc.ice_state, IceConnectionState::Checking);
        assert_eq!(pc.ice_candidates.len(), 1);
    }

    #[test]
    fn data_channel_creation() {
        let mut pc = PeerConnection::new();
        let ch = pc.create_data_channel("chat", true);
        assert_eq!(ch.label, "chat");
        assert!(ch.ordered);
        assert_eq!(ch.state, DataChannelState::Connecting);
        assert_eq!(ch.id, 0);

        let ch2 = pc.create_data_channel("video", false);
        assert_eq!(ch2.id, 1);
        assert!(!ch2.ordered);
        assert_eq!(pc.data_channels.len(), 2);
    }

    #[test]
    fn close_connection() {
        let mut pc = PeerConnection::new();
        pc.create_data_channel("ch", true);
        pc.close();
        assert_eq!(pc.signaling_state, SignalingState::Closed);
        assert_eq!(pc.ice_state, IceConnectionState::Closed);
        assert_eq!(pc.data_channels[0].state, DataChannelState::Closed);
    }

    #[test]
    fn rollback_from_local_offer() {
        let mut pc = PeerConnection::new();
        let offer = pc.create_offer();
        pc.set_local_description(offer);
        assert_eq!(pc.signaling_state, SignalingState::HaveLocalOffer);

        let rollback = SessionDescription { sdp_type: SdpType::Rollback, sdp: String::new() };
        pc.set_local_description(rollback);
        assert_eq!(pc.signaling_state, SignalingState::Stable);
    }

    #[test]
    fn multiple_ice_candidates() {
        let mut pc = PeerConnection::new();
        for i in 0..5 {
            pc.add_ice_candidate(IceCandidate {
                candidate: format!("candidate:{i}"),
                sdp_mid: None,
                sdp_m_line_index: None,
            });
        }
        assert_eq!(pc.ice_candidates.len(), 5);
        assert_eq!(pc.ice_state, IceConnectionState::Checking);
    }

    #[test]
    fn data_channel_buffered() {
        let mut pc = PeerConnection::new();
        pc.create_data_channel("buf", true);
        pc.data_channels[0].buffered.push_back(vec![1, 2, 3]);
        assert_eq!(pc.data_channels[0].buffered.len(), 1);
    }

    #[test]
    fn remote_pranswer_state() {
        let mut pc = PeerConnection::new();
        let offer = pc.create_offer();
        pc.set_local_description(offer);
        assert_eq!(pc.signaling_state, SignalingState::HaveLocalOffer);

        let pranswer = SessionDescription { sdp_type: SdpType::Pranswer, sdp: "v=0\r\n".into() };
        pc.set_remote_description(pranswer);
        assert_eq!(pc.signaling_state, SignalingState::HaveRemotePrAnswer);
    }

    #[test]
    fn descriptions_stored() {
        let mut pc = PeerConnection::new();
        let offer = pc.create_offer();
        pc.set_local_description(offer.clone());
        assert_eq!(pc.local_description.as_ref().unwrap().sdp_type, SdpType::Offer);

        let answer = SessionDescription { sdp_type: SdpType::Answer, sdp: "v=0\r\n".into() };
        pc.set_remote_description(answer);
        assert_eq!(pc.remote_description.as_ref().unwrap().sdp_type, SdpType::Answer);
    }
}
