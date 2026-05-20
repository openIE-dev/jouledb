//! WebSocket frame codec.
//!
//! Replaces `tungstenite` / `ws` with a pure-Rust WebSocket frame parser.
//! Supports frame header parsing (FIN, RSV, opcode, mask, payload length),
//! masking/unmasking, text/binary/ping/pong/close frames, fragmentation,
//! and close status codes.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// WebSocket frame errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WsFrameError {
    /// Not enough data to parse frame header.
    Incomplete,
    /// Reserved bits are set when not negotiated.
    ReservedBitsSet,
    /// Invalid opcode.
    InvalidOpcode(u8),
    /// Control frame payload exceeds 125 bytes.
    ControlFrameTooLarge(usize),
    /// Control frame is fragmented.
    FragmentedControlFrame,
    /// Masking key required but not present.
    MaskRequired,
    /// Invalid close code.
    InvalidCloseCode(u16),
    /// Invalid UTF-8 in text frame.
    InvalidUtf8,
    /// Continuation frame without initial fragment.
    UnexpectedContinuation,
    /// New data frame while fragmented message in progress.
    InterruptedFragment,
}

impl fmt::Display for WsFrameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Incomplete => write!(f, "incomplete frame data"),
            Self::ReservedBitsSet => write!(f, "reserved bits set"),
            Self::InvalidOpcode(op) => write!(f, "invalid opcode: {op:#x}"),
            Self::ControlFrameTooLarge(n) => write!(f, "control frame payload {n} > 125"),
            Self::FragmentedControlFrame => write!(f, "fragmented control frame"),
            Self::MaskRequired => write!(f, "mask required"),
            Self::InvalidCloseCode(c) => write!(f, "invalid close code: {c}"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 in text frame"),
            Self::UnexpectedContinuation => write!(f, "unexpected continuation frame"),
            Self::InterruptedFragment => write!(f, "new data frame interrupts fragment"),
        }
    }
}

impl std::error::Error for WsFrameError {}

// ── Opcode ──────────────────────────────────────────────────

/// WebSocket frame opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    Continuation,
    Text,
    Binary,
    Close,
    Ping,
    Pong,
}

impl Opcode {
    pub fn from_u8(val: u8) -> Result<Self, WsFrameError> {
        match val {
            0x0 => Ok(Self::Continuation),
            0x1 => Ok(Self::Text),
            0x2 => Ok(Self::Binary),
            0x8 => Ok(Self::Close),
            0x9 => Ok(Self::Ping),
            0xA => Ok(Self::Pong),
            n => Err(WsFrameError::InvalidOpcode(n)),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Continuation => 0x0,
            Self::Text => 0x1,
            Self::Binary => 0x2,
            Self::Close => 0x8,
            Self::Ping => 0x9,
            Self::Pong => 0xA,
        }
    }

    /// Whether this is a control frame opcode.
    pub fn is_control(self) -> bool {
        matches!(self, Self::Close | Self::Ping | Self::Pong)
    }
}

// ── Close Code ──────────────────────────────────────────────

/// WebSocket close status codes (RFC 6455 Section 7.4.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CloseCode {
    Normal,
    GoingAway,
    ProtocolError,
    Unsupported,
    NoStatus,
    Abnormal,
    InvalidPayload,
    PolicyViolation,
    MessageTooBig,
    MandatoryExtension,
    InternalError,
    ServiceRestart,
    TryAgainLater,
    Other(u16),
}

impl CloseCode {
    pub fn from_u16(val: u16) -> Self {
        match val {
            1000 => Self::Normal,
            1001 => Self::GoingAway,
            1002 => Self::ProtocolError,
            1003 => Self::Unsupported,
            1005 => Self::NoStatus,
            1006 => Self::Abnormal,
            1007 => Self::InvalidPayload,
            1008 => Self::PolicyViolation,
            1009 => Self::MessageTooBig,
            1010 => Self::MandatoryExtension,
            1011 => Self::InternalError,
            1012 => Self::ServiceRestart,
            1013 => Self::TryAgainLater,
            n => Self::Other(n),
        }
    }

    pub fn to_u16(self) -> u16 {
        match self {
            Self::Normal => 1000,
            Self::GoingAway => 1001,
            Self::ProtocolError => 1002,
            Self::Unsupported => 1003,
            Self::NoStatus => 1005,
            Self::Abnormal => 1006,
            Self::InvalidPayload => 1007,
            Self::PolicyViolation => 1008,
            Self::MessageTooBig => 1009,
            Self::MandatoryExtension => 1010,
            Self::InternalError => 1011,
            Self::ServiceRestart => 1012,
            Self::TryAgainLater => 1013,
            Self::Other(n) => n,
        }
    }

    /// Whether this code may be sent in a close frame.
    pub fn is_sendable(self) -> bool {
        !matches!(self, Self::NoStatus | Self::Abnormal)
    }
}

// ── Frame ───────────────────────────────────────────────────

/// A parsed WebSocket frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WsFrame {
    pub fin: bool,
    pub rsv1: bool,
    pub rsv2: bool,
    pub rsv3: bool,
    pub opcode: Opcode,
    pub mask: Option<[u8; 4]>,
    pub payload: Vec<u8>,
}

impl WsFrame {
    /// Create a text frame.
    pub fn text(data: &str) -> Self {
        Self {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Text,
            mask: None,
            payload: data.as_bytes().to_vec(),
        }
    }

    /// Create a binary frame.
    pub fn binary(data: &[u8]) -> Self {
        Self {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Binary,
            mask: None,
            payload: data.to_vec(),
        }
    }

    /// Create a ping frame.
    pub fn ping(data: &[u8]) -> Self {
        Self {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Ping,
            mask: None,
            payload: data.to_vec(),
        }
    }

    /// Create a pong frame.
    pub fn pong(data: &[u8]) -> Self {
        Self {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Pong,
            mask: None,
            payload: data.to_vec(),
        }
    }

    /// Create a close frame with optional code and reason.
    pub fn close(code: Option<CloseCode>, reason: &str) -> Self {
        let mut payload = Vec::new();
        if let Some(c) = code {
            payload.extend_from_slice(&c.to_u16().to_be_bytes());
            payload.extend_from_slice(reason.as_bytes());
        }
        Self {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Close,
            mask: None,
            payload,
        }
    }

    /// Parse the close code and reason from a close frame payload.
    pub fn close_info(&self) -> Option<(CloseCode, String)> {
        if self.opcode != Opcode::Close || self.payload.len() < 2 {
            return None;
        }
        let code = u16::from_be_bytes([self.payload[0], self.payload[1]]);
        let reason = String::from_utf8_lossy(&self.payload[2..]).to_string();
        Some((CloseCode::from_u16(code), reason))
    }

    /// Apply a mask to this frame.
    pub fn with_mask(mut self, mask_key: [u8; 4]) -> Self {
        self.mask = Some(mask_key);
        self
    }

    /// Create a continuation frame.
    pub fn continuation(fin: bool, data: &[u8]) -> Self {
        Self {
            fin,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Continuation,
            mask: None,
            payload: data.to_vec(),
        }
    }

    /// Validate frame constraints.
    pub fn validate(&self) -> Result<(), WsFrameError> {
        if self.opcode.is_control() {
            if self.payload.len() > 125 {
                return Err(WsFrameError::ControlFrameTooLarge(self.payload.len()));
            }
            if !self.fin {
                return Err(WsFrameError::FragmentedControlFrame);
            }
        }
        Ok(())
    }

    /// Encode the frame to bytes.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // Byte 0: FIN, RSV, opcode
        let mut b0: u8 = self.opcode.to_u8();
        if self.fin {
            b0 |= 0x80;
        }
        if self.rsv1 {
            b0 |= 0x40;
        }
        if self.rsv2 {
            b0 |= 0x20;
        }
        if self.rsv3 {
            b0 |= 0x10;
        }
        buf.push(b0);

        // Byte 1: MASK bit + payload length
        let masked = self.mask.is_some();
        let len = self.payload.len();
        let mask_bit: u8 = if masked { 0x80 } else { 0x00 };

        if len < 126 {
            buf.push(mask_bit | (len as u8));
        } else if len <= 0xFFFF {
            buf.push(mask_bit | 126);
            buf.extend_from_slice(&(len as u16).to_be_bytes());
        } else {
            buf.push(mask_bit | 127);
            buf.extend_from_slice(&(len as u64).to_be_bytes());
        }

        // Masking key
        if let Some(key) = self.mask {
            buf.extend_from_slice(&key);
            // Apply mask to payload
            let mut masked_payload = self.payload.clone();
            apply_mask(&mut masked_payload, key);
            buf.extend_from_slice(&masked_payload);
        } else {
            buf.extend_from_slice(&self.payload);
        }

        buf
    }

    /// Parse a frame from bytes. Returns the frame and number of bytes consumed.
    pub fn parse(data: &[u8]) -> Result<(Self, usize), WsFrameError> {
        if data.len() < 2 {
            return Err(WsFrameError::Incomplete);
        }

        let b0 = data[0];
        let b1 = data[1];

        let fin = (b0 & 0x80) != 0;
        let rsv1 = (b0 & 0x40) != 0;
        let rsv2 = (b0 & 0x20) != 0;
        let rsv3 = (b0 & 0x10) != 0;
        let opcode = Opcode::from_u8(b0 & 0x0F)?;

        let masked = (b1 & 0x80) != 0;
        let len_field = b1 & 0x7F;

        let mut offset = 2;
        let payload_len: usize;

        if len_field < 126 {
            payload_len = len_field as usize;
        } else if len_field == 126 {
            if data.len() < offset + 2 {
                return Err(WsFrameError::Incomplete);
            }
            payload_len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
        } else {
            if data.len() < offset + 8 {
                return Err(WsFrameError::Incomplete);
            }
            payload_len = u64::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]) as usize;
            offset += 8;
        }

        let mask_key = if masked {
            if data.len() < offset + 4 {
                return Err(WsFrameError::Incomplete);
            }
            let key = [data[offset], data[offset + 1], data[offset + 2], data[offset + 3]];
            offset += 4;
            Some(key)
        } else {
            None
        };

        if data.len() < offset + payload_len {
            return Err(WsFrameError::Incomplete);
        }

        let mut payload = data[offset..offset + payload_len].to_vec();
        if let Some(key) = mask_key {
            apply_mask(&mut payload, key);
        }

        let frame = Self {
            fin,
            rsv1,
            rsv2,
            rsv3,
            opcode,
            mask: mask_key,
            payload,
        };
        Ok((frame, offset + payload_len))
    }
}

// ── Masking ─────────────────────────────────────────────────

/// Apply or remove XOR mask in-place.
pub fn apply_mask(data: &mut [u8], key: [u8; 4]) {
    for (i, byte) in data.iter_mut().enumerate() {
        *byte ^= key[i % 4];
    }
}

// ── Fragment Assembler ──────────────────────────────────────

/// Reassembles fragmented WebSocket messages.
pub struct FragmentAssembler {
    opcode: Option<Opcode>,
    fragments: Vec<u8>,
}

impl FragmentAssembler {
    pub fn new() -> Self {
        Self { opcode: None, fragments: Vec::new() }
    }

    /// Feed a frame. Returns Some((opcode, payload)) when a complete message
    /// is assembled, or None if more fragments are needed.
    pub fn feed(&mut self, frame: &WsFrame) -> Result<Option<(Opcode, Vec<u8>)>, WsFrameError> {
        // Control frames can be interleaved and are always complete.
        if frame.opcode.is_control() {
            return Ok(Some((frame.opcode, frame.payload.clone())));
        }

        match frame.opcode {
            Opcode::Continuation => {
                if self.opcode.is_none() {
                    return Err(WsFrameError::UnexpectedContinuation);
                }
                self.fragments.extend_from_slice(&frame.payload);
                if frame.fin {
                    let opcode = self.opcode.take().unwrap();
                    let payload = std::mem::take(&mut self.fragments);
                    Ok(Some((opcode, payload)))
                } else {
                    Ok(None)
                }
            }
            Opcode::Text | Opcode::Binary => {
                if self.opcode.is_some() {
                    return Err(WsFrameError::InterruptedFragment);
                }
                if frame.fin {
                    Ok(Some((frame.opcode, frame.payload.clone())))
                } else {
                    self.opcode = Some(frame.opcode);
                    self.fragments = frame.payload.clone();
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Whether a fragmented message is in progress.
    pub fn is_assembling(&self) -> bool {
        self.opcode.is_some()
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_frame_roundtrip() {
        let frame = WsFrame::text("hello");
        let bytes = frame.to_bytes();
        let (parsed, consumed) = WsFrame::parse(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        assert!(parsed.fin);
        assert_eq!(parsed.opcode, Opcode::Text);
        assert_eq!(parsed.payload, b"hello");
    }

    #[test]
    fn binary_frame_roundtrip() {
        let data = vec![0x01, 0x02, 0x03, 0xFF];
        let frame = WsFrame::binary(&data);
        let bytes = frame.to_bytes();
        let (parsed, _) = WsFrame::parse(&bytes).unwrap();
        assert_eq!(parsed.opcode, Opcode::Binary);
        assert_eq!(parsed.payload, data);
    }

    #[test]
    fn masked_frame_roundtrip() {
        let frame = WsFrame::text("masked data").with_mask([0xAB, 0xCD, 0xEF, 0x01]);
        let bytes = frame.to_bytes();
        let (parsed, _) = WsFrame::parse(&bytes).unwrap();
        assert_eq!(parsed.payload, b"masked data");
        assert!(parsed.mask.is_some());
    }

    #[test]
    fn ping_pong_frames() {
        let ping = WsFrame::ping(b"alive");
        let bytes = ping.to_bytes();
        let (parsed, _) = WsFrame::parse(&bytes).unwrap();
        assert_eq!(parsed.opcode, Opcode::Ping);
        assert_eq!(parsed.payload, b"alive");

        let pong = WsFrame::pong(b"alive");
        let bytes = pong.to_bytes();
        let (parsed, _) = WsFrame::parse(&bytes).unwrap();
        assert_eq!(parsed.opcode, Opcode::Pong);
    }

    #[test]
    fn close_frame_with_code() {
        let frame = WsFrame::close(Some(CloseCode::Normal), "goodbye");
        let bytes = frame.to_bytes();
        let (parsed, _) = WsFrame::parse(&bytes).unwrap();
        let (code, reason) = parsed.close_info().unwrap();
        assert_eq!(code, CloseCode::Normal);
        assert_eq!(reason, "goodbye");
    }

    #[test]
    fn extended_payload_length_16bit() {
        let data = vec![0x42; 300];
        let frame = WsFrame::binary(&data);
        let bytes = frame.to_bytes();
        assert_eq!(bytes[1] & 0x7F, 126); // 16-bit length indicator
        let (parsed, _) = WsFrame::parse(&bytes).unwrap();
        assert_eq!(parsed.payload.len(), 300);
    }

    #[test]
    fn masking_applies_xor() {
        let mut data = b"test".to_vec();
        let key = [0x37, 0xFA, 0x21, 0x3D];
        apply_mask(&mut data, key);
        // Verify data changed
        assert_ne!(data, b"test");
        // Unmask
        apply_mask(&mut data, key);
        assert_eq!(data, b"test");
    }

    #[test]
    fn fragment_assembler_single_frame() {
        let mut assembler = FragmentAssembler::new();
        let frame = WsFrame::text("complete");
        let result = assembler.feed(&frame).unwrap();
        assert!(result.is_some());
        let (op, data) = result.unwrap();
        assert_eq!(op, Opcode::Text);
        assert_eq!(data, b"complete");
    }

    #[test]
    fn fragment_assembler_multi_frame() {
        let mut assembler = FragmentAssembler::new();

        // First fragment
        let f1 = WsFrame {
            fin: false,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Text,
            mask: None,
            payload: b"hel".to_vec(),
        };
        assert!(assembler.feed(&f1).unwrap().is_none());
        assert!(assembler.is_assembling());

        // Control frame interleaved
        let ping = WsFrame::ping(b"");
        let result = assembler.feed(&ping).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().0, Opcode::Ping);

        // Final fragment
        let f2 = WsFrame::continuation(true, b"lo");
        let result = assembler.feed(&f2).unwrap();
        assert!(result.is_some());
        let (op, data) = result.unwrap();
        assert_eq!(op, Opcode::Text);
        assert_eq!(data, b"hello");
        assert!(!assembler.is_assembling());
    }

    #[test]
    fn control_frame_too_large() {
        let frame = WsFrame {
            fin: true,
            rsv1: false,
            rsv2: false,
            rsv3: false,
            opcode: Opcode::Ping,
            mask: None,
            payload: vec![0; 126],
        };
        assert_eq!(frame.validate().unwrap_err(), WsFrameError::ControlFrameTooLarge(126));
    }

    #[test]
    fn close_codes() {
        assert_eq!(CloseCode::from_u16(1000), CloseCode::Normal);
        assert_eq!(CloseCode::from_u16(1001), CloseCode::GoingAway);
        assert_eq!(CloseCode::from_u16(9999), CloseCode::Other(9999));
        assert!(CloseCode::Normal.is_sendable());
        assert!(!CloseCode::NoStatus.is_sendable());
        assert!(!CloseCode::Abnormal.is_sendable());
    }

    #[test]
    fn unexpected_continuation_error() {
        let mut assembler = FragmentAssembler::new();
        let frame = WsFrame::continuation(true, b"data");
        assert_eq!(
            assembler.feed(&frame).unwrap_err(),
            WsFrameError::UnexpectedContinuation
        );
    }

    #[test]
    fn incomplete_data() {
        assert_eq!(WsFrame::parse(&[0x81]).unwrap_err(), WsFrameError::Incomplete);
    }
}
