//! MQTT protocol codec.
//!
//! Replaces `rumqttc` / `paho-mqtt` with a pure-Rust MQTT packet model.
//! Supports packet types (CONNECT, CONNACK, PUBLISH, PUBACK, SUBSCRIBE,
//! SUBACK, UNSUBSCRIBE, PINGREQ, PINGRESP, DISCONNECT), QoS levels (0/1/2),
//! topic filters with wildcards (+/#), and remaining length encoding.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// MQTT protocol errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MqttError {
    /// Packet buffer too short.
    PacketTooShort,
    /// Invalid packet type nibble.
    InvalidPacketType(u8),
    /// Remaining length encoding error.
    MalformedRemainingLength,
    /// Remaining length exceeds 268 MB limit.
    RemainingLengthOverflow,
    /// Invalid protocol name.
    InvalidProtocolName(String),
    /// Unsupported protocol level.
    UnsupportedProtocolLevel(u8),
    /// Invalid QoS value.
    InvalidQos(u8),
    /// Invalid connect flags.
    InvalidConnectFlags(String),
    /// Topic filter validation error.
    InvalidTopicFilter(String),
    /// Topic name contains wildcards.
    InvalidTopicName(String),
    /// Unexpected end of data.
    UnexpectedEof,
    /// Invalid return code.
    InvalidReturnCode(u8),
}

impl fmt::Display for MqttError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PacketTooShort => write!(f, "packet too short"),
            Self::InvalidPacketType(t) => write!(f, "invalid packet type: {t}"),
            Self::MalformedRemainingLength => write!(f, "malformed remaining length"),
            Self::RemainingLengthOverflow => write!(f, "remaining length overflow"),
            Self::InvalidProtocolName(n) => write!(f, "invalid protocol name: {n}"),
            Self::UnsupportedProtocolLevel(l) => write!(f, "unsupported protocol level: {l}"),
            Self::InvalidQos(q) => write!(f, "invalid QoS: {q}"),
            Self::InvalidConnectFlags(msg) => write!(f, "invalid connect flags: {msg}"),
            Self::InvalidTopicFilter(msg) => write!(f, "invalid topic filter: {msg}"),
            Self::InvalidTopicName(msg) => write!(f, "invalid topic name: {msg}"),
            Self::UnexpectedEof => write!(f, "unexpected end of data"),
            Self::InvalidReturnCode(c) => write!(f, "invalid return code: {c}"),
        }
    }
}

impl std::error::Error for MqttError {}

// ── QoS ─────────────────────────────────────────────────────

/// MQTT Quality of Service level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QoS {
    /// At most once (fire and forget).
    AtMostOnce,
    /// At least once (acknowledged delivery).
    AtLeastOnce,
    /// Exactly once (assured delivery).
    ExactlyOnce,
}

impl QoS {
    pub fn from_u8(val: u8) -> Result<Self, MqttError> {
        match val {
            0 => Ok(Self::AtMostOnce),
            1 => Ok(Self::AtLeastOnce),
            2 => Ok(Self::ExactlyOnce),
            n => Err(MqttError::InvalidQos(n)),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::AtMostOnce => 0,
            Self::AtLeastOnce => 1,
            Self::ExactlyOnce => 2,
        }
    }
}

// ── Remaining Length Encoding ────────────────────────────────

/// Encode a remaining length value to MQTT variable-length encoding.
pub fn encode_remaining_length(mut length: u32) -> Result<Vec<u8>, MqttError> {
    if length > 268_435_455 {
        return Err(MqttError::RemainingLengthOverflow);
    }
    let mut buf = Vec::with_capacity(4);
    loop {
        let mut byte = (length % 128) as u8;
        length /= 128;
        if length > 0 {
            byte |= 0x80;
        }
        buf.push(byte);
        if length == 0 {
            break;
        }
    }
    Ok(buf)
}

/// Decode remaining length from bytes, returning (value, bytes_consumed).
pub fn decode_remaining_length(data: &[u8]) -> Result<(u32, usize), MqttError> {
    let mut multiplier: u32 = 1;
    let mut value: u32 = 0;
    for (i, &byte) in data.iter().enumerate() {
        value += (byte as u32 & 0x7F) * multiplier;
        if multiplier > 128 * 128 * 128 {
            return Err(MqttError::MalformedRemainingLength);
        }
        if (byte & 0x80) == 0 {
            return Ok((value, i + 1));
        }
        multiplier *= 128;
    }
    Err(MqttError::MalformedRemainingLength)
}

// ── Connect Flags ───────────────────────────────────────────

/// Flags in the CONNECT packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConnectFlags {
    pub clean_session: bool,
    pub will: bool,
    pub will_qos: QoS,
    pub will_retain: bool,
    pub password: bool,
    pub username: bool,
}

impl ConnectFlags {
    pub fn from_byte(byte: u8) -> Result<Self, MqttError> {
        if byte & 0x01 != 0 {
            return Err(MqttError::InvalidConnectFlags("reserved bit set".into()));
        }
        let will_qos = QoS::from_u8((byte >> 3) & 0x03)?;
        Ok(Self {
            clean_session: (byte & 0x02) != 0,
            will: (byte & 0x04) != 0,
            will_qos,
            will_retain: (byte & 0x20) != 0,
            password: (byte & 0x40) != 0,
            username: (byte & 0x80) != 0,
        })
    }

    pub fn to_byte(&self) -> u8 {
        let mut b: u8 = 0;
        if self.clean_session {
            b |= 0x02;
        }
        if self.will {
            b |= 0x04;
        }
        b |= (self.will_qos.to_u8() & 0x03) << 3;
        if self.will_retain {
            b |= 0x20;
        }
        if self.password {
            b |= 0x40;
        }
        if self.username {
            b |= 0x80;
        }
        b
    }
}

// ── Connect Return Code ─────────────────────────────────────

/// CONNACK return code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConnectReturnCode {
    Accepted,
    UnacceptableProtocol,
    IdentifierRejected,
    ServerUnavailable,
    BadCredentials,
    NotAuthorized,
}

impl ConnectReturnCode {
    pub fn from_u8(val: u8) -> Result<Self, MqttError> {
        match val {
            0 => Ok(Self::Accepted),
            1 => Ok(Self::UnacceptableProtocol),
            2 => Ok(Self::IdentifierRejected),
            3 => Ok(Self::ServerUnavailable),
            4 => Ok(Self::BadCredentials),
            5 => Ok(Self::NotAuthorized),
            n => Err(MqttError::InvalidReturnCode(n)),
        }
    }

    pub fn to_u8(self) -> u8 {
        match self {
            Self::Accepted => 0,
            Self::UnacceptableProtocol => 1,
            Self::IdentifierRejected => 2,
            Self::ServerUnavailable => 3,
            Self::BadCredentials => 4,
            Self::NotAuthorized => 5,
        }
    }
}

// ── Subscription ────────────────────────────────────────────

/// A topic subscription request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Subscription {
    pub topic_filter: String,
    pub qos: QoS,
}

// ── Topic Validation ────────────────────────────────────────

/// Validate a topic filter (may contain + and # wildcards).
pub fn validate_topic_filter(filter: &str) -> Result<(), MqttError> {
    if filter.is_empty() {
        return Err(MqttError::InvalidTopicFilter("empty filter".into()));
    }
    let parts: Vec<&str> = filter.split('/').collect();
    for (i, part) in parts.iter().enumerate() {
        if part.contains('#') {
            if *part != "#" {
                return Err(MqttError::InvalidTopicFilter(
                    "# must occupy entire level".into(),
                ));
            }
            if i != parts.len() - 1 {
                return Err(MqttError::InvalidTopicFilter(
                    "# must be the last level".into(),
                ));
            }
        }
        if part.contains('+') && *part != "+" {
            return Err(MqttError::InvalidTopicFilter(
                "+ must occupy entire level".into(),
            ));
        }
    }
    Ok(())
}

/// Validate a topic name (no wildcards allowed).
pub fn validate_topic_name(topic: &str) -> Result<(), MqttError> {
    if topic.is_empty() {
        return Err(MqttError::InvalidTopicName("empty topic".into()));
    }
    if topic.contains('+') || topic.contains('#') {
        return Err(MqttError::InvalidTopicName("wildcards not allowed".into()));
    }
    Ok(())
}

/// Check if a topic name matches a topic filter.
pub fn topic_matches(filter: &str, topic: &str) -> bool {
    let filter_parts: Vec<&str> = filter.split('/').collect();
    let topic_parts: Vec<&str> = topic.split('/').collect();

    let mut fi = 0;
    let mut ti = 0;

    while fi < filter_parts.len() {
        if filter_parts[fi] == "#" {
            return true;
        }
        if ti >= topic_parts.len() {
            return false;
        }
        if filter_parts[fi] != "+" && filter_parts[fi] != topic_parts[ti] {
            return false;
        }
        fi += 1;
        ti += 1;
    }

    ti >= topic_parts.len()
}

// ── Packet Types ────────────────────────────────────────────

/// MQTT packet types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MqttPacket {
    Connect {
        client_id: String,
        clean_session: bool,
        keep_alive: u16,
        username: Option<String>,
        password: Option<Vec<u8>>,
        will_topic: Option<String>,
        will_message: Option<Vec<u8>>,
        will_qos: QoS,
        will_retain: bool,
    },
    Connack {
        session_present: bool,
        return_code: ConnectReturnCode,
    },
    Publish {
        dup: bool,
        qos: QoS,
        retain: bool,
        topic: String,
        packet_id: Option<u16>,
        payload: Vec<u8>,
    },
    Puback {
        packet_id: u16,
    },
    Pubrec {
        packet_id: u16,
    },
    Pubrel {
        packet_id: u16,
    },
    Pubcomp {
        packet_id: u16,
    },
    Subscribe {
        packet_id: u16,
        subscriptions: Vec<Subscription>,
    },
    Suback {
        packet_id: u16,
        return_codes: Vec<Option<QoS>>,
    },
    Unsubscribe {
        packet_id: u16,
        topics: Vec<String>,
    },
    Unsuback {
        packet_id: u16,
    },
    Pingreq,
    Pingresp,
    Disconnect,
}

// ── Encoding helpers ────────────────────────────────────────

fn encode_utf8_string(s: &str) -> Vec<u8> {
    let len = s.len() as u16;
    let mut buf = len.to_be_bytes().to_vec();
    buf.extend_from_slice(s.as_bytes());
    buf
}

fn encode_binary(data: &[u8]) -> Vec<u8> {
    let len = data.len() as u16;
    let mut buf = len.to_be_bytes().to_vec();
    buf.extend_from_slice(data);
    buf
}

fn read_utf8_string(data: &[u8], offset: usize) -> Result<(String, usize), MqttError> {
    if offset + 2 > data.len() {
        return Err(MqttError::UnexpectedEof);
    }
    let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    if offset + 2 + len > data.len() {
        return Err(MqttError::UnexpectedEof);
    }
    let s = String::from_utf8_lossy(&data[offset + 2..offset + 2 + len]).to_string();
    Ok((s, offset + 2 + len))
}

fn read_binary(data: &[u8], offset: usize) -> Result<(Vec<u8>, usize), MqttError> {
    if offset + 2 > data.len() {
        return Err(MqttError::UnexpectedEof);
    }
    let len = u16::from_be_bytes([data[offset], data[offset + 1]]) as usize;
    if offset + 2 + len > data.len() {
        return Err(MqttError::UnexpectedEof);
    }
    Ok((data[offset + 2..offset + 2 + len].to_vec(), offset + 2 + len))
}

impl MqttPacket {
    /// Encode the packet to bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MqttError> {
        match self {
            Self::Connect {
                client_id,
                clean_session,
                keep_alive,
                username,
                password,
                will_topic,
                will_message,
                will_qos,
                will_retain,
            } => {
                let mut payload = Vec::new();
                // Variable header
                payload.extend_from_slice(&[0x00, 0x04]); // protocol name length
                payload.extend_from_slice(b"MQTT");
                payload.push(0x04); // protocol level 4 (MQTT 3.1.1)
                let mut flags = ConnectFlags {
                    clean_session: *clean_session,
                    will: will_topic.is_some(),
                    will_qos: *will_qos,
                    will_retain: *will_retain,
                    password: password.is_some(),
                    username: username.is_some(),
                };
                if !flags.will {
                    flags.will_qos = QoS::AtMostOnce;
                    flags.will_retain = false;
                }
                payload.push(flags.to_byte());
                payload.extend_from_slice(&keep_alive.to_be_bytes());
                // Payload
                payload.extend(encode_utf8_string(client_id));
                if let Some(topic) = will_topic {
                    payload.extend(encode_utf8_string(topic));
                    if let Some(msg) = will_message {
                        payload.extend(encode_binary(msg));
                    } else {
                        payload.extend(encode_binary(&[]));
                    }
                }
                if let Some(user) = username {
                    payload.extend(encode_utf8_string(user));
                }
                if let Some(pass) = password {
                    payload.extend(encode_binary(pass));
                }
                let mut pkt = vec![0x10]; // CONNECT type
                pkt.extend(encode_remaining_length(payload.len() as u32)?);
                pkt.extend(payload);
                Ok(pkt)
            }
            Self::Connack { session_present, return_code } => {
                let mut pkt = vec![0x20, 0x02]; // CONNACK, remaining length 2
                pkt.push(if *session_present { 0x01 } else { 0x00 });
                pkt.push(return_code.to_u8());
                Ok(pkt)
            }
            Self::Publish { dup, qos, retain, topic, packet_id, payload } => {
                let mut var_hdr = encode_utf8_string(topic);
                if *qos != QoS::AtMostOnce {
                    if let Some(id) = packet_id {
                        var_hdr.extend_from_slice(&id.to_be_bytes());
                    }
                }
                var_hdr.extend_from_slice(payload);
                let mut fixed: u8 = 0x30;
                if *dup {
                    fixed |= 0x08;
                }
                fixed |= (qos.to_u8() & 0x03) << 1;
                if *retain {
                    fixed |= 0x01;
                }
                let mut pkt = vec![fixed];
                pkt.extend(encode_remaining_length(var_hdr.len() as u32)?);
                pkt.extend(var_hdr);
                Ok(pkt)
            }
            Self::Puback { packet_id } => {
                Ok(vec![0x40, 0x02, (packet_id >> 8) as u8, *packet_id as u8])
            }
            Self::Pubrec { packet_id } => {
                Ok(vec![0x50, 0x02, (packet_id >> 8) as u8, *packet_id as u8])
            }
            Self::Pubrel { packet_id } => {
                Ok(vec![0x62, 0x02, (packet_id >> 8) as u8, *packet_id as u8])
            }
            Self::Pubcomp { packet_id } => {
                Ok(vec![0x70, 0x02, (packet_id >> 8) as u8, *packet_id as u8])
            }
            Self::Subscribe { packet_id, subscriptions } => {
                let mut var = packet_id.to_be_bytes().to_vec();
                for sub in subscriptions {
                    var.extend(encode_utf8_string(&sub.topic_filter));
                    var.push(sub.qos.to_u8());
                }
                let mut pkt = vec![0x82];
                pkt.extend(encode_remaining_length(var.len() as u32)?);
                pkt.extend(var);
                Ok(pkt)
            }
            Self::Suback { packet_id, return_codes } => {
                let mut var = packet_id.to_be_bytes().to_vec();
                for rc in return_codes {
                    match rc {
                        Some(qos) => var.push(qos.to_u8()),
                        None => var.push(0x80),
                    }
                }
                let mut pkt = vec![0x90];
                pkt.extend(encode_remaining_length(var.len() as u32)?);
                pkt.extend(var);
                Ok(pkt)
            }
            Self::Unsubscribe { packet_id, topics } => {
                let mut var = packet_id.to_be_bytes().to_vec();
                for t in topics {
                    var.extend(encode_utf8_string(t));
                }
                let mut pkt = vec![0xA2];
                pkt.extend(encode_remaining_length(var.len() as u32)?);
                pkt.extend(var);
                Ok(pkt)
            }
            Self::Unsuback { packet_id } => {
                Ok(vec![0xB0, 0x02, (packet_id >> 8) as u8, *packet_id as u8])
            }
            Self::Pingreq => Ok(vec![0xC0, 0x00]),
            Self::Pingresp => Ok(vec![0xD0, 0x00]),
            Self::Disconnect => Ok(vec![0xE0, 0x00]),
        }
    }

    /// Parse an MQTT packet from bytes.
    pub fn parse(data: &[u8]) -> Result<(Self, usize), MqttError> {
        if data.is_empty() {
            return Err(MqttError::PacketTooShort);
        }
        let fixed_byte = data[0];
        let pkt_type = (fixed_byte >> 4) & 0x0F;
        let (remaining_len, len_bytes) = decode_remaining_length(&data[1..])?;
        let total = 1 + len_bytes + remaining_len as usize;
        if data.len() < total {
            return Err(MqttError::PacketTooShort);
        }
        let payload_start = 1 + len_bytes;
        let payload = &data[payload_start..total];

        let packet = match pkt_type {
            1 => Self::parse_connect(payload)?,
            2 => Self::parse_connack(payload)?,
            3 => Self::parse_publish(fixed_byte, payload)?,
            4 => Self::parse_ack(payload, |id| Self::Puback { packet_id: id })?,
            5 => Self::parse_ack(payload, |id| Self::Pubrec { packet_id: id })?,
            6 => Self::parse_ack(payload, |id| Self::Pubrel { packet_id: id })?,
            7 => Self::parse_ack(payload, |id| Self::Pubcomp { packet_id: id })?,
            8 => Self::parse_subscribe(payload)?,
            9 => Self::parse_suback(payload)?,
            10 => Self::parse_unsubscribe(payload)?,
            11 => Self::parse_ack(payload, |id| Self::Unsuback { packet_id: id })?,
            12 => Self::Pingreq,
            13 => Self::Pingresp,
            14 => Self::Disconnect,
            t => return Err(MqttError::InvalidPacketType(t)),
        };
        Ok((packet, total))
    }

    fn parse_connect(data: &[u8]) -> Result<Self, MqttError> {
        if data.len() < 10 {
            return Err(MqttError::PacketTooShort);
        }
        let proto_len = u16::from_be_bytes([data[0], data[1]]) as usize;
        if data.len() < 2 + proto_len + 4 {
            return Err(MqttError::PacketTooShort);
        }
        let proto_name = String::from_utf8_lossy(&data[2..2 + proto_len]).to_string();
        if proto_name != "MQTT" {
            return Err(MqttError::InvalidProtocolName(proto_name));
        }
        let level = data[2 + proto_len];
        if level != 4 {
            return Err(MqttError::UnsupportedProtocolLevel(level));
        }
        let flags = ConnectFlags::from_byte(data[2 + proto_len + 1])?;
        let keep_alive = u16::from_be_bytes([
            data[2 + proto_len + 2],
            data[2 + proto_len + 3],
        ]);
        let mut offset = 2 + proto_len + 4;
        let (client_id, next) = read_utf8_string(data, offset)?;
        offset = next;

        let (will_topic, will_message) = if flags.will {
            let (topic, next) = read_utf8_string(data, offset)?;
            offset = next;
            let (msg, next) = read_binary(data, offset)?;
            offset = next;
            (Some(topic), Some(msg))
        } else {
            (None, None)
        };

        let username = if flags.username {
            let (u, next) = read_utf8_string(data, offset)?;
            offset = next;
            Some(u)
        } else {
            None
        };

        let password = if flags.password {
            let (p, next) = read_binary(data, offset)?;
            offset = next;
            Some(p)
        } else {
            None
        };

        Ok(Self::Connect {
            client_id,
            clean_session: flags.clean_session,
            keep_alive,
            username,
            password,
            will_topic,
            will_message,
            will_qos: flags.will_qos,
            will_retain: flags.will_retain,
        })
    }

    fn parse_connack(data: &[u8]) -> Result<Self, MqttError> {
        if data.len() < 2 {
            return Err(MqttError::PacketTooShort);
        }
        Ok(Self::Connack {
            session_present: (data[0] & 0x01) != 0,
            return_code: ConnectReturnCode::from_u8(data[1])?,
        })
    }

    fn parse_publish(fixed_byte: u8, data: &[u8]) -> Result<Self, MqttError> {
        let dup = (fixed_byte & 0x08) != 0;
        let qos = QoS::from_u8((fixed_byte >> 1) & 0x03)?;
        let retain = (fixed_byte & 0x01) != 0;
        let (topic, mut offset) = read_utf8_string(data, 0)?;
        let packet_id = if qos != QoS::AtMostOnce {
            if offset + 2 > data.len() {
                return Err(MqttError::UnexpectedEof);
            }
            let id = u16::from_be_bytes([data[offset], data[offset + 1]]);
            offset += 2;
            Some(id)
        } else {
            None
        };
        let payload = data[offset..].to_vec();
        Ok(Self::Publish { dup, qos, retain, topic, packet_id, payload })
    }

    fn parse_ack<F>(data: &[u8], constructor: F) -> Result<Self, MqttError>
    where
        F: Fn(u16) -> Self,
    {
        if data.len() < 2 {
            return Err(MqttError::PacketTooShort);
        }
        Ok(constructor(u16::from_be_bytes([data[0], data[1]])))
    }

    fn parse_subscribe(data: &[u8]) -> Result<Self, MqttError> {
        if data.len() < 2 {
            return Err(MqttError::PacketTooShort);
        }
        let packet_id = u16::from_be_bytes([data[0], data[1]]);
        let mut offset = 2;
        let mut subscriptions = Vec::new();
        while offset < data.len() {
            let (filter, next) = read_utf8_string(data, offset)?;
            offset = next;
            if offset >= data.len() {
                return Err(MqttError::UnexpectedEof);
            }
            let qos = QoS::from_u8(data[offset])?;
            offset += 1;
            subscriptions.push(Subscription { topic_filter: filter, qos });
        }
        Ok(Self::Subscribe { packet_id, subscriptions })
    }

    fn parse_suback(data: &[u8]) -> Result<Self, MqttError> {
        if data.len() < 2 {
            return Err(MqttError::PacketTooShort);
        }
        let packet_id = u16::from_be_bytes([data[0], data[1]]);
        let return_codes = data[2..]
            .iter()
            .map(|b| if *b == 0x80 { None } else { QoS::from_u8(*b).ok() })
            .collect();
        Ok(Self::Suback { packet_id, return_codes })
    }

    fn parse_unsubscribe(data: &[u8]) -> Result<Self, MqttError> {
        if data.len() < 2 {
            return Err(MqttError::PacketTooShort);
        }
        let packet_id = u16::from_be_bytes([data[0], data[1]]);
        let mut offset = 2;
        let mut topics = Vec::new();
        while offset < data.len() {
            let (topic, next) = read_utf8_string(data, offset)?;
            offset = next;
            topics.push(topic);
        }
        Ok(Self::Unsubscribe { packet_id, topics })
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remaining_length_single_byte() {
        let encoded = encode_remaining_length(0).unwrap();
        assert_eq!(encoded, vec![0x00]);
        let (val, consumed) = decode_remaining_length(&encoded).unwrap();
        assert_eq!(val, 0);
        assert_eq!(consumed, 1);
    }

    #[test]
    fn remaining_length_multi_byte() {
        let encoded = encode_remaining_length(16_384).unwrap();
        assert_eq!(encoded, vec![0x80, 0x80, 0x01]);
        let (val, consumed) = decode_remaining_length(&encoded).unwrap();
        assert_eq!(val, 16_384);
        assert_eq!(consumed, 3);
    }

    #[test]
    fn remaining_length_max() {
        let max = 268_435_455;
        let encoded = encode_remaining_length(max).unwrap();
        assert_eq!(encoded.len(), 4);
        let (val, _) = decode_remaining_length(&encoded).unwrap();
        assert_eq!(val, max);
    }

    #[test]
    fn remaining_length_overflow() {
        assert!(encode_remaining_length(268_435_456).is_err());
    }

    #[test]
    fn qos_roundtrip() {
        assert_eq!(QoS::from_u8(0).unwrap(), QoS::AtMostOnce);
        assert_eq!(QoS::from_u8(1).unwrap(), QoS::AtLeastOnce);
        assert_eq!(QoS::from_u8(2).unwrap(), QoS::ExactlyOnce);
        assert!(QoS::from_u8(3).is_err());
    }

    #[test]
    fn connect_flags_roundtrip() {
        let flags = ConnectFlags {
            clean_session: true,
            will: true,
            will_qos: QoS::AtLeastOnce,
            will_retain: false,
            password: true,
            username: true,
        };
        let byte = flags.to_byte();
        let parsed = ConnectFlags::from_byte(byte).unwrap();
        assert_eq!(parsed, flags);
    }

    #[test]
    fn connect_packet_roundtrip() {
        let pkt = MqttPacket::Connect {
            client_id: "test-client".to_string(),
            clean_session: true,
            keep_alive: 60,
            username: Some("user".to_string()),
            password: Some(b"pass".to_vec()),
            will_topic: None,
            will_message: None,
            will_qos: QoS::AtMostOnce,
            will_retain: false,
        };
        let bytes = pkt.to_bytes().unwrap();
        let (parsed, consumed) = MqttPacket::parse(&bytes).unwrap();
        assert_eq!(consumed, bytes.len());
        if let MqttPacket::Connect { client_id, clean_session, username, password, .. } = parsed {
            assert_eq!(client_id, "test-client");
            assert!(clean_session);
            assert_eq!(username.as_deref(), Some("user"));
            assert_eq!(password.as_deref(), Some(b"pass".as_slice()));
        } else {
            panic!("expected Connect");
        }
    }

    #[test]
    fn connack_roundtrip() {
        let pkt = MqttPacket::Connack {
            session_present: true,
            return_code: ConnectReturnCode::Accepted,
        };
        let bytes = pkt.to_bytes().unwrap();
        let (parsed, _) = MqttPacket::parse(&bytes).unwrap();
        assert_eq!(parsed, pkt);
    }

    #[test]
    fn publish_qos0_roundtrip() {
        let pkt = MqttPacket::Publish {
            dup: false,
            qos: QoS::AtMostOnce,
            retain: true,
            topic: "sensor/temp".to_string(),
            packet_id: None,
            payload: b"25.3".to_vec(),
        };
        let bytes = pkt.to_bytes().unwrap();
        let (parsed, _) = MqttPacket::parse(&bytes).unwrap();
        assert_eq!(parsed, pkt);
    }

    #[test]
    fn publish_qos1_roundtrip() {
        let pkt = MqttPacket::Publish {
            dup: false,
            qos: QoS::AtLeastOnce,
            retain: false,
            topic: "data/log".to_string(),
            packet_id: Some(42),
            payload: b"hello".to_vec(),
        };
        let bytes = pkt.to_bytes().unwrap();
        let (parsed, _) = MqttPacket::parse(&bytes).unwrap();
        assert_eq!(parsed, pkt);
    }

    #[test]
    fn subscribe_roundtrip() {
        let pkt = MqttPacket::Subscribe {
            packet_id: 100,
            subscriptions: vec![
                Subscription { topic_filter: "a/b/c".to_string(), qos: QoS::AtLeastOnce },
                Subscription { topic_filter: "d/#".to_string(), qos: QoS::ExactlyOnce },
            ],
        };
        let bytes = pkt.to_bytes().unwrap();
        let (parsed, _) = MqttPacket::parse(&bytes).unwrap();
        assert_eq!(parsed, pkt);
    }

    #[test]
    fn topic_filter_validation() {
        assert!(validate_topic_filter("a/+/c").is_ok());
        assert!(validate_topic_filter("a/b/#").is_ok());
        assert!(validate_topic_filter("#").is_ok());
        assert!(validate_topic_filter("+").is_ok());
        assert!(validate_topic_filter("a/#/b").is_err());
        assert!(validate_topic_filter("a/b+c").is_err());
        assert!(validate_topic_filter("").is_err());
    }

    #[test]
    fn topic_matching() {
        assert!(topic_matches("sensor/+/temp", "sensor/1/temp"));
        assert!(!topic_matches("sensor/+/temp", "sensor/1/humidity"));
        assert!(topic_matches("sensor/#", "sensor/1/temp"));
        assert!(topic_matches("sensor/#", "sensor"));
        assert!(topic_matches("#", "anything/at/all"));
        assert!(!topic_matches("a/b", "a/b/c"));
    }

    #[test]
    fn pingreq_pingresp_disconnect() {
        for pkt in [MqttPacket::Pingreq, MqttPacket::Pingresp, MqttPacket::Disconnect] {
            let bytes = pkt.to_bytes().unwrap();
            assert_eq!(bytes.len(), 2);
            let (parsed, _) = MqttPacket::parse(&bytes).unwrap();
            assert_eq!(parsed, pkt);
        }
    }
}
