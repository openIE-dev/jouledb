//! DNS wire format parser and builder.
//!
//! Replaces `trust-dns` / `hickory-dns` with a pure-Rust DNS packet codec.
//! Supports header parsing (ID, QR, opcode, flags, section counts),
//! question/answer/authority/additional sections, common record types
//! (A, AAAA, CNAME, MX, NS, TXT, SOA, SRV, PTR), name compression,
//! and a packet builder for constructing DNS queries and responses.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────

/// DNS packet errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DnsError {
    /// Packet too short to contain a valid header.
    PacketTooShort,
    /// Name label exceeds 63 bytes.
    LabelTooLong(usize),
    /// Total name exceeds 255 bytes.
    NameTooLong,
    /// Compression pointer loop detected.
    CompressionLoop,
    /// Invalid compression pointer offset.
    InvalidPointer(u16),
    /// Unsupported record type.
    UnsupportedRecordType(u16),
    /// Unexpected end of data while parsing.
    UnexpectedEof,
    /// Invalid data in record.
    InvalidRecordData(String),
}

impl fmt::Display for DnsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PacketTooShort => write!(f, "packet too short for DNS header"),
            Self::LabelTooLong(n) => write!(f, "label length {n} exceeds 63"),
            Self::NameTooLong => write!(f, "domain name exceeds 255 bytes"),
            Self::CompressionLoop => write!(f, "compression pointer loop"),
            Self::InvalidPointer(off) => write!(f, "invalid compression pointer at {off}"),
            Self::UnsupportedRecordType(t) => write!(f, "unsupported record type {t}"),
            Self::UnexpectedEof => write!(f, "unexpected end of packet data"),
            Self::InvalidRecordData(msg) => write!(f, "invalid record data: {msg}"),
        }
    }
}

impl std::error::Error for DnsError {}

// ── Constants ───────────────────────────────────────────────

const DNS_HEADER_SIZE: usize = 12;
const MAX_LABEL_LEN: usize = 63;
const MAX_NAME_LEN: usize = 255;
const MAX_COMPRESSION_HOPS: usize = 128;

// ── Opcode ──────────────────────────────────────────────────

/// DNS opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Opcode {
    Query,
    IQuery,
    Status,
    Other(u8),
}

impl Opcode {
    fn from_bits(bits: u8) -> Self {
        match bits & 0x0F {
            0 => Self::Query,
            1 => Self::IQuery,
            2 => Self::Status,
            n => Self::Other(n),
        }
    }

    fn to_bits(self) -> u8 {
        match self {
            Self::Query => 0,
            Self::IQuery => 1,
            Self::Status => 2,
            Self::Other(n) => n & 0x0F,
        }
    }
}

// ── Response Code ───────────────────────────────────────────

/// DNS response code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rcode {
    NoError,
    FormatError,
    ServerFailure,
    NameError,
    NotImplemented,
    Refused,
    Other(u8),
}

impl Rcode {
    fn from_bits(bits: u8) -> Self {
        match bits & 0x0F {
            0 => Self::NoError,
            1 => Self::FormatError,
            2 => Self::ServerFailure,
            3 => Self::NameError,
            4 => Self::NotImplemented,
            5 => Self::Refused,
            n => Self::Other(n),
        }
    }

    fn to_bits(self) -> u8 {
        match self {
            Self::NoError => 0,
            Self::FormatError => 1,
            Self::ServerFailure => 2,
            Self::NameError => 3,
            Self::NotImplemented => 4,
            Self::Refused => 5,
            Self::Other(n) => n & 0x0F,
        }
    }
}

// ── Record Type ─────────────────────────────────────────────

/// DNS record type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RecordType {
    A,
    AAAA,
    CNAME,
    MX,
    NS,
    TXT,
    SOA,
    SRV,
    PTR,
    Other(u16),
}

impl RecordType {
    fn from_u16(val: u16) -> Self {
        match val {
            1 => Self::A,
            28 => Self::AAAA,
            5 => Self::CNAME,
            15 => Self::MX,
            2 => Self::NS,
            16 => Self::TXT,
            6 => Self::SOA,
            33 => Self::SRV,
            12 => Self::PTR,
            n => Self::Other(n),
        }
    }

    fn to_u16(self) -> u16 {
        match self {
            Self::A => 1,
            Self::NS => 2,
            Self::CNAME => 5,
            Self::SOA => 6,
            Self::PTR => 12,
            Self::MX => 15,
            Self::TXT => 16,
            Self::AAAA => 28,
            Self::SRV => 33,
            Self::Other(n) => n,
        }
    }
}

// ── Record Class ────────────────────────────────────────────

/// DNS class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordClass {
    IN,
    CH,
    HS,
    Other(u16),
}

impl RecordClass {
    fn from_u16(val: u16) -> Self {
        match val {
            1 => Self::IN,
            3 => Self::CH,
            4 => Self::HS,
            n => Self::Other(n),
        }
    }

    fn to_u16(self) -> u16 {
        match self {
            Self::IN => 1,
            Self::CH => 3,
            Self::HS => 4,
            Self::Other(n) => n,
        }
    }
}

// ── Header ──────────────────────────────────────────────────

/// DNS packet header (12 bytes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnsHeader {
    pub id: u16,
    pub qr: bool,
    pub opcode: Opcode,
    pub authoritative: bool,
    pub truncated: bool,
    pub recursion_desired: bool,
    pub recursion_available: bool,
    pub rcode: Rcode,
    pub question_count: u16,
    pub answer_count: u16,
    pub authority_count: u16,
    pub additional_count: u16,
}

impl DnsHeader {
    /// Parse a header from exactly 12 bytes.
    pub fn parse(data: &[u8]) -> Result<Self, DnsError> {
        if data.len() < DNS_HEADER_SIZE {
            return Err(DnsError::PacketTooShort);
        }
        let id = u16::from_be_bytes([data[0], data[1]]);
        let flags1 = data[2];
        let flags2 = data[3];
        Ok(Self {
            id,
            qr: (flags1 & 0x80) != 0,
            opcode: Opcode::from_bits((flags1 >> 3) & 0x0F),
            authoritative: (flags1 & 0x04) != 0,
            truncated: (flags1 & 0x02) != 0,
            recursion_desired: (flags1 & 0x01) != 0,
            recursion_available: (flags2 & 0x80) != 0,
            rcode: Rcode::from_bits(flags2 & 0x0F),
            question_count: u16::from_be_bytes([data[4], data[5]]),
            answer_count: u16::from_be_bytes([data[6], data[7]]),
            authority_count: u16::from_be_bytes([data[8], data[9]]),
            additional_count: u16::from_be_bytes([data[10], data[11]]),
        })
    }

    /// Serialize header to 12 bytes.
    pub fn to_bytes(&self) -> [u8; 12] {
        let mut buf = [0u8; 12];
        buf[0..2].copy_from_slice(&self.id.to_be_bytes());
        let mut flags1: u8 = 0;
        if self.qr {
            flags1 |= 0x80;
        }
        flags1 |= (self.opcode.to_bits() & 0x0F) << 3;
        if self.authoritative {
            flags1 |= 0x04;
        }
        if self.truncated {
            flags1 |= 0x02;
        }
        if self.recursion_desired {
            flags1 |= 0x01;
        }
        buf[2] = flags1;
        let mut flags2: u8 = 0;
        if self.recursion_available {
            flags2 |= 0x80;
        }
        flags2 |= self.rcode.to_bits() & 0x0F;
        buf[3] = flags2;
        buf[4..6].copy_from_slice(&self.question_count.to_be_bytes());
        buf[6..8].copy_from_slice(&self.answer_count.to_be_bytes());
        buf[8..10].copy_from_slice(&self.authority_count.to_be_bytes());
        buf[10..12].copy_from_slice(&self.additional_count.to_be_bytes());
        buf
    }
}

// ── Name encoding / decoding ────────────────────────────────

/// Read a DNS name from `packet` starting at `offset`.
/// Returns the decoded name and the new offset after the name.
pub fn read_name(packet: &[u8], offset: usize) -> Result<(String, usize), DnsError> {
    let mut labels: Vec<String> = Vec::new();
    let mut pos = offset;
    let mut jumped = false;
    let mut end_pos = 0usize;
    let mut hops = 0;

    loop {
        if pos >= packet.len() {
            return Err(DnsError::UnexpectedEof);
        }
        let len_byte = packet[pos];
        if len_byte == 0 {
            if !jumped {
                end_pos = pos + 1;
            }
            break;
        }
        // compression pointer
        if (len_byte & 0xC0) == 0xC0 {
            if pos + 1 >= packet.len() {
                return Err(DnsError::UnexpectedEof);
            }
            let ptr = u16::from_be_bytes([len_byte & 0x3F, packet[pos + 1]]) as usize;
            if !jumped {
                end_pos = pos + 2;
            }
            hops += 1;
            if hops > MAX_COMPRESSION_HOPS {
                return Err(DnsError::CompressionLoop);
            }
            pos = ptr;
            jumped = true;
            continue;
        }
        let label_len = len_byte as usize;
        if label_len > MAX_LABEL_LEN {
            return Err(DnsError::LabelTooLong(label_len));
        }
        if pos + 1 + label_len > packet.len() {
            return Err(DnsError::UnexpectedEof);
        }
        let label = String::from_utf8_lossy(&packet[pos + 1..pos + 1 + label_len]).to_string();
        labels.push(label);
        pos += 1 + label_len;
    }

    let name = if labels.is_empty() {
        ".".to_string()
    } else {
        labels.join(".")
    };

    if name.len() > MAX_NAME_LEN {
        return Err(DnsError::NameTooLong);
    }
    Ok((name, end_pos))
}

/// Encode a domain name to DNS wire format (without compression).
pub fn encode_name(name: &str) -> Result<Vec<u8>, DnsError> {
    let mut buf = Vec::new();
    let name = name.trim_end_matches('.');
    if name.is_empty() {
        buf.push(0);
        return Ok(buf);
    }
    for label in name.split('.') {
        let len = label.len();
        if len > MAX_LABEL_LEN {
            return Err(DnsError::LabelTooLong(len));
        }
        buf.push(len as u8);
        buf.extend_from_slice(label.as_bytes());
    }
    buf.push(0);
    let total: usize = buf.len();
    if total > MAX_NAME_LEN {
        return Err(DnsError::NameTooLong);
    }
    Ok(buf)
}

// ── Question ────────────────────────────────────────────────

/// DNS question entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DnsQuestion {
    pub name: String,
    pub record_type: RecordType,
    pub record_class: RecordClass,
}

impl DnsQuestion {
    pub fn parse(packet: &[u8], offset: usize) -> Result<(Self, usize), DnsError> {
        let (name, pos) = read_name(packet, offset)?;
        if pos + 4 > packet.len() {
            return Err(DnsError::UnexpectedEof);
        }
        let rtype = u16::from_be_bytes([packet[pos], packet[pos + 1]]);
        let rclass = u16::from_be_bytes([packet[pos + 2], packet[pos + 3]]);
        Ok((
            Self {
                name,
                record_type: RecordType::from_u16(rtype),
                record_class: RecordClass::from_u16(rclass),
            },
            pos + 4,
        ))
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>, DnsError> {
        let mut buf = encode_name(&self.name)?;
        buf.extend_from_slice(&self.record_type.to_u16().to_be_bytes());
        buf.extend_from_slice(&self.record_class.to_u16().to_be_bytes());
        Ok(buf)
    }
}

// ── Record Data ─────────────────────────────────────────────

/// Parsed record data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordData {
    A([u8; 4]),
    AAAA([u8; 16]),
    CNAME(String),
    MX { preference: u16, exchange: String },
    NS(String),
    TXT(Vec<String>),
    SOA {
        mname: String,
        rname: String,
        serial: u32,
        refresh: u32,
        retry: u32,
        expire: u32,
        minimum: u32,
    },
    SRV {
        priority: u16,
        weight: u16,
        port: u16,
        target: String,
    },
    PTR(String),
    Unknown(Vec<u8>),
}

// ── Resource Record ─────────────────────────────────────────

/// A DNS resource record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceRecord {
    pub name: String,
    pub record_type: RecordType,
    pub record_class: RecordClass,
    pub ttl: u32,
    pub data: RecordData,
}

impl ResourceRecord {
    /// Parse a resource record from packet at offset.
    pub fn parse(packet: &[u8], offset: usize) -> Result<(Self, usize), DnsError> {
        let (name, mut pos) = read_name(packet, offset)?;
        if pos + 10 > packet.len() {
            return Err(DnsError::UnexpectedEof);
        }
        let rtype = u16::from_be_bytes([packet[pos], packet[pos + 1]]);
        let rclass = u16::from_be_bytes([packet[pos + 2], packet[pos + 3]]);
        let ttl = u32::from_be_bytes([packet[pos + 4], packet[pos + 5], packet[pos + 6], packet[pos + 7]]);
        let rdlength = u16::from_be_bytes([packet[pos + 8], packet[pos + 9]]) as usize;
        pos += 10;
        if pos + rdlength > packet.len() {
            return Err(DnsError::UnexpectedEof);
        }
        let record_type = RecordType::from_u16(rtype);
        let data = parse_rdata(packet, pos, rdlength, record_type)?;
        Ok((
            Self {
                name,
                record_type,
                record_class: RecordClass::from_u16(rclass),
                ttl,
                data,
            },
            pos + rdlength,
        ))
    }

    /// Serialize to bytes (without name compression).
    pub fn to_bytes(&self) -> Result<Vec<u8>, DnsError> {
        let mut buf = encode_name(&self.name)?;
        buf.extend_from_slice(&self.record_type.to_u16().to_be_bytes());
        buf.extend_from_slice(&self.record_class.to_u16().to_be_bytes());
        buf.extend_from_slice(&self.ttl.to_be_bytes());
        let rdata = self.rdata_bytes()?;
        buf.extend_from_slice(&(rdata.len() as u16).to_be_bytes());
        buf.extend_from_slice(&rdata);
        Ok(buf)
    }

    fn rdata_bytes(&self) -> Result<Vec<u8>, DnsError> {
        match &self.data {
            RecordData::A(ip) => Ok(ip.to_vec()),
            RecordData::AAAA(ip) => Ok(ip.to_vec()),
            RecordData::CNAME(name) | RecordData::NS(name) | RecordData::PTR(name) => {
                encode_name(name)
            }
            RecordData::MX { preference, exchange } => {
                let mut buf = preference.to_be_bytes().to_vec();
                buf.extend(encode_name(exchange)?);
                Ok(buf)
            }
            RecordData::TXT(strings) => {
                let mut buf = Vec::new();
                for s in strings {
                    let bytes = s.as_bytes();
                    if bytes.len() > 255 {
                        return Err(DnsError::InvalidRecordData("TXT string > 255 bytes".into()));
                    }
                    buf.push(bytes.len() as u8);
                    buf.extend_from_slice(bytes);
                }
                Ok(buf)
            }
            RecordData::SOA { mname, rname, serial, refresh, retry, expire, minimum } => {
                let mut buf = encode_name(mname)?;
                buf.extend(encode_name(rname)?);
                buf.extend_from_slice(&serial.to_be_bytes());
                buf.extend_from_slice(&refresh.to_be_bytes());
                buf.extend_from_slice(&retry.to_be_bytes());
                buf.extend_from_slice(&expire.to_be_bytes());
                buf.extend_from_slice(&minimum.to_be_bytes());
                Ok(buf)
            }
            RecordData::SRV { priority, weight, port, target } => {
                let mut buf = priority.to_be_bytes().to_vec();
                buf.extend_from_slice(&weight.to_be_bytes());
                buf.extend_from_slice(&port.to_be_bytes());
                buf.extend(encode_name(target)?);
                Ok(buf)
            }
            RecordData::Unknown(raw) => Ok(raw.clone()),
        }
    }
}

fn parse_rdata(
    packet: &[u8],
    offset: usize,
    rdlength: usize,
    rtype: RecordType,
) -> Result<RecordData, DnsError> {
    match rtype {
        RecordType::A => {
            if rdlength != 4 {
                return Err(DnsError::InvalidRecordData("A record must be 4 bytes".into()));
            }
            let mut ip = [0u8; 4];
            ip.copy_from_slice(&packet[offset..offset + 4]);
            Ok(RecordData::A(ip))
        }
        RecordType::AAAA => {
            if rdlength != 16 {
                return Err(DnsError::InvalidRecordData("AAAA record must be 16 bytes".into()));
            }
            let mut ip = [0u8; 16];
            ip.copy_from_slice(&packet[offset..offset + 16]);
            Ok(RecordData::AAAA(ip))
        }
        RecordType::CNAME => {
            let (name, _) = read_name(packet, offset)?;
            Ok(RecordData::CNAME(name))
        }
        RecordType::NS => {
            let (name, _) = read_name(packet, offset)?;
            Ok(RecordData::NS(name))
        }
        RecordType::PTR => {
            let (name, _) = read_name(packet, offset)?;
            Ok(RecordData::PTR(name))
        }
        RecordType::MX => {
            if rdlength < 3 {
                return Err(DnsError::InvalidRecordData("MX record too short".into()));
            }
            let pref = u16::from_be_bytes([packet[offset], packet[offset + 1]]);
            let (exchange, _) = read_name(packet, offset + 2)?;
            Ok(RecordData::MX { preference: pref, exchange })
        }
        RecordType::TXT => {
            let mut strings = Vec::new();
            let mut pos = offset;
            let end = offset + rdlength;
            while pos < end {
                if pos >= packet.len() {
                    return Err(DnsError::UnexpectedEof);
                }
                let slen = packet[pos] as usize;
                pos += 1;
                if pos + slen > end {
                    return Err(DnsError::UnexpectedEof);
                }
                strings.push(String::from_utf8_lossy(&packet[pos..pos + slen]).to_string());
                pos += slen;
            }
            Ok(RecordData::TXT(strings))
        }
        RecordType::SOA => {
            let (mname, pos1) = read_name(packet, offset)?;
            let (rname, pos2) = read_name(packet, pos1)?;
            if pos2 + 20 > packet.len() {
                return Err(DnsError::UnexpectedEof);
            }
            let serial = u32::from_be_bytes([packet[pos2], packet[pos2 + 1], packet[pos2 + 2], packet[pos2 + 3]]);
            let refresh = u32::from_be_bytes([packet[pos2 + 4], packet[pos2 + 5], packet[pos2 + 6], packet[pos2 + 7]]);
            let retry = u32::from_be_bytes([packet[pos2 + 8], packet[pos2 + 9], packet[pos2 + 10], packet[pos2 + 11]]);
            let expire = u32::from_be_bytes([packet[pos2 + 12], packet[pos2 + 13], packet[pos2 + 14], packet[pos2 + 15]]);
            let minimum = u32::from_be_bytes([packet[pos2 + 16], packet[pos2 + 17], packet[pos2 + 18], packet[pos2 + 19]]);
            Ok(RecordData::SOA { mname, rname, serial, refresh, retry, expire, minimum })
        }
        RecordType::SRV => {
            if rdlength < 7 {
                return Err(DnsError::InvalidRecordData("SRV record too short".into()));
            }
            let priority = u16::from_be_bytes([packet[offset], packet[offset + 1]]);
            let weight = u16::from_be_bytes([packet[offset + 2], packet[offset + 3]]);
            let port = u16::from_be_bytes([packet[offset + 4], packet[offset + 5]]);
            let (target, _) = read_name(packet, offset + 6)?;
            Ok(RecordData::SRV { priority, weight, port, target })
        }
        RecordType::Other(_) => {
            Ok(RecordData::Unknown(packet[offset..offset + rdlength].to_vec()))
        }
    }
}

// ── Full Packet ─────────────────────────────────────────────

/// A fully parsed DNS packet.
#[derive(Debug, Clone)]
pub struct DnsPacket {
    pub header: DnsHeader,
    pub questions: Vec<DnsQuestion>,
    pub answers: Vec<ResourceRecord>,
    pub authorities: Vec<ResourceRecord>,
    pub additionals: Vec<ResourceRecord>,
}

impl DnsPacket {
    /// Parse a complete DNS packet from bytes.
    pub fn parse(data: &[u8]) -> Result<Self, DnsError> {
        let header = DnsHeader::parse(data)?;
        let mut offset = DNS_HEADER_SIZE;

        let mut questions = Vec::new();
        for _ in 0..header.question_count {
            let (q, next) = DnsQuestion::parse(data, offset)?;
            questions.push(q);
            offset = next;
        }

        let mut answers = Vec::new();
        for _ in 0..header.answer_count {
            let (rr, next) = ResourceRecord::parse(data, offset)?;
            answers.push(rr);
            offset = next;
        }

        let mut authorities = Vec::new();
        for _ in 0..header.authority_count {
            let (rr, next) = ResourceRecord::parse(data, offset)?;
            authorities.push(rr);
            offset = next;
        }

        let mut additionals = Vec::new();
        for _ in 0..header.additional_count {
            let (rr, next) = ResourceRecord::parse(data, offset)?;
            additionals.push(rr);
            offset = next;
        }

        Ok(Self { header, questions, answers, authorities, additionals })
    }

    /// Serialize to wire format.
    pub fn to_bytes(&self) -> Result<Vec<u8>, DnsError> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.header.to_bytes());
        for q in &self.questions {
            buf.extend(q.to_bytes()?);
        }
        for rr in &self.answers {
            buf.extend(rr.to_bytes()?);
        }
        for rr in &self.authorities {
            buf.extend(rr.to_bytes()?);
        }
        for rr in &self.additionals {
            buf.extend(rr.to_bytes()?);
        }
        Ok(buf)
    }
}

// ── Packet Builder ──────────────────────────────────────────

/// Fluent builder for DNS packets.
pub struct DnsPacketBuilder {
    id: u16,
    qr: bool,
    opcode: Opcode,
    authoritative: bool,
    truncated: bool,
    recursion_desired: bool,
    recursion_available: bool,
    rcode: Rcode,
    questions: Vec<DnsQuestion>,
    answers: Vec<ResourceRecord>,
    authorities: Vec<ResourceRecord>,
    additionals: Vec<ResourceRecord>,
}

impl DnsPacketBuilder {
    pub fn new(id: u16) -> Self {
        Self {
            id,
            qr: false,
            opcode: Opcode::Query,
            authoritative: false,
            truncated: false,
            recursion_desired: true,
            recursion_available: false,
            rcode: Rcode::NoError,
            questions: Vec::new(),
            answers: Vec::new(),
            authorities: Vec::new(),
            additionals: Vec::new(),
        }
    }

    pub fn response(mut self) -> Self {
        self.qr = true;
        self
    }

    pub fn authoritative(mut self, val: bool) -> Self {
        self.authoritative = val;
        self
    }

    pub fn rcode(mut self, code: Rcode) -> Self {
        self.rcode = code;
        self
    }

    pub fn recursion_desired(mut self, val: bool) -> Self {
        self.recursion_desired = val;
        self
    }

    pub fn recursion_available(mut self, val: bool) -> Self {
        self.recursion_available = val;
        self
    }

    pub fn question(mut self, name: &str, rtype: RecordType) -> Self {
        self.questions.push(DnsQuestion {
            name: name.to_string(),
            record_type: rtype,
            record_class: RecordClass::IN,
        });
        self
    }

    pub fn answer(mut self, rr: ResourceRecord) -> Self {
        self.answers.push(rr);
        self
    }

    pub fn authority(mut self, rr: ResourceRecord) -> Self {
        self.authorities.push(rr);
        self
    }

    pub fn additional(mut self, rr: ResourceRecord) -> Self {
        self.additionals.push(rr);
        self
    }

    pub fn build(self) -> DnsPacket {
        DnsPacket {
            header: DnsHeader {
                id: self.id,
                qr: self.qr,
                opcode: self.opcode,
                authoritative: self.authoritative,
                truncated: self.truncated,
                recursion_desired: self.recursion_desired,
                recursion_available: self.recursion_available,
                rcode: self.rcode,
                question_count: self.questions.len() as u16,
                answer_count: self.answers.len() as u16,
                authority_count: self.authorities.len() as u16,
                additional_count: self.additionals.len() as u16,
            },
            questions: self.questions,
            answers: self.answers,
            authorities: self.authorities,
            additionals: self.additionals,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_roundtrip() {
        let hdr = DnsHeader {
            id: 0xABCD,
            qr: true,
            opcode: Opcode::Query,
            authoritative: true,
            truncated: false,
            recursion_desired: true,
            recursion_available: true,
            rcode: Rcode::NoError,
            question_count: 1,
            answer_count: 2,
            authority_count: 0,
            additional_count: 1,
        };
        let bytes = hdr.to_bytes();
        assert_eq!(bytes.len(), 12);
        let parsed = DnsHeader::parse(&bytes).unwrap();
        assert_eq!(parsed, hdr);
    }

    #[test]
    fn header_too_short() {
        assert_eq!(DnsHeader::parse(&[0; 5]).unwrap_err(), DnsError::PacketTooShort);
    }

    #[test]
    fn name_encode_decode() {
        let encoded = encode_name("example.com").unwrap();
        // 7 e x a m p l e 3 c o m 0
        assert_eq!(encoded[0], 7);
        assert_eq!(&encoded[1..8], b"example");
        assert_eq!(encoded[8], 3);
        assert_eq!(&encoded[9..12], b"com");
        assert_eq!(encoded[12], 0);

        let (name, end) = read_name(&encoded, 0).unwrap();
        assert_eq!(name, "example.com");
        assert_eq!(end, encoded.len());
    }

    #[test]
    fn name_compression() {
        // Build packet with name at offset 0 then a pointer
        let mut pkt = encode_name("foo.bar").unwrap();
        let ptr_offset = pkt.len();
        // add a pointer to offset 0
        pkt.push(0xC0);
        pkt.push(0x00);
        let (name, _) = read_name(&pkt, ptr_offset).unwrap();
        assert_eq!(name, "foo.bar");
    }

    #[test]
    fn label_too_long() {
        let long_label = "a".repeat(64);
        let name = format!("{long_label}.com");
        assert!(matches!(encode_name(&name), Err(DnsError::LabelTooLong(64))));
    }

    #[test]
    fn question_roundtrip() {
        let q = DnsQuestion {
            name: "example.org".to_string(),
            record_type: RecordType::AAAA,
            record_class: RecordClass::IN,
        };
        let bytes = q.to_bytes().unwrap();
        let (parsed, _) = DnsQuestion::parse(&bytes, 0).unwrap();
        assert_eq!(parsed, q);
    }

    #[test]
    fn a_record_roundtrip() {
        let rr = ResourceRecord {
            name: "test.com".to_string(),
            record_type: RecordType::A,
            record_class: RecordClass::IN,
            ttl: 300,
            data: RecordData::A([192, 168, 1, 1]),
        };
        let bytes = rr.to_bytes().unwrap();
        let (parsed, _) = ResourceRecord::parse(&bytes, 0).unwrap();
        assert_eq!(parsed, rr);
    }

    #[test]
    fn aaaa_record_roundtrip() {
        let rr = ResourceRecord {
            name: "v6.test.com".to_string(),
            record_type: RecordType::AAAA,
            record_class: RecordClass::IN,
            ttl: 600,
            data: RecordData::AAAA([0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
        };
        let bytes = rr.to_bytes().unwrap();
        let (parsed, _) = ResourceRecord::parse(&bytes, 0).unwrap();
        assert_eq!(parsed, rr);
    }

    #[test]
    fn mx_record_roundtrip() {
        let rr = ResourceRecord {
            name: "mail.com".to_string(),
            record_type: RecordType::MX,
            record_class: RecordClass::IN,
            ttl: 3600,
            data: RecordData::MX { preference: 10, exchange: "smtp.mail.com".to_string() },
        };
        let bytes = rr.to_bytes().unwrap();
        let (parsed, _) = ResourceRecord::parse(&bytes, 0).unwrap();
        assert_eq!(parsed, rr);
    }

    #[test]
    fn txt_record_roundtrip() {
        let rr = ResourceRecord {
            name: "spf.example.com".to_string(),
            record_type: RecordType::TXT,
            record_class: RecordClass::IN,
            ttl: 300,
            data: RecordData::TXT(vec!["v=spf1 include:example.com".to_string()]),
        };
        let bytes = rr.to_bytes().unwrap();
        let (parsed, _) = ResourceRecord::parse(&bytes, 0).unwrap();
        assert_eq!(parsed, rr);
    }

    #[test]
    fn full_query_packet_builder() {
        let pkt = DnsPacketBuilder::new(0x1234)
            .recursion_desired(true)
            .question("example.com", RecordType::A)
            .build();
        assert!(!pkt.header.qr);
        assert_eq!(pkt.header.question_count, 1);
        assert_eq!(pkt.questions[0].name, "example.com");
    }

    #[test]
    fn full_packet_roundtrip() {
        let pkt = DnsPacketBuilder::new(0x5678)
            .response()
            .authoritative(true)
            .question("example.com", RecordType::A)
            .answer(ResourceRecord {
                name: "example.com".to_string(),
                record_type: RecordType::A,
                record_class: RecordClass::IN,
                ttl: 300,
                data: RecordData::A([93, 184, 216, 34]),
            })
            .build();
        let bytes = pkt.to_bytes().unwrap();
        let parsed = DnsPacket::parse(&bytes).unwrap();
        assert_eq!(parsed.header.id, 0x5678);
        assert!(parsed.header.qr);
        assert!(parsed.header.authoritative);
        assert_eq!(parsed.answers.len(), 1);
        assert_eq!(parsed.answers[0].data, RecordData::A([93, 184, 216, 34]));
    }

    #[test]
    fn srv_record_roundtrip() {
        let rr = ResourceRecord {
            name: "_http._tcp.example.com".to_string(),
            record_type: RecordType::SRV,
            record_class: RecordClass::IN,
            ttl: 120,
            data: RecordData::SRV {
                priority: 10,
                weight: 60,
                port: 8080,
                target: "web.example.com".to_string(),
            },
        };
        let bytes = rr.to_bytes().unwrap();
        let (parsed, _) = ResourceRecord::parse(&bytes, 0).unwrap();
        assert_eq!(parsed, rr);
    }

    #[test]
    fn record_type_values() {
        assert_eq!(RecordType::A.to_u16(), 1);
        assert_eq!(RecordType::AAAA.to_u16(), 28);
        assert_eq!(RecordType::MX.to_u16(), 15);
        assert_eq!(RecordType::SOA.to_u16(), 6);
        assert_eq!(RecordType::SRV.to_u16(), 33);
        assert_eq!(RecordType::from_u16(1), RecordType::A);
        assert_eq!(RecordType::from_u16(999), RecordType::Other(999));
    }

    #[test]
    fn rcode_roundtrip() {
        assert_eq!(Rcode::from_bits(Rcode::NameError.to_bits()), Rcode::NameError);
        assert_eq!(Rcode::from_bits(Rcode::Refused.to_bits()), Rcode::Refused);
    }
}
