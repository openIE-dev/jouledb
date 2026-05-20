//! ASN.1 DER/BER codec — tag-length-value encoding for universal types.
//!
//! Supports BOOLEAN, INTEGER, BIT STRING, OCTET STRING, NULL, OID,
//! SEQUENCE, SET, UTF8String, PrintableString, UTCTime, and constructed types.

use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Asn1Error {
    UnexpectedEof,
    InvalidTag(u8),
    InvalidLength,
    IndefiniteLengthNotDer,
    ContentOverflow,
    InvalidOid,
    InvalidUtf8,
    InvalidPrintableString,
    InvalidBooleanLength,
    InvalidInteger,
    TrailingData,
    NestingTooDeep,
}

impl fmt::Display for Asn1Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::InvalidTag(t) => write!(f, "invalid tag: 0x{t:02x}"),
            Self::InvalidLength => write!(f, "invalid length encoding"),
            Self::IndefiniteLengthNotDer => write!(f, "indefinite length not allowed in DER"),
            Self::ContentOverflow => write!(f, "content exceeds declared length"),
            Self::InvalidOid => write!(f, "invalid OID encoding"),
            Self::InvalidUtf8 => write!(f, "invalid UTF-8 string"),
            Self::InvalidPrintableString => write!(f, "invalid PrintableString character"),
            Self::InvalidBooleanLength => write!(f, "boolean must be exactly 1 byte"),
            Self::InvalidInteger => write!(f, "invalid integer encoding"),
            Self::TrailingData => write!(f, "trailing data after TLV"),
            Self::NestingTooDeep => write!(f, "nesting too deep"),
        }
    }
}

impl std::error::Error for Asn1Error {}

// ── Tag classes and numbers ─────────────────────────────────────

/// ASN.1 tag class.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagClass {
    Universal = 0,
    Application = 1,
    ContextSpecific = 2,
    Private = 3,
}

/// Common universal tag numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UniversalTag {
    Boolean = 1,
    Integer = 2,
    BitString = 3,
    OctetString = 4,
    Null = 5,
    Oid = 6,
    Utf8String = 12,
    Sequence = 16,
    Set = 17,
    PrintableString = 19,
    Ia5String = 22,
    UtcTime = 23,
    GeneralizedTime = 24,
}

/// A parsed ASN.1 tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tag {
    pub class: TagClass,
    pub constructed: bool,
    pub number: u32,
}

impl Tag {
    pub fn new(class: TagClass, constructed: bool, number: u32) -> Self {
        Self { class, constructed, number }
    }

    pub fn universal(number: u32) -> Self {
        Self { class: TagClass::Universal, constructed: false, number }
    }

    pub fn sequence() -> Self {
        Self { class: TagClass::Universal, constructed: true, number: 16 }
    }

    pub fn set() -> Self {
        Self { class: TagClass::Universal, constructed: true, number: 17 }
    }
}

// ── Value ───────────────────────────────────────────────────────

/// A parsed ASN.1 TLV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tlv {
    pub tag: Tag,
    pub value: TlvValue,
}

/// The content of a TLV.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TlvValue {
    /// Primitive: raw bytes.
    Primitive(Vec<u8>),
    /// Constructed: sequence of child TLVs.
    Constructed(Vec<Tlv>),
}

// ── OID ─────────────────────────────────────────────────────────

/// An ASN.1 Object Identifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Oid {
    pub components: Vec<u32>,
}

impl Oid {
    pub fn new(components: Vec<u32>) -> Self {
        Self { components }
    }

    /// Parse from dotted string, e.g. "1.2.840.113549.1.1.1"
    pub fn from_str(s: &str) -> Result<Self, Asn1Error> {
        let components: Result<Vec<u32>, _> = s.split('.').map(|p| p.parse()).collect();
        let components = components.map_err(|_| Asn1Error::InvalidOid)?;
        if components.len() < 2 {
            return Err(Asn1Error::InvalidOid);
        }
        Ok(Self { components })
    }

    /// Encode to DER bytes (just the value, no tag/length).
    pub fn encode_value(&self) -> Result<Vec<u8>, Asn1Error> {
        if self.components.len() < 2 {
            return Err(Asn1Error::InvalidOid);
        }
        let mut buf = Vec::new();
        // First two components: 40 * c0 + c1
        let first = self.components[0] * 40 + self.components[1];
        encode_oid_component(&mut buf, first);
        for c in &self.components[2..] {
            encode_oid_component(&mut buf, *c);
        }
        Ok(buf)
    }

    /// Decode from DER value bytes.
    pub fn decode_value(data: &[u8]) -> Result<Self, Asn1Error> {
        if data.is_empty() {
            return Err(Asn1Error::InvalidOid);
        }
        let mut components = Vec::new();
        let mut pos = 0;
        let (first, new_pos) = decode_oid_component(data, pos)?;
        pos = new_pos;
        // First byte encodes two components
        components.push(first / 40);
        components.push(first % 40);
        while pos < data.len() {
            let (comp, new_pos) = decode_oid_component(data, pos)?;
            pos = new_pos;
            components.push(comp);
        }
        Ok(Self { components })
    }

    /// Format as dotted string.
    pub fn to_dotted_string(&self) -> String {
        self.components.iter().map(|c| c.to_string()).collect::<Vec<_>>().join(".")
    }
}

impl fmt::Display for Oid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_dotted_string())
    }
}

fn encode_oid_component(buf: &mut Vec<u8>, mut value: u32) {
    if value == 0 {
        buf.push(0);
        return;
    }
    let mut bytes = Vec::new();
    while value > 0 {
        bytes.push((value & 0x7F) as u8);
        value >>= 7;
    }
    bytes.reverse();
    for (i, b) in bytes.iter().enumerate() {
        if i < bytes.len() - 1 {
            buf.push(b | 0x80);
        } else {
            buf.push(*b);
        }
    }
}

fn decode_oid_component(data: &[u8], mut pos: usize) -> Result<(u32, usize), Asn1Error> {
    let mut result: u32 = 0;
    loop {
        if pos >= data.len() {
            return Err(Asn1Error::UnexpectedEof);
        }
        let byte = data[pos];
        pos += 1;
        result = result.checked_shl(7).ok_or(Asn1Error::InvalidOid)?;
        result |= (byte & 0x7F) as u32;
        if byte & 0x80 == 0 {
            return Ok((result, pos));
        }
    }
}

// ── DER Encoder ─────────────────────────────────────────────────

/// DER encoder.
#[derive(Debug, Default)]
pub struct DerEncoder {
    buf: Vec<u8>,
}

impl DerEncoder {
    pub fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn write_tag(&mut self, tag: &Tag) {
        let class_bits = (tag.class as u8) << 6;
        let constructed_bit = if tag.constructed { 0x20 } else { 0 };

        if tag.number <= 30 {
            self.buf.push(class_bits | constructed_bit | tag.number as u8);
        } else {
            self.buf.push(class_bits | constructed_bit | 0x1F);
            // Multi-byte tag number
            let mut num = tag.number;
            let mut bytes = Vec::new();
            while num > 0 {
                bytes.push((num & 0x7F) as u8);
                num >>= 7;
            }
            bytes.reverse();
            for (i, b) in bytes.iter().enumerate() {
                if i < bytes.len() - 1 {
                    self.buf.push(b | 0x80);
                } else {
                    self.buf.push(*b);
                }
            }
        }
    }

    fn write_length(&mut self, len: usize) {
        if len <= 127 {
            self.buf.push(len as u8);
        } else if len <= 0xFF {
            self.buf.push(0x81);
            self.buf.push(len as u8);
        } else if len <= 0xFFFF {
            self.buf.push(0x82);
            self.buf.extend_from_slice(&(len as u16).to_be_bytes());
        } else if len <= 0xFF_FFFF {
            self.buf.push(0x83);
            self.buf.push((len >> 16) as u8);
            self.buf.push((len >> 8) as u8);
            self.buf.push(len as u8);
        } else {
            self.buf.push(0x84);
            self.buf.extend_from_slice(&(len as u32).to_be_bytes());
        }
    }

    /// Encode a complete TLV.
    pub fn encode_tlv(&mut self, tlv: &Tlv) {
        self.write_tag(&tlv.tag);
        match &tlv.value {
            TlvValue::Primitive(data) => {
                self.write_length(data.len());
                self.buf.extend_from_slice(data);
            }
            TlvValue::Constructed(children) => {
                let mut child_enc = DerEncoder::new();
                for child in children {
                    child_enc.encode_tlv(child);
                }
                let child_bytes = child_enc.finish();
                self.write_length(child_bytes.len());
                self.buf.extend_from_slice(&child_bytes);
            }
        }
    }

    /// Encode a BOOLEAN.
    pub fn encode_boolean(&mut self, val: bool) {
        self.write_tag(&Tag::universal(UniversalTag::Boolean as u32));
        self.write_length(1);
        self.buf.push(if val { 0xFF } else { 0x00 });
    }

    /// Encode an INTEGER (signed, minimal encoding).
    pub fn encode_integer(&mut self, val: i64) {
        self.write_tag(&Tag::universal(UniversalTag::Integer as u32));
        let bytes = integer_to_der_bytes(val);
        self.write_length(bytes.len());
        self.buf.extend_from_slice(&bytes);
    }

    /// Encode a NULL.
    pub fn encode_null(&mut self) {
        self.write_tag(&Tag::universal(UniversalTag::Null as u32));
        self.write_length(0);
    }

    /// Encode an OCTET STRING.
    pub fn encode_octet_string(&mut self, data: &[u8]) {
        self.write_tag(&Tag::universal(UniversalTag::OctetString as u32));
        self.write_length(data.len());
        self.buf.extend_from_slice(data);
    }

    /// Encode a BIT STRING (with zero unused bits).
    pub fn encode_bit_string(&mut self, data: &[u8], unused_bits: u8) {
        self.write_tag(&Tag::universal(UniversalTag::BitString as u32));
        self.write_length(data.len() + 1);
        self.buf.push(unused_bits);
        self.buf.extend_from_slice(data);
    }

    /// Encode an OID.
    pub fn encode_oid(&mut self, oid: &Oid) -> Result<(), Asn1Error> {
        let value = oid.encode_value()?;
        self.write_tag(&Tag::universal(UniversalTag::Oid as u32));
        self.write_length(value.len());
        self.buf.extend_from_slice(&value);
        Ok(())
    }

    /// Encode a UTF8String.
    pub fn encode_utf8_string(&mut self, s: &str) {
        self.write_tag(&Tag::universal(UniversalTag::Utf8String as u32));
        self.write_length(s.len());
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Encode a PrintableString.
    pub fn encode_printable_string(&mut self, s: &str) {
        self.write_tag(&Tag::universal(UniversalTag::PrintableString as u32));
        self.write_length(s.len());
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Encode a UTCTime string (e.g. "230101120000Z").
    pub fn encode_utc_time(&mut self, s: &str) {
        self.write_tag(&Tag::universal(UniversalTag::UtcTime as u32));
        self.write_length(s.len());
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Start a SEQUENCE, encode children, close.
    pub fn encode_sequence(&mut self, children: &[Tlv]) {
        let tag = Tag::sequence();
        self.write_tag(&tag);
        let mut child_enc = DerEncoder::new();
        for child in children {
            child_enc.encode_tlv(child);
        }
        let child_bytes = child_enc.finish();
        self.write_length(child_bytes.len());
        self.buf.extend_from_slice(&child_bytes);
    }

    /// Consume and return encoded bytes.
    pub fn finish(self) -> Vec<u8> {
        self.buf
    }
}

fn integer_to_der_bytes(val: i64) -> Vec<u8> {
    if val == 0 {
        return vec![0];
    }
    let be = val.to_be_bytes();
    // Find first significant byte
    let mut start = 0;
    if val > 0 {
        while start < 7 && be[start] == 0 {
            start += 1;
        }
        // If high bit is set, prepend 0x00
        if be[start] & 0x80 != 0 {
            let mut result = vec![0x00];
            result.extend_from_slice(&be[start..]);
            return result;
        }
    } else {
        while start < 7 && be[start] == 0xFF && be[start + 1] & 0x80 != 0 {
            start += 1;
        }
    }
    be[start..].to_vec()
}

// ── DER Decoder ─────────────────────────────────────────────────

/// DER decoder.
pub struct DerDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    depth: usize,
}

impl<'a> DerDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0, depth: 0 }
    }

    fn read_u8(&mut self) -> Result<u8, Asn1Error> {
        if self.pos >= self.data.len() {
            return Err(Asn1Error::UnexpectedEof);
        }
        let b = self.data[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_tag(&mut self) -> Result<Tag, Asn1Error> {
        let byte = self.read_u8()?;
        let class = match byte >> 6 {
            0 => TagClass::Universal,
            1 => TagClass::Application,
            2 => TagClass::ContextSpecific,
            3 => TagClass::Private,
            _ => unreachable!(),
        };
        let constructed = byte & 0x20 != 0;
        let number = if byte & 0x1F == 0x1F {
            // Long form tag
            let mut num: u32 = 0;
            loop {
                let b = self.read_u8()?;
                num = num.checked_shl(7).ok_or(Asn1Error::InvalidTag(byte))?;
                num |= (b & 0x7F) as u32;
                if b & 0x80 == 0 {
                    break;
                }
            }
            num
        } else {
            (byte & 0x1F) as u32
        };
        Ok(Tag { class, constructed, number })
    }

    fn read_length(&mut self) -> Result<usize, Asn1Error> {
        let byte = self.read_u8()?;
        if byte & 0x80 == 0 {
            Ok(byte as usize)
        } else if byte == 0x80 {
            Err(Asn1Error::IndefiniteLengthNotDer)
        } else {
            let num_bytes = (byte & 0x7F) as usize;
            if num_bytes > 4 {
                return Err(Asn1Error::InvalidLength);
            }
            let mut len: usize = 0;
            for _ in 0..num_bytes {
                len = len.checked_shl(8).ok_or(Asn1Error::InvalidLength)?;
                len |= self.read_u8()? as usize;
            }
            Ok(len)
        }
    }

    /// Decode the next TLV.
    pub fn decode_tlv(&mut self) -> Result<Tlv, Asn1Error> {
        if self.depth > 64 {
            return Err(Asn1Error::NestingTooDeep);
        }
        let tag = self.read_tag()?;
        let len = self.read_length()?;
        if self.pos + len > self.data.len() {
            return Err(Asn1Error::ContentOverflow);
        }

        let value = if tag.constructed {
            self.depth += 1;
            let end = self.pos + len;
            let mut children = Vec::new();
            while self.pos < end {
                children.push(self.decode_tlv()?);
            }
            self.depth -= 1;
            TlvValue::Constructed(children)
        } else {
            let bytes = self.data[self.pos..self.pos + len].to_vec();
            self.pos += len;
            TlvValue::Primitive(bytes)
        };

        Ok(Tlv { tag, value })
    }

    pub fn position(&self) -> usize {
        self.pos
    }
}

// ── Convenience decoders ────────────────────────────────────────

/// Decode a DER-encoded BOOLEAN.
pub fn decode_boolean(tlv: &Tlv) -> Result<bool, Asn1Error> {
    match &tlv.value {
        TlvValue::Primitive(data) => {
            if data.len() != 1 {
                return Err(Asn1Error::InvalidBooleanLength);
            }
            Ok(data[0] != 0)
        }
        _ => Err(Asn1Error::InvalidBooleanLength),
    }
}

/// Decode a DER-encoded INTEGER to i64.
pub fn decode_integer(tlv: &Tlv) -> Result<i64, Asn1Error> {
    match &tlv.value {
        TlvValue::Primitive(data) => {
            if data.is_empty() || data.len() > 8 {
                return Err(Asn1Error::InvalidInteger);
            }
            let negative = data[0] & 0x80 != 0;
            let mut result: i64 = if negative { -1 } else { 0 };
            for byte in data {
                result = result.checked_shl(8).ok_or(Asn1Error::InvalidInteger)?;
                result |= *byte as i64;
            }
            Ok(result)
        }
        _ => Err(Asn1Error::InvalidInteger),
    }
}

/// Decode an OID from a TLV.
pub fn decode_oid(tlv: &Tlv) -> Result<Oid, Asn1Error> {
    match &tlv.value {
        TlvValue::Primitive(data) => Oid::decode_value(data),
        _ => Err(Asn1Error::InvalidOid),
    }
}

// ── Public API ──────────────────────────────────────────────────

/// Encode a TLV to DER bytes.
pub fn encode(tlv: &Tlv) -> Vec<u8> {
    let mut enc = DerEncoder::new();
    enc.encode_tlv(tlv);
    enc.finish()
}

/// Decode DER bytes to a TLV.
pub fn decode(data: &[u8]) -> Result<Tlv, Asn1Error> {
    let mut dec = DerDecoder::new(data);
    dec.decode_tlv()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_roundtrip() {
        let mut enc = DerEncoder::new();
        enc.encode_boolean(true);
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        assert_eq!(decode_boolean(&tlv).unwrap(), true);

        let mut enc = DerEncoder::new();
        enc.encode_boolean(false);
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        assert_eq!(decode_boolean(&tlv).unwrap(), false);
    }

    #[test]
    fn integer_roundtrip() {
        for val in [0i64, 1, -1, 127, 128, -128, -129, 256, 32767, -32768, 1000000] {
            let mut enc = DerEncoder::new();
            enc.encode_integer(val);
            let bytes = enc.finish();
            let tlv = decode(&bytes).unwrap();
            assert_eq!(decode_integer(&tlv).unwrap(), val, "failed for {val}");
        }
    }

    #[test]
    fn null_encoding() {
        let mut enc = DerEncoder::new();
        enc.encode_null();
        let bytes = enc.finish();
        assert_eq!(bytes, &[0x05, 0x00]);
        let tlv = decode(&bytes).unwrap();
        assert_eq!(tlv.tag.number, UniversalTag::Null as u32);
    }

    #[test]
    fn octet_string_roundtrip() {
        let data = b"hello world";
        let mut enc = DerEncoder::new();
        enc.encode_octet_string(data);
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        match &tlv.value {
            TlvValue::Primitive(v) => assert_eq!(v.as_slice(), data),
            _ => panic!("expected primitive"),
        }
    }

    #[test]
    fn bit_string_roundtrip() {
        let data = &[0xDE, 0xAD, 0xBE, 0xEF];
        let mut enc = DerEncoder::new();
        enc.encode_bit_string(data, 0);
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        match &tlv.value {
            TlvValue::Primitive(v) => {
                assert_eq!(v[0], 0); // unused bits
                assert_eq!(&v[1..], data);
            }
            _ => panic!("expected primitive"),
        }
    }

    #[test]
    fn oid_roundtrip() {
        let oid = Oid::from_str("1.2.840.113549.1.1.1").unwrap();
        let mut enc = DerEncoder::new();
        enc.encode_oid(&oid).unwrap();
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        let decoded_oid = decode_oid(&tlv).unwrap();
        assert_eq!(decoded_oid.to_dotted_string(), "1.2.840.113549.1.1.1");
    }

    #[test]
    fn oid_simple() {
        let oid = Oid::from_str("2.5.4.3").unwrap();
        assert_eq!(oid.to_dotted_string(), "2.5.4.3");
        let value = oid.encode_value().unwrap();
        let decoded = Oid::decode_value(&value).unwrap();
        assert_eq!(decoded.components, oid.components);
    }

    #[test]
    fn utf8_string_roundtrip() {
        let mut enc = DerEncoder::new();
        enc.encode_utf8_string("hello");
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        assert_eq!(tlv.tag.number, UniversalTag::Utf8String as u32);
        match &tlv.value {
            TlvValue::Primitive(v) => assert_eq!(std::str::from_utf8(v).unwrap(), "hello"),
            _ => panic!("expected primitive"),
        }
    }

    #[test]
    fn sequence_constructed() {
        let bool_tlv = {
            let mut enc = DerEncoder::new();
            enc.encode_boolean(true);
            decode(&enc.finish()).unwrap()
        };
        let int_tlv = {
            let mut enc = DerEncoder::new();
            enc.encode_integer(42);
            decode(&enc.finish()).unwrap()
        };

        let mut enc = DerEncoder::new();
        enc.encode_sequence(&[bool_tlv.clone(), int_tlv.clone()]);
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();

        assert!(tlv.tag.constructed);
        assert_eq!(tlv.tag.number, UniversalTag::Sequence as u32);
        match &tlv.value {
            TlvValue::Constructed(children) => {
                assert_eq!(children.len(), 2);
                assert_eq!(decode_boolean(&children[0]).unwrap(), true);
                assert_eq!(decode_integer(&children[1]).unwrap(), 42);
            }
            _ => panic!("expected constructed"),
        }
    }

    #[test]
    fn utc_time() {
        let mut enc = DerEncoder::new();
        enc.encode_utc_time("230101120000Z");
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        assert_eq!(tlv.tag.number, UniversalTag::UtcTime as u32);
        match &tlv.value {
            TlvValue::Primitive(v) => assert_eq!(std::str::from_utf8(v).unwrap(), "230101120000Z"),
            _ => panic!("expected primitive"),
        }
    }

    #[test]
    fn printable_string() {
        let mut enc = DerEncoder::new();
        enc.encode_printable_string("US");
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        assert_eq!(tlv.tag.number, UniversalTag::PrintableString as u32);
    }

    #[test]
    fn long_length() {
        // Octet string with > 127 bytes
        let data = vec![0xAB; 200];
        let mut enc = DerEncoder::new();
        enc.encode_octet_string(&data);
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        match &tlv.value {
            TlvValue::Primitive(v) => assert_eq!(v.len(), 200),
            _ => panic!("expected primitive"),
        }
    }

    #[test]
    fn nested_sequence() {
        let inner_bool = {
            let mut e = DerEncoder::new();
            e.encode_boolean(false);
            decode(&e.finish()).unwrap()
        };
        let inner_seq_tlv = Tlv {
            tag: Tag::sequence(),
            value: TlvValue::Constructed(vec![inner_bool]),
        };
        let mut enc = DerEncoder::new();
        enc.encode_sequence(&[inner_seq_tlv]);
        let bytes = enc.finish();
        let tlv = decode(&bytes).unwrap();
        match &tlv.value {
            TlvValue::Constructed(children) => {
                assert_eq!(children.len(), 1);
                assert!(children[0].tag.constructed);
            }
            _ => panic!("expected constructed"),
        }
    }
}
