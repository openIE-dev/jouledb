//! Minimal protobuf (proto3) writer. Only what the Core ML spec needs:
//! varints, length-delimited fields, packed repeated varints, map entries.

#[derive(Default, Clone)]
pub(crate) struct Pb(pub(crate) Vec<u8>);

fn put_varint(out: &mut Vec<u8>, mut v: u64) {
    loop {
        let byte = (v & 0x7f) as u8;
        v >>= 7;
        if v == 0 {
            out.push(byte);
            return;
        }
        out.push(byte | 0x80);
    }
}

impl Pb {
    pub(crate) fn new() -> Self {
        Self(Vec::new())
    }

    fn key(&mut self, field: u32, wire: u8) {
        put_varint(&mut self.0, (u64::from(field) << 3) | u64::from(wire));
    }

    /// int32/int64/uint64/bool/enum field (wire type 0).
    pub(crate) fn varint(mut self, field: u32, v: i64) -> Self {
        self.key(field, 0);
        put_varint(&mut self.0, v as u64);
        self
    }

    pub(crate) fn bytes(mut self, field: u32, data: &[u8]) -> Self {
        self.key(field, 2);
        put_varint(&mut self.0, data.len() as u64);
        self.0.extend_from_slice(data);
        self
    }

    pub(crate) fn string(self, field: u32, s: &str) -> Self {
        self.bytes(field, s.as_bytes())
    }

    pub(crate) fn msg(self, field: u32, m: Pb) -> Self {
        self.bytes(field, &m.0)
    }

    /// Packed repeated varints (proto3 default for repeated scalars).
    pub(crate) fn packed(self, field: u32, values: &[i64]) -> Self {
        let mut buf = Vec::new();
        for v in values {
            put_varint(&mut buf, *v as u64);
        }
        self.bytes(field, &buf)
    }

    /// One entry of a `map<string, Message>` field.
    pub(crate) fn map_entry(self, field: u32, key: &str, value: Pb) -> Self {
        self.msg(field, Pb::new().string(1, key).msg(2, value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_encoding_matches_protobuf_spec() {
        // field 1, value 150 -> 08 96 01 (protobuf docs example)
        assert_eq!(Pb::new().varint(1, 150).0, vec![0x08, 0x96, 0x01]);
        // field 2, "testing" -> 12 07 74 65 73 74 69 6e 67
        assert_eq!(
            Pb::new().string(2, "testing").0,
            vec![0x12, 0x07, b't', b'e', b's', b't', b'i', b'n', b'g']
        );
        // large field number 502 (mlProgram) with wire type 2
        let encoded = Pb::new().bytes(502, &[]).0;
        assert_eq!(encoded, vec![0xb2, 0x1f, 0x00]);
    }
}
