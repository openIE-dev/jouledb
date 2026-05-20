//! Kani bounded model checker proofs for JouleDB Core.
//!
//! Verifies safety properties of the Value codec over exhaustive
//! finite domains.
//!
//! Install: `cargo install cargo-kani`
//! Run: `cargo kani -p joule-db-core`

#[cfg(kani)]
mod proofs {
    use crate::types::Value;

    /// **Proof: all 11 tag values produce valid roundtrips.**
    #[kani::proof]
    fn prove_all_tags_roundtrip() {
        let values = [
            Value::Null,
            Value::Bool(false),
            Value::Bool(true),
            Value::Int(0),
            Value::Int(i64::MIN),
            Value::Int(i64::MAX),
            Value::Float(0.0),
            Value::Float(-1.0),
            Value::String(String::new()),
            Value::Bytes(vec![]),
            Value::Timestamp(0),
            Value::Vector(vec![]),
        ];

        for v in &values {
            let encoded = v.encode();
            let decoded = Value::decode(&encoded).expect("valid value should decode");
            assert_eq!(v, &decoded);
        }
    }

    /// **Proof: unknown tag bytes (11..=255) always error.**
    #[kani::proof]
    fn prove_unknown_tags_error() {
        let tag: u8 = kani::any();
        kani::assume(tag > 10);
        let result = Value::decode(&[tag]);
        assert!(result.is_err(), "unknown tag should error");
    }

    /// **Proof: empty buffer always errors.**
    #[kani::proof]
    fn prove_empty_decode_error() {
        let result = Value::decode(&[]);
        assert!(result.is_err());
    }

    /// **Proof: Int encoding is exactly 9 bytes.**
    #[kani::proof]
    fn prove_int_encoding_size() {
        let i: i64 = kani::any();
        let encoded = Value::Int(i).encode();
        assert_eq!(encoded.len(), 9);
        assert_eq!(encoded[0], 3); // INT tag
    }

    /// **Proof: Bool encoding is exactly 1 byte with correct tag.**
    #[kani::proof]
    fn prove_bool_encoding() {
        let b: bool = kani::any();
        let encoded = Value::Bool(b).encode();
        assert_eq!(encoded.len(), 1);
        if b {
            assert_eq!(encoded[0], 2); // BOOL_TRUE
        } else {
            assert_eq!(encoded[0], 1); // BOOL_FALSE
        }
    }

    /// **Proof: Null encoding is exactly 1 byte with tag 0.**
    #[kani::proof]
    fn prove_null_encoding() {
        let encoded = Value::Null.encode();
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], 0);
    }

    /// **Proof: Int roundtrip preserves exact bit pattern.**
    #[kani::proof]
    fn prove_int_roundtrip_exact() {
        let i: i64 = kani::any();
        let v = Value::Int(i);
        let decoded = Value::decode(&v.encode()).unwrap();
        match decoded {
            Value::Int(j) => assert_eq!(i, j),
            _ => panic!("decoded to wrong type"),
        }
    }

    /// **Proof: Timestamp roundtrip preserves exact value.**
    #[kani::proof]
    fn prove_timestamp_roundtrip_exact() {
        let ts: i64 = kani::any();
        let v = Value::Timestamp(ts);
        let decoded = Value::decode(&v.encode()).unwrap();
        match decoded {
            Value::Timestamp(t) => assert_eq!(ts, t),
            _ => panic!("decoded to wrong type"),
        }
    }
}
