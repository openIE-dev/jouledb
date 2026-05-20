//! Property-based tests for JouleDB Core value codec.
//!
//! Verifies algebraic invariants of the Value encode/decode pipeline:
//! - Roundtrip identity (∀v: decode(encode(v)) == v)
//! - Encoded size consistency
//! - Error handling on malformed input
//! - Tag uniqueness and completeness

use proptest::prelude::*;
use std::collections::BTreeMap;

use crate::types::Value;

// ═══════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a leaf (non-recursive) Value.
fn arb_leaf_value() -> impl Strategy<Value = Value> {
    prop_oneof![
        Just(Value::Null),
        any::<bool>().prop_map(Value::Bool),
        any::<i64>().prop_map(Value::Int),
        // Filter NaN since NaN != NaN breaks roundtrip equality
        any::<f64>()
            .prop_filter("no NaN", |f| !f.is_nan())
            .prop_map(Value::Float),
        "[ -~]{0,64}".prop_map(|s| Value::String(s)),
        prop::collection::vec(any::<u8>(), 0..32).prop_map(Value::Bytes),
        any::<i64>().prop_map(Value::Timestamp),
        prop::collection::vec(
            any::<f32>().prop_filter("no NaN", |f| !f.is_nan()),
            0..16
        )
        .prop_map(Value::Vector),
    ]
}

/// Generate a Value with one level of nesting (arrays/maps of leaves).
fn arb_value() -> impl Strategy<Value = Value> {
    arb_leaf_value().prop_recursive(2, 32, 8, |inner| {
        prop_oneof![
            prop::collection::vec(inner.clone(), 0..8).prop_map(Value::Array),
            prop::collection::vec(("[a-z]{1,8}", inner), 0..8)
                .prop_map(|pairs| Value::Map(pairs.into_iter().collect::<BTreeMap<_, _>>())),
        ]
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════

proptest! {
    /// P1: encode→decode roundtrip for all leaf types.
    #[test]
    fn prop_leaf_roundtrip(v in arb_leaf_value()) {
        let encoded = v.encode();
        let decoded = Value::decode(&encoded).unwrap();
        prop_assert!(v == decoded, "roundtrip failed: {v:?} != {decoded:?}");
    }

    /// P2: encode→decode roundtrip for nested values.
    #[test]
    fn prop_nested_roundtrip(v in arb_value()) {
        let encoded = v.encode();
        let decoded = Value::decode(&encoded).unwrap();
        prop_assert!(v == decoded, "nested roundtrip failed");
    }

    /// P3: encode is deterministic (same value → same bytes).
    #[test]
    fn prop_encode_deterministic(v in arb_value()) {
        let enc1 = v.encode();
        let enc2 = v.encode();
        prop_assert!(enc1 == enc2, "encode not deterministic");
    }

    /// P4: encoded bytes are never empty (at minimum: 1 tag byte).
    #[test]
    fn prop_encode_nonempty(v in arb_value()) {
        let encoded = v.encode();
        prop_assert!(!encoded.is_empty(), "encode produced empty bytes");
    }

    /// P5: first byte is always a valid tag (0..=10).
    #[test]
    fn prop_first_byte_valid_tag(v in arb_value()) {
        let encoded = v.encode();
        prop_assert!(encoded[0] <= 10, "invalid tag: {}", encoded[0]);
    }

    /// P6: decode rejects unknown tags.
    #[test]
    fn prop_unknown_tag_rejected(tag in 11u8..=255) {
        let result = Value::decode(&[tag]);
        prop_assert!(result.is_err(), "unknown tag {tag} should fail");
    }

    /// P7: decode rejects empty input.
    #[test]
    fn prop_empty_input_rejected(_ in Just(())) {
        let result = Value::decode(&[]);
        prop_assert!(result.is_err(), "empty input should fail");
    }

    /// P8: decode rejects truncated input.
    #[test]
    fn prop_truncated_input_rejected(v in arb_value()) {
        let encoded = v.encode();
        if encoded.len() > 1 {
            let truncated = &encoded[..encoded.len() - 1];
            // Truncated input should either fail or decode to something different
            // (it might succeed for Null which is 1 byte, but we skip those)
            if matches!(v, Value::Null | Value::Bool(_)) {
                // These are 1 byte, truncation removes the tag entirely
            } else {
                let result = Value::decode(truncated);
                prop_assert!(result.is_err(), "truncated decode should fail for {v:?}");
            }
        }
    }

    /// P9: Null encodes to exactly 1 byte.
    #[test]
    fn prop_null_size(_ in Just(())) {
        let encoded = Value::Null.encode();
        prop_assert!(encoded.len() == 1, "Null should be 1 byte");
    }

    /// P10: Bool encodes to exactly 1 byte.
    #[test]
    fn prop_bool_size(b in any::<bool>()) {
        let encoded = Value::Bool(b).encode();
        prop_assert!(encoded.len() == 1, "Bool should be 1 byte");
    }

    /// P11: Int encodes to exactly 9 bytes (1 tag + 8 data).
    #[test]
    fn prop_int_size(i in any::<i64>()) {
        let encoded = Value::Int(i).encode();
        prop_assert!(encoded.len() == 9, "Int should be 9 bytes, got {}", encoded.len());
    }

    /// P12: Float encodes to exactly 9 bytes (1 tag + 8 data).
    #[test]
    fn prop_float_size(f in any::<f64>().prop_filter("no NaN", |f| !f.is_nan())) {
        let encoded = Value::Float(f).encode();
        prop_assert!(encoded.len() == 9, "Float should be 9 bytes");
    }

    /// P13: String encoding size = 1 tag + 4 len + N bytes.
    #[test]
    fn prop_string_size(s in "[ -~]{0,64}") {
        let encoded = Value::String(s.clone()).encode();
        prop_assert!(
            encoded.len() == 1 + 4 + s.len(),
            "String size mismatch: {} vs {}", encoded.len(), 1 + 4 + s.len()
        );
    }

    /// P14: type_name returns correct name for each variant.
    #[test]
    fn prop_type_name_consistent(v in arb_leaf_value()) {
        let name = v.type_name();
        let expected = match &v {
            Value::Null => "null",
            Value::Bool(_) => "bool",
            Value::Int(_) => "int",
            Value::Float(_) => "float",
            Value::String(_) => "string",
            Value::Bytes(_) => "bytes",
            Value::Array(_) => "array",
            Value::Map(_) => "map",
            Value::Timestamp(_) => "timestamp",
            Value::Vector(_) => "vector",
            Value::Spatial3d(s) => s.type_name(),
        };
        prop_assert!(name == expected, "type_name mismatch: {name} vs {expected}");
    }

    /// P15: encode_into appends to buffer (preserves existing data).
    #[test]
    fn prop_encode_into_appends(v in arb_leaf_value(), prefix in prop::collection::vec(any::<u8>(), 0..16)) {
        let mut buf = prefix.clone();
        v.encode_into(&mut buf);
        prop_assert!(buf.starts_with(&prefix), "encode_into overwrote prefix");
        prop_assert!(buf.len() > prefix.len(), "encode_into added nothing");
    }

    /// P16: encode idempotent over double-encode-decode.
    #[test]
    fn prop_double_roundtrip(v in arb_value()) {
        let enc1 = v.encode();
        let dec1 = Value::decode(&enc1).unwrap();
        let enc2 = dec1.encode();
        prop_assert!(enc1 == enc2, "double roundtrip produced different bytes");
    }

    /// P17: random bytes don't panic on decode (may return Ok or Err).
    #[test]
    fn prop_random_bytes_no_panic(data in prop::collection::vec(any::<u8>(), 0..128)) {
        let _ = Value::decode(&data);
    }

    /// P18: Vector encoding size = 1 tag + 4 len + 4*N floats.
    #[test]
    fn prop_vector_size(v in prop::collection::vec(any::<f32>().prop_filter("no NaN", |f| !f.is_nan()), 0..16)) {
        let encoded = Value::Vector(v.clone()).encode();
        prop_assert!(
            encoded.len() == 1 + 4 + v.len() * 4,
            "Vector size mismatch"
        );
    }

    /// P19: Bytes encoding size = 1 tag + 4 len + N bytes.
    #[test]
    fn prop_bytes_size(b in prop::collection::vec(any::<u8>(), 0..64)) {
        let encoded = Value::Bytes(b.clone()).encode();
        prop_assert!(
            encoded.len() == 1 + 4 + b.len(),
            "Bytes size mismatch"
        );
    }

    /// P20: Timestamp roundtrip preserves exact value.
    #[test]
    fn prop_timestamp_exact(ts in any::<i64>()) {
        let v = Value::Timestamp(ts);
        let decoded = Value::decode(&v.encode()).unwrap();
        prop_assert!(
            decoded.as_int() == Some(ts) || matches!(decoded, Value::Timestamp(t) if t == ts),
            "timestamp not preserved"
        );
    }
}
