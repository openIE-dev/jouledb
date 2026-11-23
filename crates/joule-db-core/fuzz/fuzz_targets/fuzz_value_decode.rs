//! Fuzz Value::decode with arbitrary bytes.
//!
//! Verifies:
//! 1. decode never panics for any byte sequence
//! 2. If decode succeeds, re-encoding and re-decoding produces the same value
//!
//! Run: `cargo fuzz run fuzz_value_decode -j$(nproc)`

#![no_main]

use libfuzzer_sys::fuzz_target;
use joule_db_core::Value;

fuzz_target!(|data: &[u8]| {
    // Property 1: decode must never panic — only Ok or Err
    let result = Value::decode(data);

    // Property 2: if decode succeeds, roundtrip must be stable
    if let Ok(val) = result {
        let encoded = val.encode();
        let decoded2 = Value::decode(&encoded)
            .expect("re-decode of encoded value must succeed");
        assert!(
            val == decoded2,
            "roundtrip mismatch: {:?} != {:?}",
            val, decoded2
        );

        // Property 3: encode is deterministic
        let encoded2 = decoded2.encode();
        assert_eq!(
            encoded, encoded2,
            "encode not deterministic"
        );
    }
});
