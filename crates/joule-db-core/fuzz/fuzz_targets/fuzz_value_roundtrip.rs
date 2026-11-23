//! Fuzz Value codec with structured inputs.
//!
//! Uses libfuzzer's structured input to generate plausible Value-like
//! byte streams (valid tag prefix + random payload), exercising more
//! code paths in the decoder than pure random bytes.
//!
//! Run: `cargo fuzz run fuzz_value_roundtrip -j$(nproc)`

#![no_main]

use libfuzzer_sys::fuzz_target;
use joule_db_core::Value;

fuzz_target!(|data: &[u8]| {
    if data.is_empty() {
        return;
    }

    // Use the first byte to select a type tag (0-10), then prepend it
    // to the remaining bytes as a "plausible" encoded value
    let tag = data[0] % 11;
    let mut input = vec![tag];
    input.extend_from_slice(&data[1..]);

    // Must never panic
    let result = Value::decode(&input);

    if let Ok(val) = result {
        // Roundtrip: encode → decode → compare
        let encoded = val.encode();
        let decoded = Value::decode(&encoded)
            .expect("re-decode must succeed");
        assert!(val == decoded, "roundtrip failed for tag {tag}");
    }

    // Also test with raw bytes (no tag manipulation)
    let _ = Value::decode(data);

    // Test truncation: progressively shorter slices must not panic
    for len in (0..data.len().min(32)).rev() {
        let _ = Value::decode(&data[..len]);
    }
});
