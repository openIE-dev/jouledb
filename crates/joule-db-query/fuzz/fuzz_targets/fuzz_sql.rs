#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        // Limit input size to avoid pathological cases
        if input.len() > 10_000 {
            return;
        }
        let mut parser = joule_db_query::sql::SqlParser::new();
        // We don't care about the result — we care that it doesn't panic or crash
        let _ = parser.parse(input);
    }
});
