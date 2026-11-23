#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(input) = std::str::from_utf8(data) {
        if input.len() > 10_000 {
            return;
        }
        let mut parser = joule_db_query::cypher::CypherParser::new();
        let _ = parser.parse(input);
    }
});
