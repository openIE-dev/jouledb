use joule_db_amorphic::{AmorphicStore, Value};
use std::collections::HashMap;
fn main() {
    let n: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(200_000);
    let mut s = AmorphicStore::new();
    let t = std::time::Instant::now();
    for chunk in (0..n).collect::<Vec<_>>().chunks(8192) {
        let rows: Vec<_> = chunk.iter().map(|&i| {
            let mut m = HashMap::new();
            m.insert("__table__".to_string(), Value::String("hits".into()));
            m.insert("id".to_string(), Value::Int(i as i64));
            m.insert("user_id".to_string(), Value::Int((i * 7919 % 50000) as i64));
            m.insert("region".to_string(), Value::Int((i % 50) as i64));
            m.insert("url".to_string(), Value::String(format!("https://example.com/p/{}", i % 1000)));
            m.insert("duration".to_string(), Value::Float(i as f64 * 0.37));
            m.insert("ts".to_string(), Value::Int(1_700_000_000 + i as i64));
            s.prepare_fields_ingest(m).unwrap()
        }).collect();
        s.apply_prepared_ingest_batch(rows).unwrap();
    }
    println!("apply {} rows: {:?}", n, t.elapsed());
    if std::env::var_os("SKIP_CATCHUP").is_some() { return; }
    let t = std::time::Instant::now();
    s.catch_up_holograms(usize::MAX);
    println!("catch-up: {:?}", t.elapsed());
}
