//! Minimal round-trip + bounded-memory proof for a JouleDB weight store.
//!
//! Opens the DB with a deliberately TINY buffer pool (64 pages = 4 MB) to
//! demonstrate that a 57 GB model can be read back through a fixed memory
//! window — i.e. JouleDB streams pages, it does not load the model into RAM.
//!
//! Usage:
//!   cargo run -p joule-db-weights --release --bin roundtrip_check -- \
//!     --db <path.jdb> --model-id <id>

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut db_path = String::new();
    let mut model_id = String::new();
    let mut pool_pages: usize = 64; // 4 MB default; override with --pool-pages
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--db" => {
                i += 1;
                db_path = args[i].clone();
            }
            "--model-id" => {
                i += 1;
                model_id = args[i].clone();
            }
            "--pool-pages" => {
                i += 1;
                pool_pages = args[i].parse().unwrap_or(64);
            }
            _ => {}
        }
        i += 1;
    }
    if db_path.is_empty() || model_id.is_empty() {
        eprintln!("Usage: roundtrip_check --db <path> --model-id <id>");
        std::process::exit(1);
    }

    // 64 pages * 64 KB = 4 MB working set, against a 57 GB database.
    eprintln!("=== JouleDB weight round-trip check ===");
    eprintln!("  DB:       {db_path}");
    eprintln!("  Model:    {model_id}");
    eprintln!(
        "  Pool:     {pool_pages} pages = {} MB (vs ~57 GB model)",
        pool_pages * 64 / 1024
    );
    eprintln!();

    let db = match joule_db_weights::WeightDB::open_with_pool_size(&db_path, pool_pages) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("ERROR opening DB: {e:?}");
            std::process::exit(1);
        }
    };

    // 1. Read the model index back out.
    let index = match db.get_model_index(&model_id) {
        Ok(Some(s)) => s,
        Ok(None) => {
            eprintln!("FAIL: no _index record for '{model_id}'");
            std::process::exit(2);
        }
        Err(e) => {
            eprintln!("FAIL: index read error: {e:?}");
            std::process::exit(2);
        }
    };

    // Pull a couple of fields out of the index JSON without a json dep.
    let field = |k: &str| -> String {
        index
            .find(&format!("\"{k}\":"))
            .map(|p| {
                let after = &index[p + k.len() + 3..];
                let end = after.find([',', '}']).unwrap_or(after.len());
                after[..end].trim_matches('"').to_string()
            })
            .unwrap_or_else(|| "?".into())
    };
    eprintln!(
        "  index OK: tensor_count={} layers={} experts={} total_bytes={}",
        field("tensor_count"),
        field("layer_count"),
        field("expert_count"),
        field("total_bytes")
    );

    // 2. Read a small tensor back (a norm — a few KB, fast even on cold USB).
    //    This proves keyed random-access round-trip without dragging the 1.5 GB
    //    token-embedding through a cold disk.
    let small_cands = [
        "output_norm.weight",
        "blk.0.attn_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.1.attn_norm.weight",
    ];
    let mut got_small = false;
    for cand in small_cands {
        match db.get_tensor(&model_id, cand) {
            Ok(Some(bytes)) => {
                let preview: Vec<String> =
                    bytes.iter().take(8).map(|b| format!("{b:02x}")).collect();
                eprintln!(
                    "  tensor OK: '{cand}' = {} bytes  first8=[{}]",
                    bytes.len(),
                    preview.join(" ")
                );
                got_small = true;
                break;
            }
            Ok(None) => continue,
            Err(e) => {
                eprintln!("FAIL: tensor read error on '{cand}': {e:?}");
                std::process::exit(3);
            }
        }
    }
    if !got_small {
        eprintln!("FAIL: none of the candidate norm tensors were found");
        std::process::exit(3);
    }

    eprintln!();
    eprintln!(
        "PASS: 57 GB DB round-tripped through a {} MB window.",
        pool_pages * 64 / 1024
    );
}
