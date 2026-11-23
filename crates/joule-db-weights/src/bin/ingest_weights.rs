//! Ingest LLM weights from GGUF into JouleDB.
//!
//! Usage:
//!   cargo run -p joule-db-weights --release --bin ingest_weights -- \
//!     --model-id nemotron-3-super-120b \
//!     --gguf /Volumes/macos_4TB_external/neuronexus_ai/model_library/nemotron-3-super-Q3/nvidia_Nemotron-3-Super-120B-A12B-Q3_K_S \
//!     --db /Volumes/macos_4TB_external/neuronexus_ai/jouledb/nemotron-3-super-120b.jdb

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_id = String::new();
    let mut gguf_path = String::new();
    let mut db_path = String::new();
    let mut blob = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model-id" => {
                i += 1;
                model_id = args[i].clone();
            }
            "--gguf" => {
                i += 1;
                gguf_path = args[i].clone();
            }
            "--db" => {
                i += 1;
                db_path = args[i].clone();
            }
            // Contiguous zero-copy weight blob (<db>.wblob) instead of paged
            // extents — instant mmap load, no buffer-pool copy.
            "--blob" => {
                blob = true;
            }
            "--help" | "-h" => {
                eprintln!("Usage: ingest_weights --model-id <id> --gguf <path> --db <path>");
                eprintln!();
                eprintln!("  --model-id  Model identifier (e.g., nemotron-3-super-120b)");
                eprintln!("  --gguf      Path to .gguf file or directory with sharded .gguf files");
                eprintln!("  --db        Path to JouleDB database file (created if not exists)");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    if model_id.is_empty() || gguf_path.is_empty() || db_path.is_empty() {
        eprintln!("Error: --model-id, --gguf, and --db are required");
        eprintln!("Run with --help for usage");
        std::process::exit(1);
    }

    eprintln!("=== JouleDB Weight Ingestor ===");
    eprintln!();
    eprintln!("  Model ID: {}", model_id);
    eprintln!("  GGUF:     {}", gguf_path);
    eprintln!("  DB:       {}", db_path);
    eprintln!();

    // Open or create the JouleDB database
    eprintln!("Opening database...");
    let db = match joule_db_weights::WeightDB::open(&db_path) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("ERROR: Failed to open database: {:?}", e);
            std::process::exit(1);
        }
    };
    eprintln!("  Database ready: {}", db_path);
    eprintln!();

    // Ingest
    eprintln!(
        "Ingesting GGUF weights... (mode: {})",
        if blob {
            "contiguous blob"
        } else {
            "paged extents"
        }
    );
    let gguf = std::path::Path::new(&gguf_path);
    let result = if blob {
        joule_db_weights::ingest::ingest_gguf_blob(&db, &model_id, gguf)
    } else {
        joule_db_weights::ingest::ingest_gguf(&db, &model_id, gguf)
    };
    match result {
        Ok(stats) => {
            eprintln!();
            eprintln!("=== Ingest Complete ===");
            eprintln!("  Tensors:    {}", stats.tensors_ingested);
            eprintln!("  Total:      {:.2} GB", stats.bytes_ingested as f64 / 1e9);
            eprintln!(
                "  Extents:    {} (>256KB, contiguous pages)",
                stats.extent_tensors
            );
            eprintln!("  Inline:     {} (<256KB)", stats.inline_tensors);
            eprintln!("  Metadata:   {} entries", stats.metadata_entries);
            eprintln!("  GGUF KVs:   {}", stats.gguf_metadata_keys);
            eprintln!("  Time:       {:.1}s", stats.elapsed_ms as f64 / 1000.0);
            eprintln!(
                "  Throughput: {:.1} MB/s",
                stats.bytes_ingested as f64 / 1e6 / (stats.elapsed_ms as f64 / 1000.0)
            );

            // Report database file size
            if let Ok(meta) = std::fs::metadata(&db_path) {
                eprintln!("  DB file:    {:.2} GB", meta.len() as f64 / 1e9);
            }
        }
        Err(e) => {
            eprintln!("ERROR: Ingest failed: {:?}", e);
            std::process::exit(1);
        }
    }
}
