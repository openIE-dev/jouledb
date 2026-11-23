//! Analyze compressibility of quantized weight tensors using zstd.
//!
//! Reads tensors from JouleDB and measures:
//! - zstd compression ratio at multiple levels (1, 3, 6, 9)
//! - Per-component compression patterns
//! - Projected total storage with compression
//!
//! Usage:
//!   cargo run -p joule-db-weights --release --bin analyze_compression -- \
//!     --db /Volumes/macos_4TB_external/neuronexus_ai/jouledb/nemotron-3-super-120b.jdb \
//!     --model-id nemotron-3-super-120b

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut db_path = String::new();
    let mut model_id = String::new();

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
            _ => {}
        }
        i += 1;
    }

    if db_path.is_empty() || model_id.is_empty() {
        eprintln!("Usage: analyze_compression --db <path> --model-id <id>");
        std::process::exit(1);
    }

    eprintln!("=== Zstd Compression Analysis ===");
    eprintln!("  DB: {}", db_path);
    eprintln!("  Model: {}", model_id);
    eprintln!();

    let db = match joule_db_weights::WeightDB::open_with_pool_size(&db_path, 16384) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("ERROR: {:?}", e);
            std::process::exit(1);
        }
    };

    let index_json = db
        .get_model_index(&model_id)
        .expect("DB error")
        .expect("Model not found");

    // Collect tensor names
    let mut tensor_names: Vec<String> = Vec::new();
    if let Some(start) = index_json.find("\"tensor_names\":[") {
        let after = &index_json[start + 16..];
        if let Some(end) = after.find(']') {
            for name in after[..end].split(',') {
                let name = name.trim().trim_matches('"');
                if !name.is_empty() {
                    tensor_names.push(name.to_string());
                }
            }
        }
    }

    eprintln!("Tensors: {}", tensor_names.len());
    eprintln!();

    // Sample representative tensors — one of each component type from layer 1
    // Plus the globals (token_embd, output, output_norm)
    let mut samples: Vec<String> = Vec::new();
    let mut seen_components: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Global tensors first
    for name in &tensor_names {
        if !name.starts_with("blk.") {
            samples.push(name.clone());
        }
    }

    // One per component from early layers
    for name in &tensor_names {
        if !name.starts_with("blk.") {
            continue;
        }
        let parts: Vec<&str> = name.splitn(3, '.').collect();
        if parts.len() < 3 {
            continue;
        }
        let component = parts[2].to_string();
        if !seen_components.contains(&component) {
            seen_components.insert(component);
            samples.push(name.clone());
        }
    }

    eprintln!("Sampling {} representative tensors:", samples.len());
    for s in &samples {
        eprintln!("  {}", s);
    }
    eprintln!();

    // Compression analysis
    println!(
        "tensor,raw_bytes,zstd_1,zstd_3,zstd_6,zstd_9,ratio_1,ratio_3,ratio_6,ratio_9,quant_type"
    );

    let mut total_raw = 0u64;
    let mut total_z1 = 0u64;
    let mut total_z3 = 0u64;
    let mut total_z6 = 0u64;
    let mut total_z9 = 0u64;

    // Track per-component stats for projection
    let mut component_stats: std::collections::HashMap<String, (u64, u64, f64)> =
        std::collections::HashMap::new(); // component -> (count * size, count, ratio_z3)

    for name in &samples {
        let data = match db.get_tensor(&model_id, name) {
            Ok(Some(d)) => d,
            _ => {
                eprintln!("  {} — not found", name);
                continue;
            }
        };

        if data.is_empty() {
            continue;
        }

        let raw_size = data.len() as u64;

        // Get quant type from metadata
        let meta_json = db
            .get_tensor_meta(&model_id, name)
            .ok()
            .flatten()
            .unwrap_or_default();
        let quant_type =
            extract_json_string(&meta_json, "type_name").unwrap_or_else(|| "unknown".to_string());

        // Compress at multiple zstd levels
        let z1 = zstd_compress(&data, 1);
        let z3 = zstd_compress(&data, 3);
        let z6 = zstd_compress(&data, 6);
        let z9 = zstd_compress(&data, 9);

        let r1 = raw_size as f64 / z1 as f64;
        let r3 = raw_size as f64 / z3 as f64;
        let r6 = raw_size as f64 / z6 as f64;
        let r9 = raw_size as f64 / z9 as f64;

        println!(
            "{},{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{}",
            name, raw_size, z1, z3, z6, z9, r1, r3, r6, r9, quant_type
        );

        eprintln!(
            "  {} ({:.1} MB, {}) → z1:{:.2}x z3:{:.2}x z6:{:.2}x z9:{:.2}x",
            name,
            raw_size as f64 / 1e6,
            quant_type,
            r1,
            r3,
            r6,
            r9
        );

        total_raw += raw_size;
        total_z1 += z1;
        total_z3 += z3;
        total_z6 += z6;
        total_z9 += z9;

        // Track component for projection
        let component = if name.starts_with("blk.") {
            name.splitn(3, '.').nth(2).unwrap_or(name).to_string()
        } else {
            name.clone()
        };

        // Count how many tensors of this component exist across all layers
        let comp_count = tensor_names
            .iter()
            .filter(|n| {
                if let Some(c) = n.splitn(3, '.').nth(2) {
                    c == component
                } else {
                    *n == &component
                }
            })
            .count() as u64;

        component_stats.insert(component, (raw_size * comp_count, comp_count, r3));
    }

    // Project total compression
    eprintln!();
    eprintln!("=== Projection: Full Model Compression ===");
    eprintln!();

    let mut projected_raw = 0u64;
    let mut projected_z3 = 0u64;

    for (component, (total_bytes, count, ratio)) in &component_stats {
        let compressed = (*total_bytes as f64 / ratio) as u64;
        projected_raw += total_bytes;
        projected_z3 += compressed;
        if *total_bytes > 100_000_000 {
            // Only show components > 100 MB
            eprintln!(
                "  {:30} {:>3} layers × {:.1} MB = {:.2} GB → {:.2} GB ({:.2}x)",
                component,
                count,
                *total_bytes as f64 / *count as f64 / 1e6,
                *total_bytes as f64 / 1e9,
                compressed as f64 / 1e9,
                ratio
            );
        }
    }

    eprintln!();
    eprintln!("  Total raw:        {:.2} GB", projected_raw as f64 / 1e9);
    eprintln!("  Total zstd-3:     {:.2} GB", projected_z3 as f64 / 1e9);
    eprintln!(
        "  Compression:      {:.2}x",
        projected_raw as f64 / projected_z3 as f64
    );
    eprintln!(
        "  Space saved:      {:.2} GB ({:.1}%)",
        (projected_raw - projected_z3) as f64 / 1e9,
        (1.0 - projected_z3 as f64 / projected_raw as f64) * 100.0
    );

    eprintln!();
    eprintln!("=== Sample Results ===");
    eprintln!("  Sampled raw:      {:.2} GB", total_raw as f64 / 1e9);
    eprintln!(
        "  zstd level 1:     {:.2} GB ({:.2}x)",
        total_z1 as f64 / 1e9,
        total_raw as f64 / total_z1 as f64
    );
    eprintln!(
        "  zstd level 3:     {:.2} GB ({:.2}x)",
        total_z3 as f64 / 1e9,
        total_raw as f64 / total_z3 as f64
    );
    eprintln!(
        "  zstd level 6:     {:.2} GB ({:.2}x)",
        total_z6 as f64 / 1e9,
        total_raw as f64 / total_z6 as f64
    );
    eprintln!(
        "  zstd level 9:     {:.2} GB ({:.2}x)",
        total_z9 as f64 / 1e9,
        total_raw as f64 / total_z9 as f64
    );
}

fn zstd_compress(data: &[u8], level: i32) -> u64 {
    match zstd::bulk::compress(data, level) {
        Ok(compressed) => compressed.len() as u64,
        Err(_) => data.len() as u64, // Fallback: no compression
    }
}

fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let search = format!("\"{}\":\"", key);
    let start = json.find(&search)? + search.len();
    let end = json[start..].find('"')?;
    Some(json[start..start + end].to_string())
}
