//! Analyze cross-layer weight similarity in a JouleDB weight database.
//!
//! Reads tensors for each layer, computes:
//! - Byte-level similarity between adjacent layers (same component)
//! - Delta size (bytes that differ)
//! - Potential compression ratio from delta encoding
//!
//! Usage:
//!   cargo run -p joule-db-weights --release --bin analyze_layers -- \
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
        eprintln!("Usage: analyze_layers --db <path> --model-id <id>");
        std::process::exit(1);
    }

    eprintln!("=== Cross-Layer Weight Similarity Analysis ===");
    eprintln!("  DB: {}", db_path);
    eprintln!("  Model: {}", model_id);
    eprintln!();

    // Open JouleDB with large buffer pool for analysis
    let db = match joule_db_weights::WeightDB::open_with_pool_size(&db_path, 16384) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("ERROR: {:?}", e);
            std::process::exit(1);
        }
    };

    // Get model index
    let index_json = db
        .get_model_index(&model_id)
        .expect("DB error")
        .expect("Model not found");

    // Parse tensor names and group by component across layers
    // Nemotron naming: blk.{N}.{component}
    // We want to compare blk.0.ssm_in.weight vs blk.1.ssm_in.weight etc.
    let mut components: std::collections::BTreeMap<String, Vec<(usize, String)>> =
        std::collections::BTreeMap::new();

    // Extract tensor names
    if let Some(start) = index_json.find("\"tensor_names\":[") {
        let after = &index_json[start + 16..];
        if let Some(end) = after.find(']') {
            for name in after[..end].split(',') {
                let name = name.trim().trim_matches('"');
                if !name.starts_with("blk.") {
                    continue;
                }

                // Parse: blk.{layer_idx}.{component}
                let parts: Vec<&str> = name.splitn(3, '.').collect();
                if parts.len() < 3 {
                    continue;
                }
                if let Ok(layer_idx) = parts[1].parse::<usize>() {
                    let component = parts[2].to_string();
                    components
                        .entry(component)
                        .or_default()
                        .push((layer_idx, name.to_string()));
                }
            }
        }
    }

    // Sort each component's tensors by layer index
    for entries in components.values_mut() {
        entries.sort_by_key(|(idx, _)| *idx);
    }

    eprintln!("Components found: {}", components.len());
    for (comp, entries) in &components {
        eprintln!("  {}: {} layers", comp, entries.len());
    }
    eprintln!();

    // Analyze each component: compare adjacent layers
    let mut total_raw_bytes: u64 = 0;
    let mut total_delta_bytes: u64 = 0;
    let mut total_comparisons: usize = 0;

    println!(
        "component,layer_a,layer_b,size_bytes,identical_bytes,delta_bytes,similarity_pct,delta_ratio"
    );

    for (component, entries) in &components {
        if entries.len() < 2 {
            continue;
        }

        // Only analyze components that appear in many layers (skip one-offs)
        // Also skip tiny tensors (norms, biases) — focus on weight matrices
        let first_name = &entries[0].1;
        let first_data = match db.get_tensor(&model_id, first_name) {
            Ok(Some(d)) => d,
            _ => continue,
        };

        if first_data.len() < 1024 * 1024 {
            // Skip tensors < 1 MB — too small to matter for compression
            continue;
        }

        eprintln!(
            "Analyzing: {} ({} layers, {:.1} MB each)",
            component,
            entries.len(),
            first_data.len() as f64 / 1e6
        );

        let mut prev_data = first_data;

        for window in entries.windows(2) {
            let (layer_a, _name_a) = &window[0];
            let (layer_b, name_b) = &window[1];

            let curr_data = match db.get_tensor(&model_id, name_b) {
                Ok(Some(d)) => d,
                _ => continue,
            };

            if prev_data.len() != curr_data.len() {
                prev_data = curr_data;
                continue;
            }

            let size = prev_data.len();
            let (identical, delta_size) = compute_delta_stats(&prev_data, &curr_data);
            let similarity = identical as f64 / size as f64 * 100.0;
            let delta_ratio = delta_size as f64 / size as f64;

            println!(
                "{},{},{},{},{},{},{:.2},{:.4}",
                component, layer_a, layer_b, size, identical, delta_size, similarity, delta_ratio
            );

            total_raw_bytes += size as u64;
            total_delta_bytes += delta_size as u64;
            total_comparisons += 1;

            prev_data = curr_data;
        }
    }

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("  Comparisons:     {}", total_comparisons);
    eprintln!("  Total raw:       {:.2} GB", total_raw_bytes as f64 / 1e9);
    eprintln!(
        "  Total deltas:    {:.2} GB",
        total_delta_bytes as f64 / 1e9
    );
    if total_raw_bytes > 0 {
        let ratio = total_raw_bytes as f64 / total_delta_bytes.max(1) as f64;
        let saving_pct = (1.0 - total_delta_bytes as f64 / total_raw_bytes as f64) * 100.0;
        eprintln!("  Compression:     {:.1}x", ratio);
        eprintln!("  Space savings:   {:.1}%", saving_pct);
        eprintln!(
            "  Projected size:  {:.2} GB (from {:.2} GB)",
            total_delta_bytes as f64 / 1e9
                + (total_raw_bytes as f64 / 1e9 / total_comparisons as f64),
            total_raw_bytes as f64 / 1e9
        );
    }
}

/// Compare two byte slices and compute delta statistics.
/// Returns (identical_bytes, delta_size).
///
/// Delta size is computed as: number of bytes that differ.
/// In practice, delta encoding would be more sophisticated (run-length,
/// xor+compress, etc.) but byte-level difference gives us the theoretical
/// minimum delta size.
fn compute_delta_stats(a: &[u8], b: &[u8]) -> (usize, usize) {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    // Compare in 8-byte chunks for speed
    let chunks = len / 8;
    let mut identical_bytes = 0usize;

    let a_u64 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u64, chunks) };
    let b_u64 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u64, chunks) };

    for i in 0..chunks {
        if a_u64[i] == b_u64[i] {
            identical_bytes += 8;
        } else {
            // Count identical bytes within this u64
            let xor = a_u64[i] ^ b_u64[i];
            let diff_bytes = xor.to_le_bytes();
            for &byte in &diff_bytes {
                if byte == 0 {
                    identical_bytes += 1;
                }
            }
        }
    }

    // Handle remaining bytes
    for i in (chunks * 8)..len {
        if a[i] == b[i] {
            identical_bytes += 1;
        }
    }

    let delta_size = len - identical_bytes;
    (identical_bytes, delta_size)
}
