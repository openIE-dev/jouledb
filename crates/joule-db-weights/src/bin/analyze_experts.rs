//! Analyze intra-layer expert similarity in a JouleDB weight database.
//!
//! For each MoE layer, reads the fused expert tensor (e.g., ffn_down_exps.weight
//! with shape [inner, outer, 512]), slices it into per-expert sub-tensors, and
//! computes pairwise similarity between all 512 experts.
//!
//! This tells us whether expert clustering/dedup can compress MoE weights.
//!
//! Usage:
//!   cargo run -p joule-db-weights --release --bin analyze_experts -- \
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
        eprintln!("Usage: analyze_experts --db <path> --model-id <id>");
        std::process::exit(1);
    }

    eprintln!("=== Intra-Layer Expert Similarity Analysis ===");
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

    // Analyze fused expert tensors: ffn_down_exps.weight and ffn_up_exps.weight
    // Shape for ffn_down_exps: [inner_dim, outer_dim, n_experts]
    // In GGUF Q3_K format, the data is quantized blocks — we'll treat it as raw bytes
    // and slice by expert (last dimension stride).

    let expert_components = ["ffn_down_exps.weight", "ffn_up_exps.weight"];

    for component in &expert_components {
        // Analyze first MoE layer (layer 1 for Nemotron — layer 0 is SSM)
        let tensor_name = format!("blk.1.{}", component);

        eprintln!("Analyzing: {}", tensor_name);

        // Get metadata to find shape
        let meta_json = match db.get_tensor_meta(&model_id, &tensor_name) {
            Ok(Some(m)) => m,
            _ => {
                eprintln!("  Metadata not found, skipping");
                continue;
            }
        };

        let shape = parse_shape(&meta_json);
        if shape.len() != 3 {
            eprintln!("  Expected 3D tensor, got {:?}, skipping", shape);
            continue;
        }

        let n_experts = shape[2];
        eprintln!("  Shape: {:?} ({} experts)", shape, n_experts);

        // Load the full tensor
        let data = match db.get_tensor(&model_id, &tensor_name) {
            Ok(Some(d)) => d,
            _ => {
                eprintln!("  Data not found, skipping");
                continue;
            }
        };

        let total_size = data.len();
        let expert_size = total_size / n_experts;
        eprintln!(
            "  Total: {:.1} MB, per expert: {:.1} MB",
            total_size as f64 / 1e6,
            expert_size as f64 / 1e6
        );

        if expert_size == 0 || n_experts < 2 {
            eprintln!("  Insufficient data, skipping");
            continue;
        }

        // Slice into per-expert sub-tensors
        // GGUF stores quantized data in row-major order with the last dimension
        // being the fastest-varying. For [inner, outer, n_experts], each expert's
        // data is NOT contiguous — it's interleaved.
        //
        // Actually, in GGUF the shape is stored as [dim0, dim1, dim2] but the
        // data layout depends on quantization block structure. For block-quantized
        // formats, the data is organized as:
        //   blocks_for_row_0, blocks_for_row_1, ...
        // where each row spans the first dimension.
        //
        // For a 3D tensor [A, B, C], the total elements = A*B*C and
        // the data is stored as C groups of (A*B) elements each.
        //
        // So expert i's data starts at offset: i * (total_size / C)
        // This means experts ARE contiguous in memory!

        // Compute pairwise similarity for a random sample of expert pairs
        // (all 512*511/2 = 130,816 pairs would take too long)
        let sample_size = 100; // Random pairs to sample
        let mut similarities = Vec::with_capacity(sample_size);

        // Also compute similarity to the mean/centroid
        // First pass: find the "centroid" by comparing all experts to expert 0
        let expert_0 = &data[0..expert_size];

        let mut sim_to_first: Vec<(usize, f64)> = Vec::new();
        for e in 0..n_experts {
            let expert_e = &data[e * expert_size..(e + 1) * expert_size];
            let sim = byte_similarity(expert_0, expert_e);
            sim_to_first.push((e, sim));
        }
        sim_to_first.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        eprintln!("  Similarity to expert 0:");
        eprintln!(
            "    Most similar:  expert {} ({:.2}%)",
            sim_to_first[1].0, sim_to_first[1].1
        );
        eprintln!(
            "    Median:        expert {} ({:.2}%)",
            sim_to_first[n_experts / 2].0,
            sim_to_first[n_experts / 2].1
        );
        eprintln!(
            "    Least similar: expert {} ({:.2}%)",
            sim_to_first[n_experts - 1].0,
            sim_to_first[n_experts - 1].1
        );

        // Distribution buckets
        let mut buckets = [0usize; 10]; // 0-10%, 10-20%, ..., 90-100%
        for &(_, sim) in &sim_to_first {
            let bucket = (sim / 10.0).min(9.0) as usize;
            buckets[bucket] += 1;
        }

        eprintln!("  Similarity distribution (vs expert 0):");
        for (i, &count) in buckets.iter().enumerate() {
            if count > 0 {
                let bar = "#".repeat((count as f64 / n_experts as f64 * 50.0) as usize);
                eprintln!(
                    "    {:>3}-{:>3}%: {:>4} experts  {}",
                    i * 10,
                    (i + 1) * 10,
                    count,
                    bar
                );
            }
        }

        // Sample random pairs for general pairwise similarity
        let mut rng_state: u64 = 42;
        for _ in 0..sample_size {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let a = (rng_state >> 32) as usize % n_experts;
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let b = (rng_state >> 32) as usize % n_experts;
            if a == b {
                continue;
            }

            let expert_a = &data[a * expert_size..(a + 1) * expert_size];
            let expert_b = &data[b * expert_size..(b + 1) * expert_size];
            let sim = byte_similarity(expert_a, expert_b);
            similarities.push(sim);
        }

        similarities.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let avg_sim: f64 = similarities.iter().sum::<f64>() / similarities.len() as f64;
        let max_sim = similarities.first().copied().unwrap_or(0.0);
        let min_sim = similarities.last().copied().unwrap_or(0.0);
        let median_sim = similarities[similarities.len() / 2];

        eprintln!(
            "  Random pairwise similarity ({} samples):",
            similarities.len()
        );
        eprintln!("    Max:    {:.2}%", max_sim);
        eprintln!("    Median: {:.2}%", median_sim);
        eprintln!("    Mean:   {:.2}%", avg_sim);
        eprintln!("    Min:    {:.2}%", min_sim);

        // Compression estimate
        // If we cluster experts and store centroids + deltas:
        // - Find groups of similar experts (>50% similarity)
        // - Store 1 centroid per group + per-expert deltas
        let high_sim_count = sim_to_first.iter().filter(|(_, s)| *s > 10.0).count();
        let very_high_count = sim_to_first.iter().filter(|(_, s)| *s > 25.0).count();

        eprintln!("  Clustering potential:");
        eprintln!(
            "    >10% similar to expert 0: {} / {} experts",
            high_sim_count, n_experts
        );
        eprintln!(
            "    >25% similar to expert 0: {} / {} experts",
            very_high_count, n_experts
        );

        // Also try: how well does zstd compress a single expert?
        eprintln!();
        eprintln!(
            "  Per-expert zstd compressibility (expert 0, {:.1} KB):",
            expert_size as f64 / 1024.0
        );
        // We don't have zstd here — estimate from byte entropy
        let entropy = byte_entropy(expert_0);
        let theoretical_min = (entropy / 8.0 * expert_size as f64) as usize;
        eprintln!("    Byte entropy:      {:.2} bits/byte (max 8.0)", entropy);
        eprintln!(
            "    Theoretical min:   {:.1} KB ({:.1}x compression)",
            theoretical_min as f64 / 1024.0,
            expert_size as f64 / theoretical_min as f64
        );

        eprintln!();
    }
}

fn byte_similarity(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let len = a.len();
    let chunks = len / 8;
    let mut identical = 0usize;

    let a_u64 = unsafe { std::slice::from_raw_parts(a.as_ptr() as *const u64, chunks) };
    let b_u64 = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u64, chunks) };

    for i in 0..chunks {
        if a_u64[i] == b_u64[i] {
            identical += 8;
        } else {
            let xor = a_u64[i] ^ b_u64[i];
            for &byte in &xor.to_le_bytes() {
                if byte == 0 {
                    identical += 1;
                }
            }
        }
    }
    for i in (chunks * 8)..len {
        if a[i] == b[i] {
            identical += 1;
        }
    }

    identical as f64 / len as f64 * 100.0
}

fn byte_entropy(data: &[u8]) -> f64 {
    let mut counts = [0u64; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let len = data.len() as f64;
    let mut entropy = 0.0f64;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

fn parse_shape(json: &str) -> Vec<usize> {
    let mut shape = Vec::new();
    if let Some(start) = json.find("\"shape\":[") {
        let after = &json[start + 9..];
        if let Some(end) = after.find(']') {
            for dim in after[..end].split(',') {
                if let Ok(d) = dim.trim().parse::<usize>() {
                    shape.push(d);
                }
            }
        }
    }
    shape
}
