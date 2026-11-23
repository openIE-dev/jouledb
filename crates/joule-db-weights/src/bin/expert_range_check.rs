//! Sparse-expert fetch proof for a blob-ingested JouleDB weight store.
//!
//! A fused MoE expert tensor `blk.N.ffn_{up,down}_exps.weight` holds all
//! `n_expert` experts contiguously at equal stride. This checks that
//! `WeightDB::get_tensor_range_mmap` can pull a single expert's bytes by
//! range, byte-identical to slicing the full tensor — and reports how little
//! of the layer a top-k routed fetch actually touches.
//!
//! Usage:
//!   cargo run -p joule-db-weights --release --bin expert_range_check -- \
//!     --db <path-to-blob.jdb> --model-id <id> [--top-k 22]

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut db_path = String::new();
    let mut model_id = String::from("nemotron-3-super-120b");
    let mut top_k: usize = 22;
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
            "--top-k" => {
                i += 1;
                top_k = args[i].parse().unwrap_or(22);
            }
            _ => {}
        }
        i += 1;
    }
    if db_path.is_empty() {
        eprintln!("Usage: expert_range_check --db <blob.jdb> --model-id <id> [--top-k 22]");
        std::process::exit(1);
    }

    eprintln!("=== Sparse-expert range-fetch proof ===");
    eprintln!("  DB:    {db_path}");
    eprintln!("  Model: {model_id}");

    // Small pool — meta lookups are tiny; tensor bytes come from the .wblob.
    let db = match joule_db_weights::WeightDB::open_with_pool_size(&db_path, 256) {
        Ok(db) => db,
        Err(e) => {
            eprintln!("ERROR opening DB: {e:?}");
            std::process::exit(1);
        }
    };

    // n_expert from the index (expert_count); default 512.
    let index = db
        .get_model_index(&model_id)
        .ok()
        .flatten()
        .unwrap_or_default();
    let n_expert = index
        .find("\"expert_count\":")
        .and_then(|p| index[p + 15..].split([',', '}']).next())
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(512);
    eprintln!("  n_expert: {n_expert}   top_k: {top_k}");

    // Find the first MoE layer that has a fused up-experts tensor in the blob.
    let mut found = None;
    for layer in 0..200 {
        let name = format!("blk.{layer}.ffn_up_exps.weight");
        match db.get_tensor_mmap(&model_id, &name) {
            Ok(Some((mmap, off, len))) => {
                found = Some((name, mmap, off, len));
                break;
            }
            _ => continue,
        }
    }
    let (name, mmap, off, len) = match found {
        Some(t) => t,
        None => {
            eprintln!("FAIL: no blob-ingested ffn_up_exps tensor found");
            std::process::exit(2);
        }
    };
    if len % n_expert != 0 {
        eprintln!("FAIL: tensor size {len} not divisible by n_expert {n_expert}");
        std::process::exit(2);
    }
    let per = len / n_expert; // bytes per expert (equal stride — see split_experts)
    eprintln!("\nFused tensor: '{name}'");
    eprintln!(
        "  total = {len} bytes ({:.1} MB)   per-expert = {per} bytes ({:.1} KB)",
        len as f64 / 1e6,
        per as f64 / 1e3
    );

    // ── Correctness: range read of expert e == slice of the full mmap ──
    let full = &mmap[off..off + len];
    let check_expert = |e: usize| -> Vec<u8> {
        let (m, o, l) = db
            .get_tensor_range_mmap(&model_id, &name, e * per, per)
            .expect("range read")
            .expect("blob-ingested");
        assert_eq!(l, per, "range len mismatch");
        m[o..o + l].to_vec()
    };
    let e_a = 5.min(n_expert - 1);
    let e_b = (n_expert - 1).max(e_a.saturating_add(1)).min(n_expert - 1);
    let range_a = check_expert(e_a);
    let slice_a = &full[e_a * per..(e_a + 1) * per];
    if range_a != slice_a {
        eprintln!("FAIL: range read of expert {e_a} != full-tensor slice");
        std::process::exit(3);
    }
    eprintln!("  OK: range read of expert {e_a} is byte-identical to full-tensor slice ({per} B)");

    // Distinctness — confirm we're addressing different experts, not aliasing.
    let range_b = check_expert(e_b);
    if e_a != e_b && range_a == range_b {
        eprintln!("  WARN: experts {e_a} and {e_b} are byte-identical (unexpected)");
    } else if e_a != e_b {
        eprintln!("  OK: expert {e_a} ≠ expert {e_b} (distinct weight bytes)");
    }

    // ── Footprint: touch only the top-k routed experts ──
    // Deterministic pseudo-routed subset (no RNG): stride through the experts.
    let stride = (n_expert / top_k).max(1);
    let routed: Vec<usize> = (0..top_k).map(|j| (j * stride) % n_expert).collect();
    let mut touched: u64 = 0;
    let mut checksum: u64 = 0;
    for &e in &routed {
        let (m, o, l) = db
            .get_tensor_range_mmap(&model_id, &name, e * per, per)
            .expect("range read")
            .expect("blob-ingested");
        // Force the pages in so resident memory reflects what we touched.
        for &byte in &m[o..o + l] {
            checksum = checksum.wrapping_add(byte as u64);
        }
        touched += l as u64;
    }
    let ratio = touched as f64 / len as f64 * 100.0;
    eprintln!("\nTop-{top_k} routed fetch (of {n_expert} experts):");
    eprintln!(
        "  touched = {touched} bytes ({:.1} MB) = {ratio:.1}% of this fused tensor",
        touched as f64 / 1e6
    );
    eprintln!("  (checksum {checksum} — forces page-in so peak RSS reflects the touch set)");
    eprintln!(
        "\nPASS: per-expert range fetch works; a routed token touches ~{ratio:.0}% of the layer's up-experts, not 100%."
    );
}
