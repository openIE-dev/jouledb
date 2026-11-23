//! Benchmark: dequant → F16 → zstd compress → store → decompress → verify
//!
//! Tests the full F16+zstd pipeline on real tensor data from JouleDB.
//! Uses a subset of tensors (one per component) to avoid reading all 61 GB.
//!
//! Usage:
//!   cargo run -p joule-db-weights --release --bin compress_benchmark -- \
//!     --db /Volumes/macos_4TB_external/neuronexus_ai/jouledb/nemotron-3-super-120b.jdb \
//!     --model-id nemotron-3-super-120b

use joule_db_weights::compress;

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
        eprintln!("Usage: compress_benchmark --db <path> --model-id <id>");
        std::process::exit(1);
    }

    println!("=== F16 + zstd Compress/Decompress Benchmark ===");
    println!();

    // Source DB: read quantized tensors
    let db = joule_db_weights::WeightDB::open_with_pool_size(&db_path, 16384)
        .expect("Failed to open source DB");

    // Compressed DB: store F16+zstd (fresh)
    let comp_db_path = db_path.replace(".jdb", "-cf16.jdb");
    // Remove stale file if exists
    let _ = std::fs::remove_file(&comp_db_path);
    let comp_db = joule_db_weights::WeightDB::open_with_pool_size(&comp_db_path, 8192)
        .expect("Failed to create compressed DB");

    // Test tensors: representative samples from different quant types
    let test_cases = [
        ("blk.0.ssm_out.weight", "Q8_0", 32), // Q8_0: 32 vals/block, 34 bytes/block
        ("blk.1.ffn_gate_inp.weight", "F32", 1), // F32: already float, just convert
        ("blk.1.ffn_down_shexp.weight", "Q8_0", 32),
        ("blk.0.ssm_in.weight", "Q3_K", 256), // Q3_K: 256 vals/block, 110 bytes/block
        ("blk.7.attn_q.weight", "Q6_K", 256), // Q6_K: 256 vals/block, 210 bytes/block
        ("output.weight", "Q6_K", 256),       // Large Q6_K tensor (440 MB)
        ("blk.1.ffn_down_exps.weight", "Q4_0", 32), // Fused experts (793 MB)
    ];

    println!(
        "{:<35} {:>10} {:>10} {:>10} {:>8} {:>10} {:>10}",
        "tensor", "raw_quant", "f16_size", "zstd_f16", "ratio", "comp_ms", "decomp_ms"
    );
    println!("{}", "-".repeat(100));

    let mut total_raw = 0u64;
    let mut total_f16 = 0u64;
    let mut total_zstd = 0u64;

    for (tensor_name, quant_type, block_size) in &test_cases {
        let raw_data = match db.get_tensor(&model_id, tensor_name) {
            Ok(Some(d)) => d,
            _ => {
                println!("  {} — not found", tensor_name);
                continue;
            }
        };

        let raw_size = raw_data.len();

        // Get metadata for element count
        let meta_json = db
            .get_tensor_meta(&model_id, tensor_name)
            .ok()
            .flatten()
            .unwrap_or_default();
        let shape = parse_shape(&meta_json);
        let n_elements: usize = shape.iter().product();

        if n_elements == 0 {
            println!("  {} — no shape info", tensor_name);
            continue;
        }

        // Step 1: Dequantize to F16
        let dequant_start = std::time::Instant::now();
        let f16_data = match *quant_type {
            "Q8_0" => compress::dequant_q8_0_to_f16(&raw_data, n_elements),
            "Q3_K" => compress::dequant_q3k_to_f16(&raw_data, n_elements),
            "Q4_K" => compress::dequant_q4k_to_f16(&raw_data, n_elements),
            "Q4_0" => compress::dequant_q4_0_to_f16(&raw_data, n_elements),
            "Q6_K" => compress::dequant_q6k_to_f16(&raw_data, n_elements),
            "F32" => {
                // F32 → F16: just truncate each float
                let mut out = vec![0u8; n_elements * 2];
                let floats = unsafe {
                    std::slice::from_raw_parts(raw_data.as_ptr() as *const f32, n_elements)
                };
                let f16s = unsafe {
                    std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u16, n_elements)
                };
                for i in 0..n_elements {
                    f16s[i] = f32_to_f16_simple(floats[i]);
                }
                out
            }
            _ => {
                println!("  {} — unsupported quant type: {}", tensor_name, quant_type);
                continue;
            }
        };
        let _dequant_time = dequant_start.elapsed();

        let f16_size = f16_data.len();

        // Step 2: Compress F16 with zstd
        let compress_start = std::time::Instant::now();
        let compressed = zstd::bulk::compress(&f16_data, 3).expect("zstd compress failed");
        let compress_time = compress_start.elapsed();
        let zstd_size = compressed.len();

        // Step 3: Decompress
        let decompress_start = std::time::Instant::now();
        let decompressed =
            zstd::bulk::decompress(&compressed, f16_size).expect("zstd decompress failed");
        let decompress_time = decompress_start.elapsed();

        // Verify zstd round-trip
        if decompressed.len() != f16_data.len() || decompressed[..] != f16_data[..] {
            println!(
                "  WARNING: zstd round-trip mismatch (size: {} vs {})",
                decompressed.len(),
                f16_data.len()
            );
        }

        let ratio = f16_size as f64 / zstd_size as f64;

        println!(
            "{:<35} {:>10} {:>10} {:>10} {:>7.2}x {:>9.1}ms {:>9.1}ms",
            tensor_name,
            format_bytes(raw_size),
            format_bytes(f16_size),
            format_bytes(zstd_size),
            ratio,
            compress_time.as_secs_f64() * 1000.0,
            decompress_time.as_secs_f64() * 1000.0,
        );

        total_raw += raw_size as u64;
        total_f16 += f16_size as u64;
        total_zstd += zstd_size as u64;

        // Step 4: Store compressed F16 in JouleDB
        let store_start = std::time::Instant::now();
        let result = compress::put_compressed_f16(&comp_db, &model_id, tensor_name, &f16_data, 3)
            .expect("Failed to store compressed F16");
        let store_time = store_start.elapsed();

        // Step 5: Read back from JouleDB
        let read_start = std::time::Instant::now();
        let readback = compress::get_compressed_f16(&comp_db, &model_id, tensor_name)
            .expect("Failed to read compressed F16")
            .expect("Not found");
        let read_time = read_start.elapsed();

        if readback.len() != f16_data.len() {
            println!(
                "  WARNING: JouleDB round-trip size mismatch: {} vs {}",
                readback.len(),
                f16_data.len()
            );
        } else if readback[..] != f16_data[..] {
            // Count differing bytes
            let diffs = readback
                .iter()
                .zip(f16_data.iter())
                .filter(|(a, b)| a != b)
                .count();
            println!(
                "  WARNING: JouleDB round-trip has {} differing bytes ({:.4}%)",
                diffs,
                diffs as f64 / f16_data.len() as f64 * 100.0
            );
        }

        println!(
            "{:<35} {:>10} {:>10} {:>10} {:>7} {:>9.1}ms {:>9.1}ms  ← JouleDB store/read",
            "",
            "",
            "",
            "",
            "",
            store_time.as_secs_f64() * 1000.0,
            read_time.as_secs_f64() * 1000.0,
        );
    }

    println!("{}", "-".repeat(100));
    println!(
        "{:<35} {:>10} {:>10} {:>10} {:>7.2}x",
        "TOTAL",
        format_bytes(total_raw as usize),
        format_bytes(total_f16 as usize),
        format_bytes(total_zstd as usize),
        total_f16 as f64 / total_zstd.max(1) as f64,
    );

    println!();
    println!("=== Storage Comparison ===");
    println!(
        "  Original (quantized):     {}",
        format_bytes(total_raw as usize)
    );
    println!(
        "  F16 (dequantized):        {}",
        format_bytes(total_f16 as usize)
    );
    println!(
        "  F16 + zstd-3:             {}",
        format_bytes(total_zstd as usize)
    );
    println!(
        "  F16+zstd vs quantized:    {:.2}x",
        total_raw as f64 / total_zstd as f64
    );
    println!(
        "  F16+zstd vs F16:          {:.2}x compression",
        total_f16 as f64 / total_zstd as f64
    );
    println!();
    println!("=== Value Proposition ===");
    println!("  Store ONE F16+zstd copy, serve ANY quantization on read.");
    println!("  5 variants (Q3_K, Q4_K, Q6_K, Q8_0, F16) from 1 stored copy");
    println!("  vs 5 separate GGUF files at different quant levels.");

    // Flush compressed DB
    comp_db.flush().expect("flush failed");

    // Report compressed DB file size
    if let Ok(meta) = std::fs::metadata(&comp_db_path) {
        println!(
            "  Compressed DB file: {} ({:.2} GB)",
            comp_db_path,
            meta.len() as f64 / 1e9
        );
    }
}

fn format_bytes(n: usize) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1} GB", n as f64 / 1e9)
    } else if n >= 1_000_000 {
        format!("{:.1} MB", n as f64 / 1e6)
    } else if n >= 1_000 {
        format!("{:.1} KB", n as f64 / 1e3)
    } else {
        format!("{} B", n)
    }
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

fn f32_to_f16_simple(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 0xFF {
        return (sign << 15) | 0x7C00 | ((mant >> 13) as u16 & 0x3FF);
    }

    let new_exp = exp - 112;
    if new_exp >= 31 {
        return (sign << 15) | 0x7C00;
    }
    if new_exp <= 0 {
        return sign << 15;
    }

    (sign << 15) | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}
