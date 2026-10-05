//! Core ML / Apple Neural Engine HDC similarity checks.
//!
//! macOS: runs the generated ML Program through Core ML with
//! `CPUAndNeuralEngine`, checks scores equal the CPU Hamming result exactly,
//! and prints the `MLComputePlan` placement. The placement is printed, not
//! asserted: Core ML decides where each op runs.
//!
//! Elsewhere: every entry point returns `AneError::NotAvailable`.

use joule_ane_rt::{
    AneError, HdcOptions, Layout, PowerSampling, cpu_dot_scores, hamming_from_dot, hdc_similarity,
    hdc_similarity_auto,
};

fn bipolar(len: usize, seed: u64) -> Vec<i8> {
    let mut state = seed | 1;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            if state >> 63 == 0 { 1 } else { -1 }
        })
        .collect()
}

#[cfg(target_os = "macos")]
#[test]
fn coreml_hdc_scores_match_cpu_hamming_exactly() {
    let cases = [
        (Layout::MatMul, 8, 1024, 64),
        (Layout::MatMul, 8, 2048, 64),
        (Layout::ChannelsFirstConv, 8, 1024, 64),
        (Layout::ChannelsFirstConv, 8, 2048, 64),
        // d > 2048: two exact fp16 tiles summed in i32.
        (Layout::MatMul, 4, 3000, 32),
    ];
    for (layout, n, d, k) in cases {
        let q = bipolar(n * d, 0x1234 + d as u64);
        let m = bipolar(k * d, 0xBEEF + d as u64);
        let run = hdc_similarity(&q, &m, d, HdcOptions { layout, power: PowerSampling::Off })
            .unwrap_or_else(|e| panic!("Core ML run failed for {layout:?} d={d}: {e}"));
        let want = cpu_dot_scores(&q, &m, d);
        let want_h: Vec<u32> = want.iter().map(|s| hamming_from_dot(*s, d)).collect();
        let exact = run.scores == want && run.hamming() == want_h;
        println!(
            "ane_hdc layout={} n={n} d={d} k={k} tiles={} scores_match_cpu_hamming={} compile_s={:.3} predict_s={:.6} placement={}",
            layout.as_str(),
            run.tiles,
            if exact { "yes" } else { "NO" },
            run.compile_s,
            run.predict_s,
            run.placement.summary()
        );
        for line in run.placement.lines() {
            println!("  plan op: {line}");
        }
        assert!(exact, "Core ML scores differ from CPU Hamming for {layout:?} d={d}");
    }
}

#[cfg(target_os = "macos")]
#[test]
fn coreml_hdc_auto_layout_and_power() {
    // Large enough that MLComputePlan prefers the Neural Engine on an M5 Max
    // (smaller jobs are planned on the CPU); printed, not asserted.
    let (n, d, k) = (64, 2048, 1024);
    let q = bipolar(n * d, 77);
    let m = bipolar(k * d, 99);
    let run = hdc_similarity_auto(&q, &m, d, PowerSampling::Always).expect("Core ML auto run");
    assert_eq!(run.scores, cpu_dot_scores(&q, &m, d));
    let attempts: Vec<String> = run
        .attempts
        .iter()
        .map(|(l, dev)| format!("{}@{}", l.as_str(), dev.as_str()))
        .collect();
    println!(
        "ane_hdc_auto n={n} d={d} k={k} chosen={} attempts=[{}] on_neural_engine={}",
        run.placement.summary(),
        attempts.join(", "),
        run.placement.similarity_on_neural_engine()
    );
    for line in run.placement.lines() {
        println!("  plan op: {line}");
    }
    println!(
        "ane_hdc_auto first_call setup_performed={} setup_s={:.4} compile_s={:.4} predict_s={:.6}",
        run.setup_performed, run.setup_s, run.compile_s, run.predict_s
    );
    let again = hdc_similarity_auto(&q, &m, d, PowerSampling::Off).expect("Core ML repeat run");
    println!(
        "ane_hdc_auto second_call setup_performed={} setup_s={:.4} compile_s={:.4} predict_s={:.6}",
        again.setup_performed, again.setup_s, again.compile_s, again.predict_s
    );
    assert_eq!(again.scores, run.scores);
    assert!(!again.setup_performed, "same shape + codebook must reuse the loaded model");
    assert_eq!(again.setup_s, 0.0);
    match &run.power {
        Some(p) => println!(
            "ane_hdc_power energy_source={} mean_ane_watts={:.4} joules_per_prediction={:.3e} samples={} predictions={} window_s={:.3}",
            p.source, p.mean_ane_watts, p.joules_per_prediction, p.samples, p.predictions, p.window_s
        ),
        None => println!(
            "ane_hdc_power energy_source=estimate (powermetrics_available={})",
            joule_ane_rt::powermetrics_available()
        ),
    }
}

#[cfg(not(target_os = "macos"))]
#[test]
fn ane_lane_is_not_available_off_macos() {
    let q = bipolar(16, 1);
    let m = bipolar(32, 2);
    for layout in [Layout::MatMul, Layout::ChannelsFirstConv] {
        let err = hdc_similarity(&q, &m, 16, HdcOptions { layout, power: PowerSampling::Always }).unwrap_err();
        assert!(matches!(err, AneError::NotAvailable { .. }), "{err}");
    }
    assert!(matches!(
        hdc_similarity_auto(&q, &m, 16, PowerSampling::Off),
        Err(AneError::NotAvailable { .. })
    ));
    assert!(!joule_ane_rt::is_available());
}
