//! Sweep shapes and layouts, print the `MLComputePlan` device for the
//! similarity op under `CPUAndNeuralEngine`, and check scores vs CPU.
//!
//! `cargo run --release -p joule-ane-rt --example ane_placement_probe`

#[cfg(target_os = "macos")]
fn main() {
    use joule_ane_rt::{HdcOptions, Layout, PowerSampling, cpu_dot_scores, hdc_similarity};
    let bipolar = |len: usize, seed: u64| -> Vec<i8> {
        let mut s = seed | 1;
        (0..len)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                if s >> 63 == 0 { 1 } else { -1 }
            })
            .collect()
    };
    let d = 2048;
    for &(n, k) in &[(1, 64), (8, 64), (64, 256), (64, 1024), (256, 1024), (512, 4096), (1024, 1024)] {
        let q = bipolar(n * d, 3 + n as u64);
        let m = bipolar(k * d, 5 + k as u64);
        let want = cpu_dot_scores(&q, &m, d);
        for layout in [Layout::MatMul, Layout::ChannelsFirstConv] {
            match hdc_similarity(&q, &m, d, HdcOptions { layout, power: PowerSampling::Off }) {
                Ok(run) => {
                    let op = run.placement.similarity_op();
                    println!(
                        "probe n={n:>5} k={k:>5} d={d} layout={:<13} similarity_op={} preferred={} supported=[{}] cost={} exact={} predict_s={:.6}",
                        layout.as_str(),
                        op.map(|o| o.operator.as_str()).unwrap_or("?"),
                        run.placement.similarity_device().as_str(),
                        op.map(|o| o.supported.iter().map(|d| d.as_str()).collect::<Vec<_>>().join(",")).unwrap_or_default(),
                        op.and_then(|o| o.estimated_cost).map(|c| format!("{c:.3}")).unwrap_or_else(|| "none".into()),
                        run.scores == want,
                        run.predict_s
                    );
                }
                Err(e) => println!("probe n={n} k={k} d={d} layout={} error: {e}", layout.as_str()),
            }
        }
    }
}

#[cfg(not(target_os = "macos"))]
fn main() {
    println!("ane_placement_probe: Core ML is macOS-only; nothing to probe here.");
}
