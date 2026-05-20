//! NeuroPlay Query Example
//!
//! Demonstrates SigQL for neurological assessment:
//! "How does tremor change under cognitive load?"
//!
//! This example shows:
//! - Signal source definition
//! - Frequency-domain filtering (4-12Hz Parkinsonian tremor band)
//! - Cross-signal correlation (tremor vs cognitive load)
//! - Uncertainty-aware results

use sigql::prelude::*;
use sigql::runtime::{Runtime, RuntimeConfig, SignalSource};
use sigql::types::DynSignal;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== SigQL NeuroPlay Example ===\n");

    // =========================================================
    // 1. Define the clinical query in SigQL
    // =========================================================

    let query_str = r#"
        FROM controller.right_hand.accel AS tremor
        
        LET filtered = tremor |> bandpass(4Hz, 12Hz) |> hilbert |> envelope
        
        WHERE task_phase = 'dual_task'
        
        TRANSFORM bandpass(4Hz, 12Hz)
        
        WINDOW sliding(2s, 500ms)
        
        AGGREGATE {
            tremor_power: band_power(4Hz..12Hz),
            dominant_freq: dominant_frequency,
            complexity: spectral_entropy
        }
        
        RETURNING confidence(0.95)
    "#;

    println!("Query:\n{}\n", query_str);

    // =========================================================
    // 2. Parse the query
    // =========================================================

    let query = sigql::parse(query_str)?;
    println!("✓ Query parsed successfully");
    println!("  - FROM clauses: {}", query.from.len());
    println!("  - Transforms: {}", query.transforms.len());
    println!("  - Window: {:?}", query.window.is_some());
    println!("  - Aggregate: {:?}", query.aggregate.is_some());
    println!();

    // =========================================================
    // 3. Generate synthetic tremor data
    // =========================================================

    let sample_rate = 200u32;
    let duration_sec = 10.0;
    let num_samples = (sample_rate as f64 * duration_sec) as usize;

    // Simulate Parkinsonian tremor: 5Hz oscillation with noise
    let tremor_freq = 5.0; // Hz
    let tremor_amplitude = 0.5; // m/s²
    let noise_level = 0.1;

    let samples: Vec<f64> = (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate as f64;
            // Tremor signal + cognitive load modulation + noise
            let tremor = tremor_amplitude * (2.0 * std::f64::consts::PI * tremor_freq * t).sin();
            // Add cognitive load effect (tremor increases during load)
            let load_factor = if t > 5.0 { 1.5 } else { 1.0 };
            let noise = (rand_simple(i) - 0.5) * noise_level;
            tremor * load_factor + noise
        })
        .collect();

    println!("✓ Generated synthetic tremor data");
    println!("  - Sample rate: {} Hz", sample_rate);
    println!("  - Duration: {} seconds", duration_sec);
    println!("  - Samples: {}", num_samples);
    println!("  - Tremor frequency: {} Hz", tremor_freq);
    println!();

    // =========================================================
    // 4. Set up runtime and register signals
    // =========================================================

    let mut runtime = Runtime::new(RuntimeConfig::default());

    let signal = DynSignal::new(
        "controller.right_hand.accel",
        samples.clone(),
        sample_rate,
        0, // start timestamp
    );

    runtime.register_signal("controller.right_hand.accel", signal);
    println!("✓ Registered signal source");
    println!();

    // =========================================================
    // 5. Compile and execute (demonstration)
    // =========================================================

    let plan = sigql::compile(&query, sigql::Target::Simd)?;
    println!("✓ Compiled to SIMD target");
    println!("  - Execution steps: {}", plan.steps.len());
    println!();

    // =========================================================
    // 6. Manual DSP analysis (since runtime is stub)
    // =========================================================

    println!("=== Manual Analysis (Demonstration) ===\n");

    // Compute band power in 4-12Hz range
    let band_power = compute_band_power(&samples, sample_rate, 4.0, 12.0);
    println!("Tremor Band Power (4-12Hz):");
    println!("  Value: {:.4} m²/s⁴", band_power.value);
    println!(
        "  95% CI: [{:.4}, {:.4}]",
        band_power.lower_bound, band_power.upper_bound
    );
    println!("  Samples: {}", band_power.n_samples);
    println!();

    // Find dominant frequency
    let dominant = find_dominant_frequency(&samples, sample_rate, 4.0, 12.0);
    println!("Dominant Tremor Frequency:");
    println!("  Value: {:.2} Hz", dominant.value);
    println!(
        "  95% CI: [{:.2}, {:.2}] Hz",
        dominant.lower_bound, dominant.upper_bound
    );
    println!();

    // Compute spectral entropy
    let entropy = compute_spectral_entropy(&samples, sample_rate);
    println!("Spectral Entropy (complexity):");
    println!("  Value: {:.4}", entropy.value);
    println!(
        "  Interpretation: {}",
        if entropy.value > 0.7 {
            "High complexity (noisy/irregular)"
        } else {
            "Low complexity (regular tremor)"
        }
    );
    println!();

    // =========================================================
    // 7. Clinical interpretation
    // =========================================================

    println!("=== Clinical Interpretation ===\n");

    // Split into baseline and cognitive load periods
    let baseline_power = compute_band_power(&samples[..num_samples / 2], sample_rate, 4.0, 12.0);
    let load_power = compute_band_power(&samples[num_samples / 2..], sample_rate, 4.0, 12.0);

    let power_increase = (load_power.value - baseline_power.value) / baseline_power.value * 100.0;

    println!("Baseline vs Cognitive Load:");
    println!("  Baseline power: {:.4} m²/s⁴", baseline_power.value);
    println!("  Load power: {:.4} m²/s⁴", load_power.value);
    println!("  Change: {:.1}%", power_increase);
    println!();

    if power_increase > 30.0 {
        println!("⚠️  FINDING: Tremor significantly worsens under cognitive load");
        println!("   This pattern is characteristic of Parkinsonian tremor.");
        println!("   Consider medication adjustment or further evaluation.");
    } else if power_increase > 10.0 {
        println!("📊 FINDING: Moderate tremor increase under cognitive load");
        println!("   Monitor for changes over time.");
    } else {
        println!("✓  FINDING: Tremor stable under cognitive load");
        println!("   Current management appears effective.");
    }
    println!();

    // =========================================================
    // 8. Show what SigQL enables
    // =========================================================

    println!("=== What SigQL Enables ===\n");
    println!("Instead of writing 100+ lines of Python/MATLAB:");
    println!();
    println!("  ```sql");
    println!("  SELECT tremor_power(hand_imu, 4-12Hz),");
    println!("         cognitive_impact(tremor, nback_difficulty)");
    println!("  FROM therapy_session");
    println!("  WHERE diagnosis = 'PD'");
    println!("  RETURNING confidence(0.95)");
    println!("  ```");
    println!();
    println!("SigQL compiles this to:");
    println!("  • WebGPU (browser) for real-time visualization");
    println!("  • CUDA (Jetson) for edge deployment");
    println!("  • SIMD (anywhere) for portable execution");
    println!();
    println!("All with uncertainty quantification built in.");

    Ok(())
}

// ============================================================
// Helper functions for demonstration
// ============================================================

/// Simple deterministic "random" for reproducibility
fn rand_simple(seed: usize) -> f64 {
    let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
    (x % 1000) as f64 / 1000.0
}

/// Compute band power using simple DFT (educational, not optimized)
fn compute_band_power(
    samples: &[f64],
    sample_rate: u32,
    low_hz: f64,
    high_hz: f64,
) -> UncertainValue<f64> {
    let n = samples.len();
    if n == 0 {
        return UncertainValue::default();
    }

    // Simple DFT for demonstration (real implementation uses FFT)
    let freq_resolution = sample_rate as f64 / n as f64;
    let low_bin = (low_hz / freq_resolution).ceil() as usize;
    let high_bin = (high_hz / freq_resolution).floor() as usize;

    // Compute power in band
    let mut total_power = 0.0;
    let mut bin_powers = Vec::new();

    for k in low_bin..=high_bin.min(n / 2) {
        let mut real = 0.0;
        let mut imag = 0.0;
        for (i, &sample) in samples.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }
        let power = (real * real + imag * imag) / (n * n) as f64;
        total_power += power;
        bin_powers.push(power);
    }

    // Estimate uncertainty from power variance across bins
    let mean_power = total_power / bin_powers.len().max(1) as f64;
    let variance = bin_powers
        .iter()
        .map(|p| (p - mean_power).powi(2))
        .sum::<f64>()
        / bin_powers.len().max(1) as f64;
    let std_error = variance.sqrt() / (bin_powers.len() as f64).sqrt();

    UncertainValue::from_mean_se(total_power, std_error, n)
}

/// Find dominant frequency in band
fn find_dominant_frequency(
    samples: &[f64],
    sample_rate: u32,
    low_hz: f64,
    high_hz: f64,
) -> UncertainValue<f64> {
    let n = samples.len();
    if n == 0 {
        return UncertainValue::default();
    }

    let freq_resolution = sample_rate as f64 / n as f64;
    let low_bin = (low_hz / freq_resolution).ceil() as usize;
    let high_bin = (high_hz / freq_resolution).floor() as usize;

    let mut max_power = 0.0;
    let mut max_bin = low_bin;

    for k in low_bin..=high_bin.min(n / 2) {
        let mut real = 0.0;
        let mut imag = 0.0;
        for (i, &sample) in samples.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }
        let power = real * real + imag * imag;
        if power > max_power {
            max_power = power;
            max_bin = k;
        }
    }

    let dominant_freq = max_bin as f64 * freq_resolution;

    // Uncertainty: ±1 bin
    let uncertainty = freq_resolution;

    UncertainValue::from_ci(dominant_freq, uncertainty, 0.95, n)
}

/// Compute spectral entropy
fn compute_spectral_entropy(samples: &[f64], sample_rate: u32) -> UncertainValue<f64> {
    let n = samples.len();
    if n == 0 {
        return UncertainValue::default();
    }

    // Compute power spectrum
    let mut powers = Vec::new();
    for k in 1..=n / 2 {
        let mut real = 0.0;
        let mut imag = 0.0;
        for (i, &sample) in samples.iter().enumerate() {
            let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
            real += sample * angle.cos();
            imag -= sample * angle.sin();
        }
        powers.push(real * real + imag * imag);
    }

    // Normalize to probability distribution
    let total: f64 = powers.iter().sum();
    if total < f64::EPSILON {
        return UncertainValue::from_ci(0.0, 0.0, 0.95, n);
    }

    // Shannon entropy
    let entropy: f64 = powers
        .iter()
        .map(|&p| {
            let prob = p / total;
            if prob > f64::EPSILON {
                -prob * prob.ln()
            } else {
                0.0
            }
        })
        .sum();

    // Normalize by max entropy
    let max_entropy = (powers.len() as f64).ln();
    let normalized = entropy / max_entropy;

    // Bootstrap-style uncertainty estimation (simplified)
    let uncertainty = 0.05; // Placeholder

    UncertainValue::from_ci(normalized, uncertainty, 0.95, n)
}
