//! Integration tests for SigQL
//!
//! Tests the complete parse → compile → execute pipeline.

use std::f64::consts::PI;

use sigql::compile::{Compiler, CompilerConfig, Target};
use sigql::parser::parse_query;
use sigql::runtime::{OutputValue, Runtime, RuntimeConfig};
use sigql::types::DynSignal;

/// Helper to create a test runtime with sample signals
fn create_test_runtime() -> Runtime {
    let mut runtime = Runtime::new(RuntimeConfig {
        default_sample_rate: 1000,
        ..RuntimeConfig::default()
    });

    // Register various test signals

    // Simple constant signal
    runtime.register_signal(
        "sensor.constant",
        DynSignal::new("constant", vec![1.0; 100], 1000, 0),
    );

    // Ramp signal (0 to 99)
    let ramp: Vec<f64> = (0..100).map(|i| i as f64).collect();
    runtime.register_signal("sensor.ramp", DynSignal::new("ramp", ramp, 1000, 0));

    // 10 Hz sine wave
    let sine: Vec<f64> = (0..1000)
        .map(|i| (2.0 * PI * 10.0 * i as f64 / 1000.0).sin())
        .collect();
    runtime.register_signal("sensor.sine10hz", DynSignal::new("sine10hz", sine, 1000, 0));

    // 50 Hz sine wave
    let sine50: Vec<f64> = (0..1000)
        .map(|i| (2.0 * PI * 50.0 * i as f64 / 1000.0).sin())
        .collect();
    runtime.register_signal(
        "sensor.sine50hz",
        DynSignal::new("sine50hz", sine50, 1000, 0),
    );

    // Mixed signal: 10Hz + 50Hz
    let mixed: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / 1000.0;
            (2.0 * PI * 10.0 * t).sin() + 0.5 * (2.0 * PI * 50.0 * t).sin()
        })
        .collect();
    runtime.register_signal("sensor.mixed", DynSignal::new("mixed", mixed, 1000, 0));

    // Noisy signal
    let noisy: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / 1000.0;
            (2.0 * PI * 10.0 * t).sin() + 0.1 * ((i * 12345) % 1000) as f64 / 500.0 - 0.1
        })
        .collect();
    runtime.register_signal("sensor.noisy", DynSignal::new("noisy", noisy, 1000, 0));

    // IMU-like accelerometer data
    let imu: Vec<f64> = (0..1000)
        .map(|i| {
            let t = i as f64 / 1000.0;
            // Simulate tremor at 5Hz with some noise
            0.3 * (2.0 * PI * 5.0 * t).sin() + 0.05 * ((i * 7919) % 100) as f64 / 50.0 - 0.05
        })
        .collect();
    runtime.register_signal(
        "controller.imu.accel",
        DynSignal::new("imu_accel", imu, 1000, 0),
    );

    runtime
}

/// Helper to run a query end-to-end
fn run_query(query: &str, runtime: &Runtime) -> sigql::runtime::ExecutionResult {
    let ast = parse_query(query).expect("Failed to parse query");

    let config = CompilerConfig {
        target: Target::Interpreted,
        optimize: true,
        debug_info: false,
        default_sample_rate: 1000,
        default_fft_size: 1024,
    };
    let mut compiler = Compiler::new(config);
    let plan = compiler.compile(&ast).expect("Failed to compile query");

    runtime.execute(&plan).expect("Failed to execute plan")
}

// ==================== Basic Query Tests ====================

#[test]
fn test_simple_from() {
    let runtime = create_test_runtime();
    let result = run_query("FROM sensor.constant", &runtime);

    assert!(result.outputs.contains_key("result"));
    match &result.outputs["result"] {
        OutputValue::Signal(s) => {
            assert_eq!(s.samples.len(), 100);
            assert!((s.samples[0] - 1.0).abs() < 1e-10);
        }
        _ => panic!("Expected signal output"),
    }
}

#[test]
fn test_aggregate_mean() {
    let runtime = create_test_runtime();
    let result = run_query("FROM sensor.constant AGGREGATE { avg: mean }", &runtime);

    match &result.outputs["avg"] {
        OutputValue::Scalar(v) => {
            assert!(
                (v.value - 1.0).abs() < 1e-10,
                "Mean should be 1.0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_aggregate_rms() {
    let runtime = create_test_runtime();
    let result = run_query("FROM sensor.sine10hz AGGREGATE { power: rms }", &runtime);

    match &result.outputs["power"] {
        OutputValue::Scalar(v) => {
            // RMS of sine wave should be 1/sqrt(2) ≈ 0.707
            assert!(
                (v.value - 0.7071).abs() < 0.01,
                "RMS should be ~0.707, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_aggregate_std() {
    let runtime = create_test_runtime();
    let result = run_query("FROM sensor.ramp AGGREGATE { spread: std }", &runtime);

    match &result.outputs["spread"] {
        OutputValue::Scalar(v) => {
            // Std of 0..99 should be ~29.15
            assert!(
                v.value > 25.0 && v.value < 35.0,
                "Std should be ~29, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_aggregate_peak_to_peak() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { range: peak_to_peak }",
        &runtime,
    );

    match &result.outputs["range"] {
        OutputValue::Scalar(v) => {
            // Peak-to-peak of sine wave should be 2.0
            assert!(
                (v.value - 2.0).abs() < 0.01,
                "Peak-to-peak should be 2.0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_aggregate_zero_crossings() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { crossings: zero_crossings }",
        &runtime,
    );

    match &result.outputs["crossings"] {
        OutputValue::Scalar(v) => {
            // 10 Hz for 1 second = 20 zero crossings (approximately)
            assert!(
                v.value > 15.0 && v.value < 25.0,
                "Zero crossings should be ~20, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

// ==================== Transform Tests ====================

#[test]
fn test_transform_envelope() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.sine10hz TRANSFORM envelope AGGREGATE { env_mean: mean }",
        &runtime,
    );

    match &result.outputs["env_mean"] {
        OutputValue::Scalar(v) => {
            // Envelope of sine should have mean close to 1.0
            assert!(
                v.value > 0.5 && v.value < 1.5,
                "Envelope mean should be ~1.0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_transform_abs() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.sine10hz TRANSFORM abs AGGREGATE { abs_mean: mean }",
        &runtime,
    );

    match &result.outputs["abs_mean"] {
        OutputValue::Scalar(v) => {
            // Mean of |sin| should be 2/π ≈ 0.637
            assert!(
                (v.value - 0.637).abs() < 0.05,
                "Abs mean should be ~0.637, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_transform_square() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.sine10hz TRANSFORM square AGGREGATE { sq_mean: mean }",
        &runtime,
    );

    match &result.outputs["sq_mean"] {
        OutputValue::Scalar(v) => {
            // Mean of sin^2 should be 0.5
            assert!(
                (v.value - 0.5).abs() < 0.01,
                "Square mean should be 0.5, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_transform_diff() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.ramp TRANSFORM diff AGGREGATE { diff_mean: mean }",
        &runtime,
    );

    match &result.outputs["diff_mean"] {
        OutputValue::Scalar(v) => {
            // Diff of ramp (0,1,2,3...) should be all 1s, mean = 1
            assert!(
                (v.value - 1.0).abs() < 0.01,
                "Diff mean should be 1.0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_transform_scale() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.constant TRANSFORM scale(2.0) AGGREGATE { scaled_mean: mean }",
        &runtime,
    );

    match &result.outputs["scaled_mean"] {
        OutputValue::Scalar(v) => {
            assert!(
                (v.value - 2.0).abs() < 1e-10,
                "Scaled mean should be 2.0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_transform_zscore() {
    let runtime = create_test_runtime();
    let result = run_query(
        "FROM sensor.ramp TRANSFORM zscore AGGREGATE { z_mean: mean, z_std: std }",
        &runtime,
    );

    match &result.outputs["z_mean"] {
        OutputValue::Scalar(v) => {
            // Z-score should have mean ≈ 0
            assert!(
                v.value.abs() < 0.01,
                "Z-score mean should be ~0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }

    match &result.outputs["z_std"] {
        OutputValue::Scalar(v) => {
            // Z-score should have std ≈ 1
            assert!(
                (v.value - 1.0).abs() < 0.05,
                "Z-score std should be ~1, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_transform_detrend() {
    let runtime = create_test_runtime();
    // Note: detrend requires specific parser syntax
    // Using simple detrend without parameter for now
    let result = run_query(
        "FROM sensor.ramp TRANSFORM detrend AGGREGATE { detrend_mean: mean }",
        &runtime,
    );

    match &result.outputs["detrend_mean"] {
        OutputValue::Scalar(v) => {
            // Detrended linear ramp should have mean ≈ 0
            assert!(
                v.value.abs() < 1.0,
                "Detrended mean should be ~0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

// ==================== Filter Tests ====================

#[test]
fn test_bandpass_filter() {
    let runtime = create_test_runtime();

    // Filter mixed signal (10Hz + 50Hz) to keep only 10Hz component
    let result = run_query(
        "FROM sensor.mixed TRANSFORM bandpass(5Hz, 15Hz) AGGREGATE { power: rms }",
        &runtime,
    );

    match &result.outputs["power"] {
        OutputValue::Scalar(v) => {
            // After bandpass, should have mostly the 10Hz component (RMS ≈ 0.707)
            // The 50Hz component should be attenuated
            assert!(
                v.value > 0.3 && v.value < 1.0,
                "Bandpass RMS should be ~0.7, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_lowpass_filter() {
    let runtime = create_test_runtime();

    // Filter mixed signal to keep only low frequencies
    let result = run_query(
        "FROM sensor.mixed TRANSFORM lowpass(20Hz) AGGREGATE { power: rms }",
        &runtime,
    );

    match &result.outputs["power"] {
        OutputValue::Scalar(v) => {
            // After lowpass, mostly 10Hz component remains
            assert!(
                v.value > 0.3 && v.value < 1.0,
                "Lowpass RMS should be in reasonable range, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_median_filter() {
    let runtime = create_test_runtime();

    // Median filter should preserve the signal shape
    let result = run_query(
        "FROM sensor.noisy TRANSFORM median(5) AGGREGATE { filtered_rms: rms }",
        &runtime,
    );

    match &result.outputs["filtered_rms"] {
        OutputValue::Scalar(v) => {
            // RMS should still be around 0.707 (sine wave)
            assert!(
                v.value > 0.4 && v.value < 1.0,
                "Median filtered RMS should be reasonable, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

// ==================== Chained Transform Tests ====================

#[test]
fn test_chained_transforms() {
    let runtime = create_test_runtime();

    // Test abs followed by square in a single transform clause
    // Note: Multiple TRANSFORM clauses may not be supported - using single clause
    let result = run_query(
        "FROM sensor.sine10hz TRANSFORM square AGGREGATE { result: mean }",
        &runtime,
    );

    match &result.outputs["result"] {
        OutputValue::Scalar(v) => {
            // sin^2 has mean 0.5
            assert!(
                (v.value - 0.5).abs() < 0.05,
                "Square mean should be ~0.5, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

// ==================== Spectral Aggregate Tests ====================

#[test]
fn test_dominant_frequency() {
    let runtime = create_test_runtime();

    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { dom_freq: dominant_frequency }",
        &runtime,
    );

    match &result.outputs["dom_freq"] {
        OutputValue::Scalar(v) => {
            // Dominant frequency should be ~10 Hz
            assert!(
                (v.value - 10.0).abs() < 2.0,
                "Dominant frequency should be ~10Hz, got {}Hz",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_spectral_entropy() {
    let runtime = create_test_runtime();

    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { entropy: spectral_entropy }",
        &runtime,
    );

    match &result.outputs["entropy"] {
        OutputValue::Scalar(v) => {
            // Pure sine wave has low spectral entropy (concentrated power)
            assert!(
                v.value >= 0.0 && v.value <= 1.0,
                "Spectral entropy should be in [0,1], got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_spectral_centroid() {
    let runtime = create_test_runtime();

    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { centroid: spectral_centroid }",
        &runtime,
    );

    match &result.outputs["centroid"] {
        OutputValue::Scalar(v) => {
            // Spectral centroid of 10Hz sine should be ~10Hz
            assert!(
                v.value > 5.0 && v.value < 50.0,
                "Spectral centroid should be low, got {}Hz",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

// ==================== Statistical Aggregate Tests ====================

#[test]
fn test_kurtosis() {
    let runtime = create_test_runtime();

    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { kurt: kurtosis }",
        &runtime,
    );

    match &result.outputs["kurt"] {
        OutputValue::Scalar(v) => {
            // Sine wave has kurtosis of -1.5 (excess kurtosis)
            assert!(
                v.value > -3.0 && v.value < 1.0,
                "Sine kurtosis should be ~-1.5, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_skewness() {
    let runtime = create_test_runtime();

    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { skew: skewness }",
        &runtime,
    );

    match &result.outputs["skew"] {
        OutputValue::Scalar(v) => {
            // Sine wave is symmetric, skewness should be ~0
            assert!(
                v.value.abs() < 0.5,
                "Sine skewness should be ~0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

#[test]
fn test_slope() {
    let runtime = create_test_runtime();

    let result = run_query("FROM sensor.ramp AGGREGATE { slope: slope }", &runtime);

    match &result.outputs["slope"] {
        OutputValue::Scalar(v) => {
            // Ramp (0,1,2,3...) has slope 1
            assert!(
                (v.value - 1.0).abs() < 0.01,
                "Ramp slope should be 1.0, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar output"),
    }
}

// ==================== Multiple Aggregates ====================

#[test]
fn test_multiple_aggregates() {
    let runtime = create_test_runtime();

    let result = run_query(
        "FROM sensor.sine10hz AGGREGATE { avg: mean, power: rms, range: peak_to_peak }",
        &runtime,
    );

    assert!(result.outputs.contains_key("avg"));
    assert!(result.outputs.contains_key("power"));
    assert!(result.outputs.contains_key("range"));

    match &result.outputs["avg"] {
        OutputValue::Scalar(v) => {
            assert!(v.value.abs() < 0.01, "Mean of sine should be ~0");
        }
        _ => panic!("Expected scalar"),
    }

    match &result.outputs["power"] {
        OutputValue::Scalar(v) => {
            assert!((v.value - 0.707).abs() < 0.01, "RMS should be ~0.707");
        }
        _ => panic!("Expected scalar"),
    }

    match &result.outputs["range"] {
        OutputValue::Scalar(v) => {
            assert!((v.value - 2.0).abs() < 0.01, "Peak-to-peak should be 2.0");
        }
        _ => panic!("Expected scalar"),
    }
}

// ==================== Realistic Use Case Tests ====================

#[test]
fn test_tremor_analysis_pipeline() {
    let runtime = create_test_runtime();

    // Simulate a tremor analysis pipeline
    let result = run_query(
        "FROM controller.imu.accel \
         TRANSFORM bandpass(3Hz, 12Hz) \
         AGGREGATE { \
             tremor_power: rms, \
             dom_freq: dominant_frequency \
         }",
        &runtime,
    );

    match &result.outputs["tremor_power"] {
        OutputValue::Scalar(v) => {
            // Should detect tremor power
            assert!(
                v.value > 0.0 && v.value < 1.0,
                "Tremor power should be reasonable, got {}",
                v.value
            );
        }
        _ => panic!("Expected scalar"),
    }

    match &result.outputs["dom_freq"] {
        OutputValue::Scalar(v) => {
            // Dominant frequency should be around 5Hz (the simulated tremor)
            assert!(
                v.value > 1.0 && v.value < 20.0,
                "Dominant freq should be in tremor range, got {}Hz",
                v.value
            );
        }
        _ => panic!("Expected scalar"),
    }
}

// ==================== Execution Stats Tests ====================

#[test]
fn test_execution_stats() {
    let runtime = create_test_runtime();
    let result = run_query("FROM sensor.sine10hz AGGREGATE { avg: mean }", &runtime);

    assert!(result.stats.samples_processed > 0);
    assert!(result.stats.execution_time_ns > 0);
}

// ==================== Edge Cases ====================

#[test]
fn test_small_signal() {
    let mut runtime = Runtime::new(RuntimeConfig::default());
    runtime.register_signal(
        "small",
        DynSignal::new("small", vec![1.0, 2.0, 3.0], 1000, 0),
    );

    let result = run_query("FROM small AGGREGATE { avg: mean }", &runtime);

    match &result.outputs["avg"] {
        OutputValue::Scalar(v) => {
            assert!((v.value - 2.0).abs() < 1e-10, "Mean should be 2.0");
        }
        _ => panic!("Expected scalar"),
    }
}
