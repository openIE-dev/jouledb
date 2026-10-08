//! Real GPU check for `fabric_reduce_sum`.
//!
//! On macOS this compiles the Metal shader, dispatches it on the system GPU
//! and compares the result with a CPU reference. Off macOS `sum_f64` must
//! return `MetalError::NotAvailable` and never pretend a GPU ran.

use joule_metal_rt::{MetalError, hdc_dot_i8, reference_hdc_dot, sum_f64};

/// Sizes that exercise the empty-tail, exact-threadgroup, one-over and
/// large non-power-of-two cases.
const SIZES: [usize; 8] = [1, 255, 256, 257, 4096, 65_537, 1_000_003, 4_194_319];

/// Small integers: every partial sum is an integer below 2^24, so an f32
/// reduction in any order is exact. A dropped tail element or a missed
/// threadgroup partial shows up as an exact mismatch.
fn integer_values(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 2_654_435_761usize) % 5 + 1) as f64).collect()
}

/// Fractional, mixed-sign values. Compared within fp tolerance.
fn fractional_values(n: usize) -> Vec<f64> {
    let mut state = 0x9E37_79B9_7F4A_7C15u64 ^ n as u64;
    (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 0.75
        })
        .collect()
}

/// CPU reference over the same f32-rounded inputs the GPU sees, in f64.
fn cpu_reference(values: &[f64]) -> (f64, f64) {
    let mut sum = 0f64;
    let mut abs = 0f64;
    for v in values {
        let x = f64::from(*v as f32);
        sum += x;
        abs += x.abs();
    }
    (sum, abs)
}

#[cfg(target_os = "macos")]
#[test]
fn fabric_reduce_sum_on_gpu_matches_cpu_reference() {
    let Some(device) = joule_metal_rt::device_or_skip("fabric_reduce_sum_on_gpu_matches_cpu_reference") else {
        return;
    };
    println!("metal device: {}", device.name());
    let mut failures = Vec::new();
    for &n in &SIZES {
        let ints = integer_values(n);
        let (want, _) = cpu_reference(&ints);
        let got = sum_f64(&ints).expect("GPU dispatch");
        let ok = got == want;
        println!(
            "gpu_reduce size={n:>9} kind=integer   cpu={want:.1} gpu={got:.1} exact={ok} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            failures.push(format!("integer n={n}: cpu={want} gpu={got}"));
        }

        let fracs = fractional_values(n);
        let (want, abs) = cpu_reference(&fracs);
        let got = sum_f64(&fracs).expect("GPU dispatch");
        // f32 accumulation: allow a relative error of 1e-5 of the L1 mass.
        let tol = 1e-5 * abs + 1e-6;
        let err = (got - want).abs();
        let ok = err <= tol;
        println!(
            "gpu_reduce size={n:>9} kind=fractional cpu={want:.6} gpu={got:.6} abs_err={err:.3e} tol={tol:.3e} {}",
            if ok { "PASS" } else { "FAIL" }
        );
        if !ok {
            failures.push(format!("fractional n={n}: cpu={want} gpu={got} err={err} tol={tol}"));
        }
    }
    let empty = sum_f64(&[]).expect("GPU dispatch of empty input");
    println!("gpu_reduce size=        0 kind=empty      gpu={empty}");
    assert_eq!(empty, 0.0);
    assert!(failures.is_empty(), "GPU reduction mismatches: {failures:#?}");
}

#[cfg(not(target_os = "macos"))]
#[test]
fn fabric_reduce_sum_is_not_available_off_macos() {
    for &n in &SIZES[..4] {
        let err = sum_f64(&integer_values(n)).unwrap_err();
        assert!(
            matches!(err, MetalError::NotAvailable { .. }),
            "expected NotAvailable, got {err:?}"
        );
    }
}

#[test]
fn reference_helpers_are_consistent() {
    let (sum, abs) = cpu_reference(&integer_values(257));
    assert_eq!(sum, integer_values(257).iter().sum::<f64>());
    assert!(abs >= sum);
    let _ = MetalError::not_macos();
}

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

/// GPU fallback kernel for the fabric HDC lane: exact integer dot products.
#[cfg(target_os = "macos")]
#[test]
fn fabric_hdc_dot_on_gpu_matches_cpu_exactly() {
    let Some(device) = joule_metal_rt::device_or_skip("fabric_hdc_dot_on_gpu_matches_cpu_exactly") else {
        return;
    };
    println!("metal device: {}", device.name());
    for &(n, k, d) in &[(1usize, 1usize, 1usize), (3, 5, 257), (8, 64, 1024), (16, 300, 2048), (2, 7, 10_000)] {
        let q = bipolar(n * d, 11 + d as u64);
        let m = bipolar(k * d, 29 + k as u64);
        let got = hdc_dot_i8(&q, &m, d).expect("GPU HDC dispatch");
        let want = reference_hdc_dot(&q, &m, d);
        let ok = got == want;
        println!("gpu_hdc_dot n={n} k={k} d={d} exact={ok} {}", if ok { "PASS" } else { "FAIL" });
        assert!(ok, "GPU HDC dot mismatch n={n} k={k} d={d}");
    }
    // The pipeline is cached per process: a repeat call pays no setup.
    let q = bipolar(64, 1);
    let (_, first) = joule_metal_rt::hdc_dot_i8_timed(&q, &q, 64).expect("GPU HDC dispatch");
    let (_, second) = joule_metal_rt::hdc_dot_i8_timed(&q, &q, 64).expect("GPU HDC dispatch");
    println!("gpu_hdc_dot timing first={first:?} second={second:?}");
    assert_eq!(second.setup_s, 0.0);
}

#[cfg(not(target_os = "macos"))]
#[test]
fn fabric_hdc_dot_is_not_available_off_macos() {
    let err = hdc_dot_i8(&bipolar(8, 1), &bipolar(8, 2), 8).unwrap_err();
    assert!(matches!(err, MetalError::NotAvailable { .. }), "{err:?}");
}
