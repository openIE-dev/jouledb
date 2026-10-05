//! ANE energy via `powermetrics`.
//!
//! On macOS 27 (checked on an M5 Max, build 26A434) `--samplers ane_power`
//! prints only the sample header; the `ANE Power: N mW` line is emitted by
//! the `cpu_power` sampler (next to CPU/GPU power and
//! `Combined Power (CPU + GPU + ANE)`). This module therefore samples
//! `cpu_power` and reads only the `ANE Power` line.
//!
//! There is no public Core ML API that returns joules. `powermetrics` needs
//! root. This module only ever runs `sudo -n` (non-interactive): if sudo
//! would prompt, it fails immediately and the caller falls back to the
//! watts-snapshot estimate. No password is prompted for, read, or stored.

use std::sync::OnceLock;

/// One `powermetrics` sample: wall interval and mean ANE power over it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PowerSample {
    pub elapsed_ms: f64,
    pub ane_mw: f64,
}

/// Measured ANE power while the Core ML prediction ran in a loop.
#[derive(Debug, Clone, PartialEq)]
pub struct PowerMeasurement {
    /// Always `"powermetrics"`. Estimates never produce this struct.
    pub source: &'static str,
    /// Time-weighted mean ANE power over all samples, watts.
    pub mean_ane_watts: f64,
    /// `mean_ane_watts * (window / reps)`: joules for one prediction.
    pub joules_per_prediction: f64,
    pub samples: usize,
    pub predictions: u64,
    pub window_s: f64,
}

/// Parse `powermetrics` text output into samples.
///
/// Recognises the per-sample header
/// `*** Sampled system activity (...) (20.51ms elapsed) ***` and the line
/// `ANE Power: 123 mW`. An `ANE Power` line belongs to the most recent header.
pub fn parse_powermetrics(text: &str) -> Vec<PowerSample> {
    let mut out = Vec::new();
    let mut elapsed: Option<f64> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("*** Sampled system activity") {
            elapsed = line
                .rfind("ms elapsed)")
                .and_then(|end| {
                    let head = &line[..end];
                    head.rfind('(').map(|start| head[start + 1..].trim().to_string())
                })
                .and_then(|s| s.parse::<f64>().ok());
            continue;
        }
        if let Some(rest) = line.strip_prefix("ANE Power:") {
            let rest = rest.trim();
            let num: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            let Ok(value) = num.parse::<f64>() else { continue };
            let mw = if rest.ends_with(" W") && !rest.ends_with("mW") {
                value * 1000.0
            } else {
                value
            };
            if let Some(ms) = elapsed {
                out.push(PowerSample { elapsed_ms: ms, ane_mw: mw });
            }
        }
    }
    out
}

/// Time-weighted mean power in watts.
pub fn mean_watts(samples: &[PowerSample]) -> Option<f64> {
    let total_ms: f64 = samples.iter().map(|s| s.elapsed_ms).sum();
    if samples.is_empty() || total_ms <= 0.0 {
        return None;
    }
    let mj: f64 = samples.iter().map(|s| s.ane_mw * s.elapsed_ms).sum();
    Some(mj / total_ms / 1000.0)
}

/// `true` when `powermetrics` can run through `sudo -n` without a prompt.
/// Checked once per process.
pub fn powermetrics_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        #[cfg(target_os = "macos")]
        {
            use std::process::{Command, Stdio};
            if !std::path::Path::new("/usr/bin/powermetrics").exists() {
                return false;
            }
            Command::new("/usr/bin/sudo")
                .args(["-n", "true"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    })
}

/// Run `work` in a loop while `powermetrics` takes `samples` samples at
/// `interval_ms`. Every sample overlaps the loop, because the loop runs
/// from before powermetrics starts until it exits. Returns `Ok(None)` when
/// powermetrics is unavailable or produced no ANE samples.
#[cfg(target_os = "macos")]
pub fn measure<E>(
    interval_ms: u32,
    samples: u32,
    mut work: impl FnMut() -> Result<(), E>,
) -> Result<Option<PowerMeasurement>, E> {
    use std::io::Read;
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    if !powermetrics_available() {
        return Ok(None);
    }
    let child = Command::new("/usr/bin/sudo")
        .args([
            "-n",
            "/usr/bin/powermetrics",
            "--samplers",
            "cpu_power",
            "-i",
            &interval_ms.to_string(),
            "-n",
            &samples.to_string(),
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn();
    let Ok(mut child) = child else { return Ok(None) };
    let stdout = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut out) = stdout {
            let _ = out.read_to_string(&mut text);
        }
        text
    });

    let started = Instant::now();
    let deadline = Duration::from_secs(15);
    let mut predictions = 0u64;
    let mut failure = None;
    loop {
        if matches!(child.try_wait(), Ok(Some(_))) {
            break;
        }
        if started.elapsed() > deadline {
            let _ = child.kill();
            break;
        }
        match work() {
            Ok(()) => predictions += 1,
            Err(e) => {
                failure = Some(e);
                break;
            }
        }
    }
    let window_s = started.elapsed().as_secs_f64();
    let _ = child.wait();
    let text = reader.join().unwrap_or_default();
    if let Some(e) = failure {
        return Err(e);
    }
    let parsed = parse_powermetrics(&text);
    let (Some(mean), true) = (mean_watts(&parsed), predictions > 0) else {
        return Ok(None);
    };
    Ok(Some(PowerMeasurement {
        source: "powermetrics",
        mean_ane_watts: mean,
        joules_per_prediction: mean * window_s / predictions as f64,
        samples: parsed.len(),
        predictions,
        window_s,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "\
Machine model: Mac17,6
OS version: 26A434
*** Sampled system activity (Mon Oct  5 05:00:00 2026 -0400) (20.51ms elapsed) ***

**** ANE usage ****

ANE Power: 120 mW

*** Sampled system activity (Mon Oct  5 05:00:00 2026 -0400) (19.49ms elapsed) ***

CPU Power: 3000 mW
GPU Power: 20 mW
ANE Power: 80 mW
Combined Power (CPU + GPU + ANE): 3100 mW
";

    #[test]
    fn parses_samples_and_integrates() {
        let s = parse_powermetrics(FIXTURE);
        assert_eq!(s.len(), 2);
        assert_eq!(s[0], PowerSample { elapsed_ms: 20.51, ane_mw: 120.0 });
        assert_eq!(s[1].ane_mw, 80.0);
        let w = mean_watts(&s).unwrap_or(0.0);
        let want = (120.0 * 20.51 + 80.0 * 19.49) / 40.0 / 1000.0;
        assert!((w - want).abs() < 1e-12, "{w} vs {want}");
    }

    #[test]
    fn no_samples_means_no_measurement() {
        assert!(parse_powermetrics("nothing here").is_empty());
        assert!(mean_watts(&[]).is_none());
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn powermetrics_is_never_available_off_macos() {
        assert!(!powermetrics_available());
    }
}
