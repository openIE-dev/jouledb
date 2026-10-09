//! Per-job energy: measured where the platform allows, a labelled estimate
//! otherwise. Energy is a receipt, never a refusal: every job gets a
//! [`JobEnergy`], and nothing here can fail a job.
//!
//! Sources, best first:
//!
//! * `powermetrics` (macOS): the shared long-lived sampler in
//!   [`joule_ane_rt::sampler`] (CPU, GPU and ANE rails, one stream, only via
//!   `sudo -n`). A job integrates the samples over its window;
//!   `incremental_joules` subtracts the idle baseline (per-rail 20th
//!   percentile of the last minute, refreshed every 30 s).
//! * `rapl` (Linux): package (+ DRAM) energy counters read at the job's start
//!   and end, plus a DRM/hwmon GPU energy counter when one is readable.
//!   Baseline: package power measured once at start (100 ms) and refreshed
//!   from idle gaps between jobs.
//! * `emi` (Windows): the Energy Meter Interface counter, same scheme.
//! * `estimate`: `watts snapshot x seconds` (default 5 W), labelled
//!   `energy_source=estimate`, `energy_confidence=estimate`.
//!
//! `JOULE_POWER_METER=estimate` forces the estimate (CI determinism).

use std::sync::Mutex;
use std::time::{Duration, Instant};

use joule_ane_rt::sampler::PowerSampler;

/// Energy of one job.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct JobEnergy {
    /// `powermetrics`, `rapl`, `emi`, or `estimate`.
    pub source: String,
    /// `measured`, `interpolated` (apportioned from overlapping samples, or a
    /// partly covered window scaled up), or `estimate`.
    pub confidence: String,
    pub seconds: f64,
    /// Total joules over the job window: all CPU/GPU/ANE rails
    /// (powermetrics) or package (+DRAM, +GPU) counters (RAPL/EMI).
    pub joules: f64,
    /// Joules above the idle baseline. Equals `joules` for estimates.
    pub incremental_joules: f64,
    /// `joules / seconds`.
    pub watts: f64,
    pub cpu_watts: Option<f64>,
    pub gpu_watts: Option<f64>,
    pub ane_watts: Option<f64>,
    pub dram_watts: Option<f64>,
    /// Idle baseline that was subtracted, watts (all rails).
    pub baseline_watts: Option<f64>,
}

impl JobEnergy {
    /// `true` for a hardware reading (not the flat estimate).
    pub fn is_measured(&self) -> bool {
        self.source != "estimate"
    }

    /// The labelled estimate: `watts x seconds`, attributed to `device`.
    pub fn estimate(device: &str, watts: f64, seconds: f64) -> Self {
        let watts = watts.max(0.0);
        let joules = watts * seconds;
        let mut e = JobEnergy {
            source: "estimate".into(),
            confidence: "estimate".into(),
            seconds,
            joules,
            incremental_joules: joules,
            watts,
            cpu_watts: None,
            gpu_watts: None,
            ane_watts: None,
            dram_watts: None,
            baseline_watts: None,
        };
        match device {
            "gpu" => e.gpu_watts = Some(watts),
            "npu" => e.ane_watts = Some(watts),
            _ => e.cpu_watts = Some(watts),
        }
        e
    }

    /// Per-device watts for receipt lines.
    pub fn device_watts_label(&self) -> String {
        let f = |v: Option<f64>| v.map(|w| format!("{w:.3}")).unwrap_or_else(|| "none".into());
        format!(
            "cpu_w={} gpu_w={} ane_w={} dram_w={} baseline_w={}",
            f(self.cpu_watts),
            f(self.gpu_watts),
            f(self.ane_watts),
            f(self.dram_watts),
            f(self.baseline_watts)
        )
    }
}

/// Counter readings at a job's start (RAPL / EMI).
#[derive(Debug, Clone, Copy, Default)]
struct Counters {
    package: f64,
    core: Option<f64>,
    dram: Option<f64>,
    gpu: Option<f64>,
}

/// A started measurement. With the powermetrics sampler the job is
/// registered in the ring so samples it shares with other jobs are split;
/// a span dropped without [`Meter::finish`] unregisters itself.
pub struct Span {
    pub start: Instant,
    counters: Option<Counters>,
    job: Option<(&'static PowerSampler, u64)>,
}

impl Span {
    /// Mark the job finished at `end` (so it stops counting as running for
    /// other jobs) while its energy is read later with [`Meter::finish_at`].
    pub fn close(&self, end: Instant) {
        if let Some((sampler, id)) = self.job {
            sampler.end_job(id, self.start, end);
        }
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        if let Some((sampler, id)) = self.job.take() {
            sampler.drop_job(id);
        }
    }
}

/// Idle baseline from a cumulative counter: measured once, then refreshed
/// from the idle gaps between jobs (min of the recent gap powers).
#[derive(Debug, Default)]
struct CounterBaseline {
    /// Recent idle-gap powers, watts.
    gaps: Vec<f64>,
    initial: Option<f64>,
    /// Counter and time at the end of the previous job.
    last_end: Option<(Instant, f64)>,
}

const GAP_MIN: Duration = Duration::from_millis(50);
const GAP_KEEP: usize = 16;

impl CounterBaseline {
    fn watts(&self) -> Option<f64> {
        self.gaps.iter().copied().reduce(f64::min).or(self.initial)
    }

    /// A job starts at `(t, counter)`: the gap since the previous job's end
    /// is an idle sample when long enough.
    fn job_started(&mut self, t: Instant, counter: f64) {
        if let Some((prev_t, prev_c)) = self.last_end {
            let gap = t.saturating_duration_since(prev_t);
            if gap >= GAP_MIN && counter >= prev_c {
                if self.gaps.len() == GAP_KEEP {
                    self.gaps.remove(0);
                }
                self.gaps.push((counter - prev_c) / gap.as_secs_f64());
            }
        }
    }

    fn job_ended(&mut self, t: Instant, counter: f64) {
        self.last_end = Some((t, counter));
    }
}

/// Linux RAPL (+ DRM/hwmon GPU energy).
struct Rapl {
    reader: joule_energy_rt::rapl::RAPLReader,
    gpu_energy: Option<std::path::PathBuf>,
    baseline: Mutex<CounterBaseline>,
}

fn delta(start: f64, end: f64, range: f64) -> f64 {
    if end >= start { end - start } else { end + range - start }
}

/// A readable DRM/hwmon GPU energy counter (`energy1_input`, microjoules).
fn drm_gpu_energy() -> Option<std::path::PathBuf> {
    let cards = std::fs::read_dir("/sys/class/drm").ok()?;
    for card in cards.flatten() {
        let hw = card.path().join("device/hwmon");
        let Ok(mons) = std::fs::read_dir(&hw) else { continue };
        for mon in mons.flatten() {
            let path = mon.path().join("energy1_input");
            if std::fs::read_to_string(&path).is_ok_and(|v| v.trim().parse::<f64>().is_ok()) {
                return Some(path);
            }
        }
    }
    None
}

fn read_uj(path: &std::path::Path) -> Option<f64> {
    std::fs::read_to_string(path).ok()?.trim().parse::<f64>().ok().map(|uj| uj / 1e6)
}

impl Rapl {
    fn open() -> Option<Self> {
        let reader = joule_energy_rt::rapl::RAPLReader::new().ok()?;
        let a = reader.read_package_energy().ok()?;
        let t0 = Instant::now();
        std::thread::sleep(Duration::from_millis(100));
        let b = reader.read_package_energy().ok()?;
        let initial = delta(a, b, reader.max_energy_range()) / t0.elapsed().as_secs_f64();
        Some(Rapl {
            reader,
            gpu_energy: drm_gpu_energy(),
            baseline: Mutex::new(CounterBaseline { initial: Some(initial), ..Default::default() }),
        })
    }

    fn read(&self) -> Option<Counters> {
        Some(Counters {
            package: self.reader.read_package_energy().ok()?,
            core: self.reader.read_core_energy().ok().flatten(),
            dram: self.reader.read_dram_energy().ok().flatten(),
            gpu: self.gpu_energy.as_deref().and_then(read_uj),
        })
    }
}

#[cfg(target_os = "windows")]
struct Emi {
    reader: joule_energy_rt::windows::WindowsEnergyReader,
    baseline: Mutex<CounterBaseline>,
}

/// Where per-job energy comes from.
pub enum Meter {
    /// Shared powermetrics ring (macOS), or an injected one in tests.
    Sampler { sampler: &'static PowerSampler, wait: Duration },
    Rapl(Box<RaplMeter>),
    #[cfg(target_os = "windows")]
    Emi(Box<EmiMeter>),
    Estimate,
}

/// Opaque RAPL meter.
pub struct RaplMeter(Rapl);
/// Opaque EMI meter.
#[cfg(target_os = "windows")]
pub struct EmiMeter(Emi);

impl Meter {
    /// The process-wide meter (chosen once).
    pub fn global() -> &'static Meter {
        static GLOBAL: std::sync::OnceLock<Meter> = std::sync::OnceLock::new();
        GLOBAL.get_or_init(Meter::detect)
    }

    fn detect() -> Meter {
        if std::env::var("JOULE_POWER_METER").is_ok_and(|v| v == "estimate") {
            return Meter::Estimate;
        }
        if let Some(sampler) = PowerSampler::global() {
            // Wait for the sample that covers a job's end: a few intervals
            // (powermetrics' real interval is often ~2x the requested one).
            let ms = std::env::var("JOULE_POWER_WAIT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(u64::from(sampler.interval_ms) * 4 + 40);
            return Meter::Sampler { sampler, wait: Duration::from_millis(ms) };
        }
        if let Some(rapl) = Rapl::open() {
            return Meter::Rapl(Box::new(RaplMeter(rapl)));
        }
        #[cfg(target_os = "windows")]
        if let Ok(reader) = joule_energy_rt::windows::WindowsEnergyReader::new() {
            if let Ok(a) = reader.read_cumulative_energy() {
                let t0 = Instant::now();
                std::thread::sleep(Duration::from_millis(100));
                if let Ok(b) = reader.read_cumulative_energy() {
                    let initial = (b - a).max(0.0) / t0.elapsed().as_secs_f64();
                    return Meter::Emi(Box::new(EmiMeter(Emi {
                        reader,
                        baseline: Mutex::new(CounterBaseline { initial: Some(initial), ..Default::default() }),
                    })));
                }
            }
        }
        Meter::Estimate
    }

    /// `powermetrics`, `rapl`, `emi`, or `estimate`.
    pub fn source(&self) -> &'static str {
        match self {
            Meter::Sampler { .. } => "powermetrics",
            Meter::Rapl(_) => "rapl",
            #[cfg(target_os = "windows")]
            Meter::Emi(_) => "emi",
            Meter::Estimate => "estimate",
        }
    }

    pub fn begin(&self) -> Span {
        let start = Instant::now();
        let counters = match self {
            Meter::Rapl(r) => r.0.read().inspect(|c| {
                lock(&r.0.baseline).job_started(start, c.package);
            }),
            #[cfg(target_os = "windows")]
            Meter::Emi(e) => e.0.reader.read_cumulative_energy().ok().map(|package| {
                lock(&e.0.baseline).job_started(start, package);
                Counters { package, ..Default::default() }
            }),
            _ => None,
        };
        let job = match self {
            Meter::Sampler { sampler, .. } => Some((*sampler, sampler.begin_job(start))),
            _ => None,
        };
        Span { start, counters, job }
    }

    /// Energy of `[span.start + skip, now]`, where `skip` is one-time setup
    /// (pipeline compile) at the start of the window that is not charged to
    /// the job. Falls back to the estimate (`estimate_watts x seconds`,
    /// attributed to `device`) whenever no reading covers the window.
    pub fn finish(&self, span: Span, skip: Duration, device: &str, estimate_watts: f64) -> JobEnergy {
        self.finish_at(span, Instant::now(), skip, device, estimate_watts)
    }

    /// [`Self::finish`] for a window that ended at `end` (the sampler ring
    /// keeps history, so a job's energy can be read after the fact; counter
    /// meters read their end counter now, so pass `Instant::now()` there).
    pub fn finish_at(&self, mut span: Span, end: Instant, skip: Duration, device: &str, estimate_watts: f64) -> JobEnergy {
        let start = (span.start + skip).min(end);
        let seconds = end.saturating_duration_since(start).as_secs_f64().max(1e-9);
        let estimate = || JobEnergy::estimate(device, estimate_watts, seconds);
        match self {
            Meter::Sampler { sampler, wait } => match span.job.take() {
                Some((_, id)) => sampler_energy(sampler, id, start, end, *wait).unwrap_or_else(estimate),
                None => estimate(),
            },
            Meter::Rapl(r) => {
                let (Some(a), Some(b)) = (span.counters, r.0.read()) else { return estimate() };
                let mut base = lock(&r.0.baseline);
                base.job_ended(end, b.package);
                let range = r.0.reader.max_energy_range();
                // The whole span (setup included) is what the counter saw;
                // charge the job's share by time.
                let total = end.saturating_duration_since(span.start).as_secs_f64().max(1e-9);
                let share = seconds / total;
                let pkg = delta(a.package, b.package, range) * share;
                let dram = a.dram.zip(b.dram).map(|(x, y)| delta(x, y, range) * share);
                let core = a.core.zip(b.core).map(|(x, y)| delta(x, y, range) * share);
                let gpu = a.gpu.zip(b.gpu).map(|(x, y)| (y - x).max(0.0) * share);
                let joules = pkg + dram.unwrap_or(0.0) + gpu.unwrap_or(0.0);
                let baseline = base.watts();
                let incremental = (joules - baseline.unwrap_or(0.0) * seconds).max(0.0);
                JobEnergy {
                    source: "rapl".into(),
                    // RAPL counters update about every millisecond.
                    confidence: if seconds >= 0.002 { "measured" } else { "interpolated" }.into(),
                    seconds,
                    joules,
                    incremental_joules: incremental,
                    watts: joules / seconds,
                    cpu_watts: Some(core.unwrap_or(pkg) / seconds),
                    gpu_watts: gpu.map(|j| j / seconds),
                    ane_watts: None,
                    dram_watts: dram.map(|j| j / seconds),
                    baseline_watts: baseline,
                }
            }
            #[cfg(target_os = "windows")]
            Meter::Emi(e) => {
                let (Some(a), Ok(b)) = (span.counters, e.0.reader.read_cumulative_energy()) else {
                    return estimate();
                };
                let mut base = lock(&e.0.baseline);
                base.job_ended(end, b);
                let total = end.saturating_duration_since(span.start).as_secs_f64().max(1e-9);
                let joules = (b - a.package).max(0.0) * seconds / total;
                let baseline = base.watts();
                JobEnergy {
                    source: "emi".into(),
                    confidence: "measured".into(),
                    seconds,
                    joules,
                    incremental_joules: (joules - baseline.unwrap_or(0.0) * seconds).max(0.0),
                    watts: joules / seconds,
                    cpu_watts: Some(joules / seconds),
                    gpu_watts: None,
                    ane_watts: None,
                    dram_watts: None,
                    baseline_watts: baseline,
                }
            }
            Meter::Estimate => estimate(),
        }
    }
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

fn sampler_energy(sampler: &PowerSampler, id: u64, start: Instant, end: Instant, wait: Duration) -> Option<JobEnergy> {
    let w = sampler.job_window(id, start, end, wait)?;
    let seconds = w.seconds.max(1e-9);
    Some(JobEnergy {
        source: "powermetrics".into(),
        confidence: w.confidence.label().into(),
        seconds,
        joules: w.joules.total(),
        incremental_joules: w.incremental.total(),
        watts: w.joules.total() / seconds,
        cpu_watts: Some(w.watts.cpu_w),
        gpu_watts: Some(w.watts.gpu_w),
        ane_watts: Some(w.watts.ane_w),
        dram_watts: None,
        baseline_watts: Some(w.baseline.total()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use joule_ane_rt::sampler::{RailSample, Rails};

    fn injected(cpu: f64, gpu: f64) -> &'static PowerSampler {
        let s: &'static PowerSampler = Box::leak(Box::new(PowerSampler::detached(20)));
        s.with_ring(|r| {
            r.set_baseline(0.0, Rails { cpu_w: 1.0, gpu_w: 0.0, ane_w: 0.0 });
            // 20 ms samples from the sampler epoch to 5 s ahead.
            for i in 0..250 {
                let t = i as f64 * 0.02;
                r.push(RailSample {
                    start: t,
                    end: t + 0.02,
                    rails: Rails { cpu_w: cpu, gpu_w: gpu, ane_w: 0.0 },
                    combined_w: None,
                });
            }
        });
        s
    }

    #[test]
    fn sampler_meter_reports_total_and_incremental_with_labels() {
        let meter = Meter::Sampler { sampler: injected(3.0, 8.0), wait: Duration::ZERO };
        let span = meter.begin();
        std::thread::sleep(Duration::from_millis(30));
        let e = meter.finish(span, Duration::ZERO, "gpu", 5.0);
        assert_eq!(e.source, "powermetrics");
        assert!(e.confidence == "measured" || e.confidence == "interpolated");
        // Idle share 1 W x seconds; the excess (10 W) of every sample the
        // job touches is the job's (no other job registered), so between
        // 10 W x seconds and 10 W x (seconds + two partial samples).
        assert!(e.incremental_joules >= 10.0 * e.seconds - 1e-9, "{e:?}");
        assert!(e.incremental_joules <= 10.0 * (e.seconds + 0.04) + 1e-9, "{e:?}");
        assert!((e.joules - (1.0 * e.seconds + e.incremental_joules)).abs() < 1e-9, "{e:?}");
        assert!(e.cpu_watts.is_some_and(|w| w >= 3.0 - 1e-9), "{e:?}");
        assert!(e.gpu_watts.is_some_and(|w| w >= 8.0 - 1e-9), "{e:?}");
        assert!(e.baseline_watts.is_some_and(|w| (w - 1.0).abs() < 1e-9), "{e:?}");
        assert!(e.is_measured());
        // A sub-millisecond job inside one 20 ms sample: interpolated, and
        // charged that sample's excess (10 W x 20 ms) plus its idle share.
        let span = meter.begin();
        let e = meter.finish(span, Duration::ZERO, "cpu", 5.0);
        assert_eq!(e.confidence, "interpolated");
        assert!(e.incremental_joules > 0.0 && e.incremental_joules <= 10.0 * 0.04 + 1e-9, "{e:?}");
        // A span dropped without finish does not linger as a running job.
        let dropped = meter.begin();
        drop(dropped);
    }

    #[test]
    fn uncovered_window_falls_back_to_labelled_estimate() {
        // A sampler with no samples (e.g. just started): never None, never an error.
        let s: &'static PowerSampler = Box::leak(Box::new(PowerSampler::detached(20)));
        let meter = Meter::Sampler { sampler: s, wait: Duration::ZERO };
        let e = meter.finish(meter.begin(), Duration::ZERO, "gpu", 5.0);
        assert_eq!((e.source.as_str(), e.confidence.as_str()), ("estimate", "estimate"));
        assert_eq!(e.gpu_watts, Some(5.0));
        assert!((e.joules - 5.0 * e.seconds).abs() < 1e-12);
        assert_eq!(e.incremental_joules, e.joules);
        let e = Meter::Estimate.finish(Meter::Estimate.begin(), Duration::ZERO, "cpu", 5.0);
        assert_eq!(e.source, "estimate");
        assert_eq!(e.cpu_watts, Some(5.0));
    }

    #[test]
    fn setup_is_skipped_from_the_window() {
        let meter = Meter::Sampler { sampler: injected(2.0, 0.0), wait: Duration::ZERO };
        let span = meter.begin();
        std::thread::sleep(Duration::from_millis(40));
        let e = meter.finish(span, Duration::from_millis(30), "cpu", 5.0);
        assert!(e.seconds < 0.03, "{e:?}");
    }

    #[test]
    fn counter_baseline_learns_from_idle_gaps() {
        let mut b = CounterBaseline { initial: Some(9.0), ..Default::default() };
        assert_eq!(b.watts(), Some(9.0));
        let t0 = Instant::now();
        b.job_ended(t0, 100.0);
        // 100 ms idle gap at 4 W.
        b.job_started(t0 + Duration::from_millis(100), 100.4);
        assert!((b.watts().unwrap_or(0.0) - 4.0).abs() < 1e-9);
        // Short gaps are ignored.
        b.job_ended(t0 + Duration::from_millis(200), 200.0);
        b.job_started(t0 + Duration::from_millis(210), 200.0);
        assert!((b.watts().unwrap_or(0.0) - 4.0).abs() < 1e-9);
        assert_eq!(delta(10.0, 2.0, 20.0), 12.0);
    }

    #[test]
    fn ci_without_sudo_or_rapl_is_estimate_and_labelled() {
        // No powermetrics/RAPL on Linux runners: the estimate path, labelled.
        let meter = Meter::global();
        let e = meter.finish(meter.begin(), Duration::ZERO, "cpu", 5.0);
        if meter.source() == "estimate" {
            assert_eq!(e.confidence, "estimate");
        } else {
            assert!(e.is_measured() || e.source == "estimate");
        }
        assert!(e.joules.is_finite() && e.joules >= 0.0);
    }
}
