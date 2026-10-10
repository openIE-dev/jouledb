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
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
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
    /// Source per rail, e.g. `cpu=rapl gpu=nvml-energy`.
    #[serde(default)]
    pub rails: String,
    /// Incremental joules of the CPU side (package + DRAM) and of the GPU
    /// (NVML) when they are metered separately.
    #[serde(default)]
    pub cpu_incremental_joules: Option<f64>,
    #[serde(default)]
    pub gpu_incremental_joules: Option<f64>,
    /// Rejected or degraded readings, and why.
    #[serde(default)]
    pub notes: Option<String>,
    /// Joules and seconds of verification work inside the job window (a
    /// CPU cross-check of an accelerator result), metered as its own
    /// `kind=verify` job and *not* included in `joules` / `seconds`.
    #[serde(default)]
    pub verify_joules: Option<f64>,
    #[serde(default)]
    pub verify_seconds: Option<f64>,
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
            rails: format!("{device}=estimate"),
            ..Default::default()
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
    nvml: Option<crate::nvml::GpuSnapshot>,
}

/// A started measurement. With the powermetrics sampler the job is
/// registered in the ring so samples it shares with other jobs are split;
/// a span dropped without [`Meter::finish`] unregisters itself.
pub struct Span {
    pub start: Instant,
    counters: Option<Counters>,
    job: Option<(&'static PowerSampler, u64)>,
    /// Verification intervals inside this job (not its time).
    holes: Vec<(Instant, Instant)>,
    /// Verification energy already read (counter meters).
    verified: Vec<JobEnergy>,
    /// Verification jobs still to read (sampler: after this job's wait).
    pending: Vec<(Box<Span>, Instant)>,
    /// Counters read when the job closed (counter meters), so energy read
    /// later does not include work done after the job.
    end_counters: Option<Counters>,
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
    /// Lowest power seen over any job window: idle can be no higher.
    floor: Option<f64>,
    /// Counter and time at the end of the previous job.
    last_end: Option<(Instant, f64)>,
}

const GAP_MIN: Duration = Duration::from_millis(50);
const GAP_KEEP: usize = 16;

impl CounterBaseline {
    /// Lowest of: the initial idle reading, the recent idle gaps, and every
    /// job's own power (a baseline above a job's power would zero its
    /// incremental joules, which is an artifact, not a measurement).
    fn watts(&self) -> Option<f64> {
        self.gaps.iter().copied().chain(self.initial).chain(self.floor).reduce(f64::min)
    }

    fn observe_job_power(&mut self, w: f64) {
        if w.is_finite() && w >= 0.0 {
            self.floor = Some(self.floor.map_or(w, |f| f.min(w)));
        }
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

/// CPU energy counters (joules): RAPL, or a fake in tests.
pub trait CpuCounters: Send + Sync {
    fn package(&self) -> Option<f64>;
    fn core(&self) -> Option<f64>;
    fn dram(&self) -> Option<f64>;
    /// Counter wrap range, joules.
    fn range(&self) -> f64;
}

impl CpuCounters for joule_energy_rt::rapl::RAPLReader {
    fn package(&self) -> Option<f64> {
        self.read_package_energy().ok()
    }
    fn core(&self) -> Option<f64> {
        self.read_core_energy().ok().flatten()
    }
    fn dram(&self) -> Option<f64> {
        self.read_dram_energy().ok().flatten()
    }
    fn range(&self) -> f64 {
        self.max_energy_range()
    }
}

/// The NVIDIA GPU rail: NVML readings with their own idle baseline.
struct NvmlRail {
    gpu: Box<dyn crate::nvml::GpuEnergy>,
    limit_mw: Option<u32>,
    baseline: Mutex<CounterBaseline>,
}

impl NvmlRail {
    fn new(gpu: Box<dyn crate::nvml::GpuEnergy>) -> Self {
        let limit_mw = gpu.limit_mw();
        // Idle GPU: lowest of three 150 ms windows (the energy counter
        // updates in coarse steps, so short windows can read zero).
        let mut initial: Option<f64> = None;
        for _ in 0..3 {
            let a = crate::nvml::snapshot(gpu.as_ref());
            let t0 = Instant::now();
            std::thread::sleep(Duration::from_millis(150));
            let b = crate::nvml::snapshot(gpu.as_ref());
            if let Some(g) = crate::nvml::between(a, b, t0.elapsed().as_secs_f64(), limit_mw) {
                let w = g.joules / t0.elapsed().as_secs_f64();
                initial = Some(initial.map_or(w, |i: f64| i.min(w)));
            }
        }
        NvmlRail { gpu, limit_mw, baseline: Mutex::new(CounterBaseline { initial, ..Default::default() }) }
    }
}

/// Linux RAPL (+ DRM/hwmon GPU energy, + NVML for NVIDIA GPUs).
struct Rapl {
    reader: Box<dyn CpuCounters>,
    gpu_energy: Option<std::path::PathBuf>,
    nvml: Option<NvmlRail>,
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

/// Why NVML was not used (set once at meter detection), for receipts.
static NVML_STATUS: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// `nvml=<library> device=<name> bus=<id>` or `nvml=none (<why>)`.
pub fn nvml_status() -> &'static str {
    NVML_STATUS.get().map(String::as_str).unwrap_or("nvml=not probed")
}

impl Rapl {
    fn open() -> Option<Self> {
        let reader = joule_energy_rt::rapl::RAPLReader::new().ok()?;
        let nvml = if std::env::var("JOULE_NVML").is_ok_and(|v| v == "0") {
            let _ = NVML_STATUS.set("nvml=none (disabled by JOULE_NVML=0)".into());
            None
        } else {
            match crate::nvml::Nvml::open(crate::sigql_executor::gpu_adapter_id()) {
                Ok(n) => {
                    let _ = NVML_STATUS.set(format!(
                        "nvml={} device={} bus={}",
                        crate::nvml::GpuEnergy::library(&n),
                        crate::nvml::GpuEnergy::name(&n),
                        crate::nvml::GpuEnergy::bus_id(&n)
                    ));
                    Some(Box::new(n) as Box<dyn crate::nvml::GpuEnergy>)
                }
                Err(e) => {
                    let _ = NVML_STATUS.set(format!("nvml=none ({e})"));
                    None
                }
            }
        };
        Self::with(Box::new(reader), drm_gpu_energy(), nvml)
    }

    fn with(
        reader: Box<dyn CpuCounters>,
        gpu_energy: Option<std::path::PathBuf>,
        nvml: Option<Box<dyn crate::nvml::GpuEnergy>>,
    ) -> Option<Self> {
        // Idle: the lowest of five 20 ms windows (one window can catch a
        // burst from process start-up).
        let mut initial = f64::INFINITY;
        for _ in 0..5 {
            let a = reader.package()?;
            let t0 = Instant::now();
            std::thread::sleep(Duration::from_millis(20));
            let b = reader.package()?;
            initial = initial.min(delta(a, b, reader.range()) / t0.elapsed().as_secs_f64());
        }
        Some(Rapl {
            reader,
            gpu_energy,
            nvml: nvml.map(NvmlRail::new),
            baseline: Mutex::new(CounterBaseline { initial: Some(initial), ..Default::default() }),
        })
    }

    fn read(&self) -> Option<Counters> {
        Some(Counters {
            package: self.reader.package()?,
            core: self.reader.core(),
            dram: self.reader.dram(),
            gpu: self.gpu_energy.as_deref().and_then(read_uj),
            nvml: self.nvml.as_ref().map(|n| crate::nvml::snapshot(n.gpu.as_ref())),
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
    /// A counter meter over injected CPU counters and an optional GPU
    /// (fake NVML in tests).
    pub fn counters(cpu: Box<dyn CpuCounters>, gpu: Option<Box<dyn crate::nvml::GpuEnergy>>) -> Option<Meter> {
        Rapl::with(cpu, None, gpu).map(|r| Meter::Rapl(Box::new(RaplMeter(r))))
    }

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
            // Wait for the sample that covers a job's end: real samples are
            // 19-40 ms whatever `-i` says, so never less than 150 ms (with
            // 60 ms some receipts fell back to the estimate on M3/M4).
            let ms = std::env::var("JOULE_POWER_WAIT_MS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or((u64::from(sampler.interval_ms) * 4 + 40).max(150));
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
                if let (Some(n), Some(e)) = (&r.0.nvml, c.nvml.and_then(|s| s.energy_mj)) {
                    lock(&n.baseline).job_started(start, e as f64 / 1000.0);
                }
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
        Span { start, counters, job, holes: Vec::new(), verified: Vec::new(), pending: Vec::new(), end_counters: None }
    }

    /// Mark `span` finished at `end` (call right when the work ends) while
    /// its energy is read later with [`Self::finish_at`]: closes the sampler
    /// job, and snapshots counter meters so later work is not charged.
    pub fn close(&self, span: &mut Span, end: Instant) {
        span.close(end);
        if let Meter::Rapl(r) = self {
            span.end_counters = r.0.read();
        }
    }

    /// Run `f`, a verification of `span`'s result (e.g. the CPU
    /// cross-check of a GPU sum), as its own `kind=verify` meter job: its
    /// time and energy are excluded from `span` (sampler samples are split
    /// between the two) and reported as `verify_joules`.
    pub fn verify<T>(&self, span: &mut Span, f: impl FnOnce() -> T) -> T {
        let v = self.begin();
        let out = f();
        let end = Instant::now();
        if let Some((sampler, id)) = span.job {
            sampler.exclude_from_job(id, v.start, end);
        }
        span.holes.push((v.start, end));
        match self {
            // Reading a sampler window waits for its covering sample: do
            // that after the job, not inside it.
            Meter::Sampler { .. } => {
                v.close(end);
                span.pending.push((Box::new(v), end));
            }
            _ => {
                let e = self.finish_at(v, end, Duration::ZERO, "cpu", VERIFY_ESTIMATE_W);
                span.verified.push(e);
            }
        }
        out
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
        let holes = std::mem::take(&mut span.holes);
        let verified = std::mem::take(&mut span.verified);
        let pending = std::mem::take(&mut span.pending);
        let mut e = self.finish_job(&mut span, end, skip, device, estimate_watts, &holes, &verified);
        if !holes.is_empty() {
            let mut vs: Vec<JobEnergy> = verified;
            for (v, vend) in pending {
                vs.push(self.finish_at(*v, vend, Duration::ZERO, "cpu", VERIFY_ESTIMATE_W));
            }
            e.verify_joules = Some(vs.iter().map(|v| v.joules).sum());
            e.verify_seconds = Some(vs.iter().map(|v| v.seconds).sum());
        }
        e
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_job(
        &self,
        span: &mut Span,
        end: Instant,
        skip: Duration,
        device: &str,
        estimate_watts: f64,
        holes: &[(Instant, Instant)],
        verified: &[JobEnergy],
    ) -> JobEnergy {
        let start = (span.start + skip).min(end);
        let hole_s: f64 = holes
            .iter()
            .map(|(a, b)| b.min(&end).saturating_duration_since(*a.max(&start)).as_secs_f64())
            .sum();
        let seconds = (end.saturating_duration_since(start).as_secs_f64() - hole_s).max(1e-9);
        // Verification energy on the CPU-side counters (already read).
        let verify_cpu: f64 = verified
            .iter()
            .map(|v| v.joules - if v.source.contains("nvml") { v.gpu_watts.unwrap_or(0.0) * v.seconds } else { 0.0 })
            .sum();
        let estimate = || JobEnergy::estimate(device, estimate_watts, seconds);
        match self {
            Meter::Sampler { sampler, wait } => match span.job.take() {
                Some((_, id)) => sampler_energy(sampler, id, start, end, *wait).unwrap_or_else(estimate),
                None => estimate(),
            },
            Meter::Rapl(r) => {
                let (Some(a), Some(b)) = (span.counters, span.end_counters.or_else(|| r.0.read())) else {
                    return estimate();
                };
                let mut base = lock(&r.0.baseline);
                base.job_ended(end, b.package);
                let range = r.0.reader.range();
                // The whole span (setup included) is what the counter saw;
                // charge the job's share by time.
                let total = end.saturating_duration_since(span.start).as_secs_f64().max(1e-9);
                let all_hole_s: f64 = holes.iter().map(|(a, b)| b.saturating_duration_since(*a).as_secs_f64()).sum();
                let share = seconds / (total - all_hole_s).max(1e-9);
                // Verification read its own counters: take it off the
                // window first, then charge the job its share by time.
                let pkg_all = delta(a.package, b.package, range);
                let dram_all = a.dram.zip(b.dram).map(|(x, y)| delta(x, y, range));
                let drm_all = a.gpu.zip(b.gpu).map(|(x, y)| (y - x).max(0.0));
                let window = pkg_all + dram_all.unwrap_or(0.0) + drm_all.unwrap_or(0.0);
                let keep = if window > 0.0 { ((window - verify_cpu) / window).clamp(0.0, 1.0) } else { 1.0 };
                let pkg = pkg_all * keep * share;
                let dram = dram_all.map(|j| j * keep * share);
                let core = a.core.zip(b.core).map(|(x, y)| delta(x, y, range) * keep * share);
                let drm = drm_all.map(|j| j * keep * share);
                let cpu_side = pkg + dram.unwrap_or(0.0) + drm.unwrap_or(0.0);
                base.observe_job_power(cpu_side / seconds);
                let cpu_base = base.watts();
                drop(base);
                let cpu_inc = (cpu_side - cpu_base.unwrap_or(0.0) * seconds).max(0.0);
                // NVML rail: its own counter and baseline.
                let mut notes = None;
                let nv = match (&r.0.nvml, a.nvml, b.nvml) {
                    (Some(n), Some(x), Some(y)) => {
                        let g = crate::nvml::between(x, y, total, n.limit_mw);
                        if g.is_none() {
                            notes = Some("NVML: no usable reading (all rejected or unsupported)".to_string());
                        }
                        g.map(|g| {
                            let mut nb = lock(&n.baseline);
                            if let Some(e) = y.energy_mj {
                                nb.job_ended(end, e as f64 / 1000.0);
                            }
                            let j = g.joules * seconds / total;
                            nb.observe_job_power(j / seconds);
                            let gb = nb.watts();
                            if g.note.is_some() {
                                notes = g.note.clone();
                            }
                            (j, (j - gb.unwrap_or(0.0) * seconds).max(0.0), gb, g.source)
                        })
                    }
                    _ => None,
                };
                let gpu_j = nv.map(|v| v.0).or(drm);
                let joules = cpu_side + nv.map_or(0.0, |v| v.0);
                let incremental = cpu_inc + nv.map_or(0.0, |v| v.1);
                let gpu_rail = match (nv, drm) {
                    (Some(v), _) => v.3,
                    (None, Some(_)) => "drm-hwmon",
                    (None, None) => "none",
                };
                JobEnergy {
                    source: if nv.is_some() { "rapl+nvml" } else { "rapl" }.into(),
                    // RAPL counters update about every millisecond; NVML's
                    // energy counter is coarser (tens of ms).
                    confidence: if seconds >= 0.002 && (nv.is_none() || seconds >= 0.1) {
                        "measured"
                    } else {
                        "interpolated"
                    }
                    .into(),
                    seconds,
                    joules,
                    incremental_joules: incremental,
                    watts: joules / seconds,
                    cpu_watts: Some(core.unwrap_or(pkg) / seconds),
                    gpu_watts: gpu_j.map(|j| j / seconds),
                    ane_watts: None,
                    dram_watts: dram.map(|j| j / seconds),
                    baseline_watts: cpu_base.map(|c| c + nv.and_then(|v| v.2).unwrap_or(0.0)),
                    rails: format!("cpu=rapl gpu={gpu_rail}"),
                    cpu_incremental_joules: Some(cpu_inc),
                    gpu_incremental_joules: nv.map(|v| v.1),
                    notes,
                    ..Default::default()
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
                let joules = ((b - a.package).max(0.0) - verify_cpu).max(0.0) * seconds / total;
                base.observe_job_power(joules / seconds);
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
                    rails: "cpu=emi".into(),
                    ..Default::default()
                }
            }
            Meter::Estimate => estimate(),
        }
    }
}

/// Estimate watts for a verification job when nothing is measured.
const VERIFY_ESTIMATE_W: f64 = 5.0;

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
        rails: "cpu=powermetrics gpu=powermetrics ane=powermetrics".into(),
        cpu_incremental_joules: Some(w.incremental.cpu_w),
        gpu_incremental_joules: Some(w.incremental.gpu_w),
        ..Default::default()
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
        // A job drawing less than the start-up reading lowers the baseline,
        // so incremental joules are not clamped to zero.
        let mut b = CounterBaseline { initial: Some(66.0), ..Default::default() };
        b.observe_job_power(35.0);
        assert_eq!(b.watts(), Some(35.0));
    }

    /// Package counter advancing at `watts` of wall time.
    struct FakeRapl {
        start: Instant,
        watts: f64,
    }

    impl CpuCounters for FakeRapl {
        fn package(&self) -> Option<f64> {
            Some(self.start.elapsed().as_secs_f64() * self.watts)
        }
        fn core(&self) -> Option<f64> {
            None
        }
        fn dram(&self) -> Option<f64> {
            None
        }
        fn range(&self) -> f64 {
            1e9
        }
    }

    #[test]
    fn rapl_plus_fake_nvml_reports_both_rails_and_rejects_absurd_power() {
        let now = Instant::now();
        let gpu = crate::nvml::FakeNvml {
            start: now,
            watts: 12.0,
            // The 590 W instantaneous draw jetson-hub's RTX 4050 reports.
            power_mw: Some(590_010),
            limit_mw: Some(35_000),
            counter: true,
        };
        let meter = Meter::counters(Box::new(FakeRapl { start: now, watts: 20.0 }), Some(Box::new(gpu))).expect("meter");
        let span = meter.begin();
        std::thread::sleep(Duration::from_millis(120));
        let e = meter.finish(span, Duration::ZERO, "gpu", 5.0);
        assert_eq!(e.source, "rapl+nvml");
        assert_eq!(e.rails, "cpu=rapl gpu=nvml-energy");
        let gw = e.gpu_watts.expect("gpu_w");
        assert!((gw - 12.0).abs() < 1.5, "{e:?}");
        assert!(e.cpu_watts.is_some_and(|w| (w - 20.0).abs() < 2.0), "{e:?}");
        // Flat fakes: both rails sit at their baselines, so increments ~0
        // and the total is the sum of both rails.
        assert!((e.watts - 32.0).abs() < 3.0, "{e:?}");
        assert!(e.gpu_incremental_joules.is_some() && e.cpu_incremental_joules.is_some());
        assert!(e.incremental_joules < 0.5, "{e:?}");

        // No energy counter: the absurd power sample is rejected with a
        // reason and the GPU rail is left out (CPU still measured).
        let gpu = crate::nvml::FakeNvml {
            start: now,
            watts: 0.0,
            power_mw: Some(590_010),
            limit_mw: Some(35_000),
            counter: false,
        };
        let meter = Meter::counters(Box::new(FakeRapl { start: now, watts: 20.0 }), Some(Box::new(gpu))).expect("meter");
        let e = meter.finish(meter.begin(), Duration::ZERO, "gpu", 5.0);
        assert_eq!(e.source, "rapl");
        assert_eq!(e.rails, "cpu=rapl gpu=none");
        assert!(e.notes.as_deref().unwrap_or("").contains("NVML"), "{e:?}");
        assert!(e.is_measured());
    }

    #[test]
    fn verification_is_not_billed_to_the_job() {
        // Sampler: a 10 W excess over the whole window; half of the job's
        // window is a CPU cross-check registered as its own verify job.
        let meter = Meter::Sampler { sampler: injected(11.0, 0.0), wait: Duration::ZERO };
        let mut span = meter.begin();
        std::thread::sleep(Duration::from_millis(60));
        let ok = meter.verify(&mut span, || {
            std::thread::sleep(Duration::from_millis(60));
            true
        });
        assert!(ok);
        let total = span.start.elapsed().as_secs_f64();
        let e = meter.finish(span, Duration::ZERO, "gpu", 5.0);
        let vj = e.verify_joules.expect("verify joules on the receipt");
        let vs = e.verify_seconds.expect("verify seconds");
        assert!(vs >= 0.06 && e.seconds < total - 0.055, "{e:?}");
        // Job + verify cover the window once: nothing double-billed.
        assert!((e.seconds + vs - total).abs() < 0.005, "{e:?}");
        assert!(e.incremental_joules < 10.0 * (e.seconds + 0.045), "{e:?}");
        assert!(vj > 0.0);
        // Without verify the same window is billed whole to the job.
        let span = meter.begin();
        std::thread::sleep(Duration::from_millis(120));
        let whole = meter.finish(span, Duration::ZERO, "gpu", 5.0);
        assert!(whole.verify_joules.is_none());
        assert!(whole.joules > e.joules + 0.3 * vj, "{whole:?} vs {e:?}");

        // Counters (RAPL): verify reads its own counters; the job keeps the rest.
        let now = Instant::now();
        let meter = Meter::counters(Box::new(FakeRapl { start: now, watts: 20.0 }), None).expect("meter");
        let mut span = meter.begin();
        std::thread::sleep(Duration::from_millis(40));
        meter.verify(&mut span, || std::thread::sleep(Duration::from_millis(40)));
        let e = meter.finish(span, Duration::ZERO, "gpu", 5.0);
        let vj = e.verify_joules.expect("verify joules");
        assert!((vj - 20.0 * e.verify_seconds.unwrap_or(0.0)).abs() < 0.1, "{e:?}");
        assert!((e.joules - 20.0 * e.seconds).abs() < 0.1, "{e:?}");
        assert!(e.seconds < 0.06, "{e:?}");
    }

    #[test]
    fn closed_counter_span_is_not_charged_for_later_work() {
        let now = Instant::now();
        let meter = Meter::counters(Box::new(FakeRapl { start: now, watts: 20.0 }), None).expect("meter");
        let mut span = meter.begin();
        std::thread::sleep(Duration::from_millis(20));
        let end = Instant::now();
        meter.close(&mut span, end);
        // Other work (e.g. the GPU lane) runs before the energy is read.
        std::thread::sleep(Duration::from_millis(80));
        let e = meter.finish_at(span, end, Duration::ZERO, "cpu", 5.0);
        assert!((e.joules - 20.0 * e.seconds).abs() < 0.05, "{e:?}");
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
