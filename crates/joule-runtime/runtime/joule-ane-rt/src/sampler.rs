//! Shared per-rail power sampler (CPU, GPU, ANE) for receipts and routing.
//!
//! One long-lived `powermetrics --samplers cpu_power,gpu_power` process per
//! process (macOS, only through `sudo -n`, never a prompt, never a stored
//! password) feeds a ring buffer of timestamped samples. A job's energy is
//! the integral of the samples over its `[start, end]` window:
//!
//! * the idle share is `baseline watts x job seconds`;
//! * each overlapping sample's energy above the baseline is apportioned
//!   among the registered jobs that overlap that sample, by overlap time
//!   ([`PowerRing::integrate_job`]). A lone 2 ms job inside a 35 ms sample
//!   is charged that sample's whole excess, not 2/35 of it (the burst is
//!   the job's), and two jobs in one sample split it. Such sub-interval
//!   jobs are `interpolated`;
//! * a job covered by whole samples is `measured`;
//! * no overlapping sample means the caller falls back to an estimate.
//!
//! The idle baseline is the per-rail 20th percentile of the samples of the
//! last [`BASELINE_HORIZON_S`] seconds, measured once and refreshed every
//! [`BASELINE_REFRESH_S`] seconds. Receipts carry both the total rail joules
//! and the incremental joules above that baseline.
//!
//! The ring, the integration, and the baseline are platform independent
//! (and tested with injected samples); only the powermetrics feed is macOS.

use std::collections::VecDeque;
use std::sync::Mutex;

/// Mean power per rail over one interval, watts.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Rails {
    pub cpu_w: f64,
    pub gpu_w: f64,
    pub ane_w: f64,
}

impl Rails {
    pub fn total(&self) -> f64 {
        self.cpu_w + self.gpu_w + self.ane_w
    }

    fn scaled(&self, k: f64) -> Rails {
        Rails { cpu_w: self.cpu_w * k, gpu_w: self.gpu_w * k, ane_w: self.ane_w * k }
    }

    fn add(&mut self, o: &Rails) {
        self.cpu_w += o.cpu_w;
        self.gpu_w += o.gpu_w;
        self.ane_w += o.ane_w;
    }

    /// Per-rail power above `base`, floored at zero.
    pub fn above(&self, base: &Rails) -> Rails {
        Rails {
            cpu_w: (self.cpu_w - base.cpu_w).max(0.0),
            gpu_w: (self.gpu_w - base.gpu_w).max(0.0),
            ane_w: (self.ane_w - base.ane_w).max(0.0),
        }
    }
}

/// One sample: the interval `[start, end]` (seconds on the sampler clock)
/// and the mean rail power over it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RailSample {
    pub start: f64,
    pub end: f64,
    pub rails: Rails,
    /// `Combined Power (CPU + GPU + ANE)` when printed.
    pub combined_w: Option<f64>,
}

/// How a window's energy was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Confidence {
    /// Covered by samples and at least one sample interval long.
    Measured,
    /// Shorter than a sample interval (apportioned from the overlapping
    /// samples) or partly covered (scaled to the window).
    Interpolated,
}

impl Confidence {
    pub fn label(&self) -> &'static str {
        match self {
            Confidence::Measured => "measured",
            Confidence::Interpolated => "interpolated",
        }
    }
}

/// Energy of one `[start, end]` window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowEnergy {
    pub seconds: f64,
    /// Rail joules over the window (`J` per rail, stored as `Rails` of joules).
    pub joules: Rails,
    /// Rail joules above the idle baseline.
    pub incremental: Rails,
    /// Mean rail watts over the window.
    pub watts: Rails,
    /// Baseline used (watts per rail).
    pub baseline: Rails,
    /// Fraction of the window covered by samples (0, 1].
    pub coverage: f64,
    /// Samples that overlapped the window.
    pub samples: usize,
    pub confidence: Confidence,
}

/// Idle baseline percentile.
pub const BASELINE_PERCENTILE: f64 = 0.20;
/// Seconds of history the baseline is taken from.
pub const BASELINE_HORIZON_S: f64 = 60.0;
/// Seconds after which the baseline is recomputed.
pub const BASELINE_REFRESH_S: f64 = 30.0;
/// Minimum samples for a baseline.
pub const BASELINE_MIN_SAMPLES: usize = 5;

/// Ring buffer of rail samples with window integration.
#[derive(Debug)]
pub struct PowerRing {
    samples: VecDeque<RailSample>,
    capacity: usize,
    baseline: Option<(f64, Rails)>,
    /// Registered job windows `(id, start, end)`; `end = None` while running.
    jobs: VecDeque<(u64, f64, Option<f64>)>,
    next_job: u64,
}

/// Registered job windows kept for apportioning.
const JOB_CAPACITY: usize = 4096;

fn overlap(a0: f64, a1: f64, b0: f64, b1: f64) -> f64 {
    (a1.min(b1) - a0.max(b0)).max(0.0)
}

impl PowerRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            baseline: None,
            jobs: VecDeque::new(),
            next_job: 0,
        }
    }

    pub fn push(&mut self, sample: RailSample) {
        if self.samples.len() == self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(sample);
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// End of the newest sample.
    pub fn latest_end(&self) -> Option<f64> {
        self.samples.back().map(|s| s.end)
    }

    /// Median sample length, seconds.
    pub fn typical_interval(&self) -> Option<f64> {
        let mut d: Vec<f64> = self.samples.iter().map(|s| s.end - s.start).filter(|d| *d > 0.0).collect();
        if d.is_empty() {
            return None;
        }
        d.sort_by(f64::total_cmp);
        Some(d[d.len() / 2])
    }

    /// Per-rail percentile over the samples ending in `[now - horizon, now]`.
    pub fn percentile_baseline(&self, now: f64, horizon: f64, q: f64) -> Option<Rails> {
        let recent: Vec<&RailSample> =
            self.samples.iter().filter(|s| s.end <= now && s.end >= now - horizon).collect();
        if recent.len() < BASELINE_MIN_SAMPLES {
            return None;
        }
        let pick = |f: &dyn Fn(&RailSample) -> f64| {
            let mut v: Vec<f64> = recent.iter().map(|s| f(s)).collect();
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * q).round() as usize]
        };
        Some(Rails {
            cpu_w: pick(&|s| s.rails.cpu_w),
            gpu_w: pick(&|s| s.rails.gpu_w),
            ane_w: pick(&|s| s.rails.ane_w),
        })
    }

    /// The idle baseline at `now`: computed once, refreshed every
    /// [`BASELINE_REFRESH_S`]. Zero until enough samples exist (then
    /// incremental equals total, which the caller can see in `baseline`).
    pub fn baseline(&mut self, now: f64) -> Rails {
        let stale = match self.baseline {
            Some((at, _)) => now - at >= BASELINE_REFRESH_S,
            None => true,
        };
        if let Some(b) = stale
            .then(|| self.percentile_baseline(now, BASELINE_HORIZON_S, BASELINE_PERCENTILE))
            .flatten()
        {
            self.baseline = Some((now, b));
        }
        self.baseline.map(|(_, b)| b).unwrap_or_default()
    }

    /// Set the baseline explicitly (tests, or a deliberate idle measurement).
    pub fn set_baseline(&mut self, at: f64, rails: Rails) {
        self.baseline = Some((at, rails));
    }

    /// Integrate `[start, end]` against the samples, subtracting `baseline`.
    /// `None` when no sample overlaps the window.
    pub fn integrate_with(&self, start: f64, end: f64, baseline: Rails) -> Option<WindowEnergy> {
        let (start, end) = if end < start { (end, start) } else { (start, end) };
        let seconds = end - start;
        let mut joules = Rails::default();
        let mut incremental = Rails::default();
        let mut covered = 0.0;
        let mut used = 0usize;
        let mut point: Option<Rails> = None;
        for s in &self.samples {
            if s.end <= start || s.start >= end {
                // A zero-length window inside a sample still reads that sample.
                if seconds == 0.0 && s.start <= start && start <= s.end {
                    point = Some(s.rails);
                }
                continue;
            }
            let ov = s.end.min(end) - s.start.max(start);
            if ov <= 0.0 {
                continue;
            }
            joules.add(&s.rails.scaled(ov));
            incremental.add(&s.rails.above(&baseline).scaled(ov));
            covered += ov;
            used += 1;
        }
        if seconds == 0.0 {
            let w = point?;
            return Some(WindowEnergy {
                seconds,
                joules: Rails::default(),
                incremental: Rails::default(),
                watts: w,
                baseline,
                coverage: 1.0,
                samples: 1,
                confidence: Confidence::Interpolated,
            });
        }
        if used == 0 || covered <= 0.0 {
            return None;
        }
        let coverage = (covered / seconds).min(1.0);
        // Gaps (sampler hiccup, window past the newest sample) are filled at
        // the mean of the covered part.
        let fill = 1.0 / coverage;
        let joules = joules.scaled(fill);
        let incremental = incremental.scaled(fill);
        let watts = joules.scaled(1.0 / seconds);
        let interval = self.typical_interval().unwrap_or(seconds);
        let confidence = if coverage >= 0.95 && seconds >= interval {
            Confidence::Measured
        } else {
            Confidence::Interpolated
        };
        Some(WindowEnergy { seconds, joules, incremental, watts, baseline, coverage, samples: used, confidence })
    }

    /// Register a job that started at `start`; returns its id.
    pub fn begin_job(&mut self, start: f64) -> u64 {
        if self.jobs.len() == JOB_CAPACITY {
            self.jobs.pop_front();
        }
        let id = self.next_job;
        self.next_job += 1;
        self.jobs.push_back((id, start, None));
        id
    }

    /// Forget a job that never finished (it would otherwise count as
    /// running in every later sample).
    pub fn drop_job(&mut self, id: u64) {
        self.jobs.retain(|j| j.0 != id || j.2.is_some());
    }

    /// Set a registered job's window (start may move forward past setup).
    pub fn end_job(&mut self, id: u64, start: f64, end: f64) {
        if let Some(j) = self.jobs.iter_mut().find(|j| j.0 == id) {
            j.1 = start;
            j.2 = Some(end);
        }
    }

    /// Energy of registered job `id` over `[start, end]`: idle share
    /// `baseline x seconds` plus, per overlapping sample, the sample's energy
    /// above the baseline times this job's share of the job-time in that
    /// sample (jobs still running count up to the sample end). `None` when
    /// no sample overlaps.
    pub fn integrate_job_with(&self, id: u64, start: f64, end: f64, baseline: Rails) -> Option<WindowEnergy> {
        let seconds = (end - start).max(0.0);
        if seconds == 0.0 {
            return self.integrate_with(start, end, baseline);
        }
        let mut incremental = Rails::default();
        let mut covered = 0.0;
        let mut used = 0usize;
        for s in &self.samples {
            let mine = overlap(start, end, s.start, s.end);
            if mine <= 0.0 {
                continue;
            }
            let others: f64 = self
                .jobs
                .iter()
                .filter(|j| j.0 != id)
                .map(|j| overlap(j.1, j.2.unwrap_or(s.end), s.start, s.end))
                .sum();
            let share = mine / (mine + others);
            incremental.add(&s.rails.above(&baseline).scaled((s.end - s.start) * share));
            covered += mine;
            used += 1;
        }
        if used == 0 {
            return None;
        }
        let coverage = (covered / seconds).min(1.0);
        let incremental = incremental.scaled(1.0 / coverage);
        let mut joules = baseline.scaled(seconds);
        joules.add(&incremental);
        let interval = self.typical_interval().unwrap_or(seconds);
        let confidence = if coverage >= 0.95 && seconds >= interval {
            Confidence::Measured
        } else {
            Confidence::Interpolated
        };
        Some(WindowEnergy {
            seconds,
            joules,
            incremental,
            watts: joules.scaled(1.0 / seconds),
            baseline,
            coverage,
            samples: used,
            confidence,
        })
    }

    /// [`Self::integrate_job_with`] using the (refreshed) idle baseline.
    pub fn integrate_job(&mut self, id: u64, start: f64, end: f64) -> Option<WindowEnergy> {
        let base = self.baseline(end);
        self.integrate_job_with(id, start, end, base)
    }

    /// [`Self::integrate_with`] using the (refreshed) idle baseline.
    pub fn integrate(&mut self, start: f64, end: f64) -> Option<WindowEnergy> {
        let base = self.baseline(end);
        self.integrate_with(start, end, base)
    }
}

/// Streaming parser of `powermetrics` text (`cpu_power`, `gpu_power`).
///
/// A sample is complete at its `Combined Power` line, or at the next sample
/// header when no combined line was printed.
#[derive(Debug, Default)]
pub struct RailParser {
    elapsed_ms: Option<f64>,
    cpu: Option<f64>,
    gpu: Option<f64>,
    ane: Option<f64>,
    combined: Option<f64>,
    done: bool,
}

/// A parsed sample before it is placed on a clock.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParsedSample {
    pub elapsed_ms: f64,
    pub rails: Rails,
    pub combined_w: Option<f64>,
}

fn milliwatts(rest: &str) -> Option<f64> {
    let rest = rest.trim();
    let num: String = rest.chars().take_while(|c| c.is_ascii_digit() || *c == '.').collect();
    let value = num.parse::<f64>().ok()?;
    Some(if rest.ends_with("mW") { value } else if rest.ends_with('W') { value * 1000.0 } else { value })
}

/// `(20.51ms elapsed)` from a sample header.
pub fn header_elapsed_ms(line: &str) -> Option<f64> {
    let line = line.trim();
    if !line.starts_with("*** Sampled system activity") {
        return None;
    }
    let end = line.rfind("ms elapsed)")?;
    let head = &line[..end];
    let start = head.rfind('(')?;
    head[start + 1..].trim().parse::<f64>().ok()
}

impl RailParser {
    fn take(&mut self) -> Option<ParsedSample> {
        let ms = self.elapsed_ms?;
        if self.done || (self.cpu.is_none() && self.gpu.is_none() && self.ane.is_none()) {
            return None;
        }
        self.done = true;
        Some(ParsedSample {
            elapsed_ms: ms,
            rails: Rails {
                cpu_w: self.cpu.unwrap_or(0.0) / 1000.0,
                gpu_w: self.gpu.unwrap_or(0.0) / 1000.0,
                ane_w: self.ane.unwrap_or(0.0) / 1000.0,
            },
            combined_w: self.combined.map(|mw| mw / 1000.0),
        })
    }

    /// Feed one line; returns a sample when one completes, and whether the
    /// line was a header (the caller timestamps headers).
    pub fn feed(&mut self, line: &str) -> (Option<ParsedSample>, bool) {
        if let Some(ms) = header_elapsed_ms(line) {
            let prev = self.take();
            *self = RailParser { elapsed_ms: Some(ms), ..Default::default() };
            return (prev, true);
        }
        if self.done {
            return (None, false);
        }
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("CPU Power:") {
            self.cpu = milliwatts(rest);
        } else if let Some(rest) = line.strip_prefix("GPU Power:") {
            if self.gpu.is_none() {
                self.gpu = milliwatts(rest);
            }
        } else if let Some(rest) = line.strip_prefix("ANE Power:") {
            self.ane = milliwatts(rest);
        } else if let Some(rest) = line.strip_prefix("Combined Power") {
            self.combined = rest.split_once(':').and_then(|(_, v)| milliwatts(v));
            return (self.take(), false);
        }
        (None, false)
    }

    /// Flush a pending sample at end of stream.
    pub fn finish(&mut self) -> Option<ParsedSample> {
        self.take()
    }
}

/// Parse a whole text into samples (no clock).
pub fn parse_rails(text: &str) -> Vec<ParsedSample> {
    let mut p = RailParser::default();
    let mut out: Vec<ParsedSample> = text.lines().filter_map(|l| p.feed(l).0).collect();
    out.extend(p.finish());
    out
}

/// The long-lived sampler: a ring fed by one `powermetrics` stream.
pub struct PowerSampler {
    ring: Mutex<PowerRing>,
    epoch: std::time::Instant,
    /// Requested `-i` interval, ms.
    pub interval_ms: u32,
}

/// Ring capacity: ~10 minutes at 50 ms per sample.
pub const RING_CAPACITY: usize = 12_000;

impl PowerSampler {
    /// A sampler with no feed (tests / injected samples).
    pub fn detached(interval_ms: u32) -> Self {
        Self { ring: Mutex::new(PowerRing::new(RING_CAPACITY)), epoch: std::time::Instant::now(), interval_ms }
    }

    /// Seconds on the sampler clock.
    pub fn at(&self, t: std::time::Instant) -> f64 {
        t.saturating_duration_since(self.epoch).as_secs_f64()
    }

    pub fn now(&self) -> f64 {
        self.at(std::time::Instant::now())
    }

    pub fn with_ring<R>(&self, f: impl FnOnce(&mut PowerRing) -> R) -> R {
        let mut ring = self.ring.lock().unwrap_or_else(|p| p.into_inner());
        f(&mut ring)
    }

    /// Wait (bounded) until a sample ends at or after `t`.
    pub fn wait_until_covered(&self, t: f64, timeout: std::time::Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if self.with_ring(|r| r.latest_end()).is_some_and(|e| e >= t) {
                return true;
            }
            if std::time::Instant::now() >= deadline {
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
    }

    /// Integrate `[start, end]` (Instants), waiting at most `wait` for the
    /// sample that covers `end`.
    pub fn window(
        &self,
        start: std::time::Instant,
        end: std::time::Instant,
        wait: std::time::Duration,
    ) -> Option<WindowEnergy> {
        let (s, e) = (self.at(start), self.at(end));
        self.wait_until_covered(e, wait);
        self.with_ring(|r| r.integrate(s, e))
    }

    /// Register a job starting now; pass the id to [`Self::job_window`].
    pub fn begin_job(&self, start: std::time::Instant) -> u64 {
        let t = self.at(start);
        self.with_ring(|r| r.begin_job(t))
    }

    /// Close job `id` at `[start, end]` without reading its energy yet.
    pub fn end_job(&self, id: u64, start: std::time::Instant, end: std::time::Instant) {
        let (s, e) = (self.at(start), self.at(end));
        self.with_ring(|r| r.end_job(id, s, e));
    }

    /// Forget a job that never finished.
    pub fn drop_job(&self, id: u64) {
        self.with_ring(|r| r.drop_job(id));
    }

    /// Energy of registered job `id` over `[start, end]` (apportioned among
    /// overlapping jobs), waiting at most `wait` for the covering sample.
    pub fn job_window(
        &self,
        id: u64,
        start: std::time::Instant,
        end: std::time::Instant,
        wait: std::time::Duration,
    ) -> Option<WindowEnergy> {
        let (s, e) = (self.at(start), self.at(end));
        self.with_ring(|r| r.end_job(id, s, e));
        self.wait_until_covered(e, wait);
        self.with_ring(|r| r.integrate_job(id, s, e))
    }

    /// Start the process-wide sampler (macOS with passwordless `sudo -n`
    /// only). `JOULE_POWER_SAMPLER=off` disables it;
    /// `JOULE_POWERMETRICS_INTERVAL_MS` sets the interval (default 20).
    pub fn global() -> Option<&'static PowerSampler> {
        static GLOBAL: std::sync::OnceLock<Option<&'static PowerSampler>> = std::sync::OnceLock::new();
        *GLOBAL.get_or_init(|| {
            if std::env::var("JOULE_POWER_SAMPLER").is_ok_and(|v| v == "off" || v == "0") {
                return None;
            }
            if !crate::power::powermetrics_available() {
                return None;
            }
            let interval_ms = std::env::var("JOULE_POWERMETRICS_INTERVAL_MS")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|v| *v >= 5)
                .unwrap_or(20);
            let sampler: &'static PowerSampler = Box::leak(Box::new(PowerSampler::detached(interval_ms)));
            sampler.spawn_feed().then_some(sampler)
        })
    }

    #[cfg(target_os = "macos")]
    fn spawn_feed(&'static self) -> bool {
        use std::io::BufRead;
        use std::process::{Command, Stdio};
        let child = Command::new("/usr/bin/sudo")
            .args([
                "-n",
                "/usr/bin/powermetrics",
                "--samplers",
                "cpu_power,gpu_power",
                "-i",
                &self.interval_ms.to_string(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn();
        let Ok(mut child) = child else { return false };
        let Some(stdout) = child.stdout.take() else {
            let _ = child.kill();
            return false;
        };
        let spawned = std::thread::Builder::new().name("joule-powermetrics".into()).spawn(move || {
            let mut parser = RailParser::default();
            let mut header_at: Option<f64> = None;
            let mut prev_end: Option<f64> = None;
            let reader = std::io::BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(line) = line else { break };
                let arrived = self.now();
                let (done, header) = parser.feed(&line);
                if let Some(p) = done {
                    if let Some(end) = header_at {
                        let mut start = end - p.elapsed_ms / 1000.0;
                        // Samples are back to back; do not overlap the previous one.
                        if let Some(pe) = prev_end {
                            start = start.max(pe);
                        }
                        if end > start {
                            self.with_ring(|r| {
                                r.push(RailSample { start, end, rails: p.rails, combined_w: p.combined_w })
                            });
                            prev_end = Some(end);
                        }
                    }
                }
                if header {
                    // powermetrics prints a sample right after its interval
                    // ends, so the header's arrival is the interval end.
                    header_at = Some(arrived);
                }
            }
            let _ = child.kill();
            let _ = child.wait();
        });
        spawned.is_ok()
    }

    #[cfg(not(target_os = "macos"))]
    fn spawn_feed(&'static self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(start: f64, end: f64, cpu: f64, gpu: f64, ane: f64) -> RailSample {
        RailSample { start, end, rails: Rails { cpu_w: cpu, gpu_w: gpu, ane_w: ane }, combined_w: None }
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    /// 10 idle samples (2 W CPU, 0.1 W GPU) of 20 ms, then a burst.
    fn ring() -> PowerRing {
        let mut r = PowerRing::new(64);
        for i in 0..10 {
            let t = i as f64 * 0.02;
            r.push(sample(t, t + 0.02, 2.0, 0.1, 0.0));
        }
        // 0.20..0.24: GPU job at 12 W GPU, CPU 3 W.
        r.push(sample(0.20, 0.22, 3.0, 12.0, 0.0));
        r.push(sample(0.22, 0.24, 3.0, 12.0, 0.0));
        // 0.24..0.26: ANE at 1.5 W.
        r.push(sample(0.24, 0.26, 2.0, 0.1, 1.5));
        r
    }

    #[test]
    fn integrates_whole_samples_deterministically() {
        let r = ring();
        let base = Rails { cpu_w: 2.0, gpu_w: 0.1, ane_w: 0.0 };
        let w = r.integrate_with(0.20, 0.24, base).expect("covered");
        assert_eq!(w.confidence, Confidence::Measured);
        assert_eq!(w.samples, 2);
        assert!(close(w.joules.cpu_w, 3.0 * 0.04));
        assert!(close(w.joules.gpu_w, 12.0 * 0.04));
        assert!(close(w.incremental.cpu_w, 1.0 * 0.04));
        assert!(close(w.incremental.gpu_w, 11.9 * 0.04));
        assert!(close(w.incremental.total(), (1.0 + 11.9) * 0.04));
        assert!(close(w.watts.gpu_w, 12.0));
    }

    #[test]
    fn short_jobs_are_apportioned_and_labelled_interpolated() {
        let r = ring();
        let base = Rails { cpu_w: 2.0, gpu_w: 0.1, ane_w: 0.0 };
        // 5 ms inside one GPU sample.
        let w = r.integrate_with(0.205, 0.210, base).expect("covered");
        assert_eq!(w.confidence, Confidence::Interpolated);
        assert!(close(w.joules.gpu_w, 12.0 * 0.005));
        assert!(close(w.incremental.gpu_w, 11.9 * 0.005));
        // 10 ms straddling the GPU and ANE samples: each contributes its share.
        let w = r.integrate_with(0.235, 0.245, base).expect("covered");
        assert_eq!(w.samples, 2);
        assert_eq!(w.confidence, Confidence::Interpolated);
        assert!(close(w.joules.gpu_w, 12.0 * 0.005 + 0.1 * 0.005));
        assert!(close(w.joules.ane_w, 1.5 * 0.005));
        assert!(close(w.incremental.ane_w, 1.5 * 0.005));
    }

    #[test]
    fn sub_interval_jobs_split_the_sample_excess() {
        let mut r = ring();
        let base = Rails { cpu_w: 2.0, gpu_w: 0.1, ane_w: 0.0 };
        // GPU sample 0.20..0.22: excess (1 W CPU + 11.9 W GPU) x 20 ms.
        let excess = 12.9 * 0.02;
        // A lone 5 ms job is charged the whole excess, plus idle for 5 ms.
        let a = r.begin_job(0.205);
        r.end_job(a, 0.205, 0.210);
        let w = r.integrate_job_with(a, 0.205, 0.210, base).expect("covered");
        assert_eq!(w.confidence, Confidence::Interpolated);
        assert!(close(w.incremental.total(), excess), "{w:?}");
        assert!(close(w.joules.total(), excess + 2.1 * 0.005));
        assert!(close(w.incremental.gpu_w, 11.9 * 0.02));
        // A second 15 ms job in the same sample: 1/4 and 3/4.
        let b = r.begin_job(0.2);
        r.end_job(b, 0.200, 0.215);
        let wa = r.integrate_job_with(a, 0.205, 0.210, base).expect("covered");
        let wb = r.integrate_job_with(b, 0.200, 0.215, base).expect("covered");
        assert!(close(wa.incremental.total(), excess * 0.25), "{wa:?}");
        assert!(close(wb.incremental.total(), excess * 0.75), "{wb:?}");
        // A long job over whole samples gets their whole excess: same as time integration.
        let c = r.begin_job(0.0);
        r.end_job(c, 0.0, 0.2);
        let wc = r.integrate_job_with(c, 0.0, 0.2, base).expect("covered");
        let wt = r.integrate_with(0.0, 0.2, base).expect("covered");
        assert!(close(wc.incremental.total(), wt.incremental.total()));
        assert!(close(wc.joules.total(), wt.joules.total()));
        assert_eq!(wc.confidence, Confidence::Measured);
    }

    #[test]
    fn baseline_is_low_percentile_and_refreshed() {
        let mut r = ring();
        let b = r.baseline(0.26);
        assert!(close(b.cpu_w, 2.0) && close(b.gpu_w, 0.1) && close(b.ane_w, 0.0), "{b:?}");
        // Cached until the refresh period passes.
        for i in 0..20 {
            let t = 0.26 + i as f64 * 0.02;
            r.push(sample(t, t + 0.02, 5.0, 0.2, 0.0));
        }
        assert!(close(r.baseline(1.0).cpu_w, 2.0));
        // Past the refresh period and the horizon: only the new idle level counts.
        let later = 1.0 + BASELINE_HORIZON_S;
        for i in 0..20 {
            let t = later - 0.4 + i as f64 * 0.02;
            r.push(sample(t, t + 0.02, 4.0, 0.3, 0.0));
        }
        assert!(close(r.baseline(later).cpu_w, 4.0));
        // Incremental never goes negative.
        let w = r.integrate_with(0.0, 0.02, Rails { cpu_w: 9.0, gpu_w: 9.0, ane_w: 9.0 }).expect("covered");
        assert_eq!(w.incremental.total(), 0.0);
    }

    #[test]
    fn uncovered_windows_are_none_and_gaps_are_filled() {
        let r = ring();
        assert!(r.integrate_with(5.0, 5.1, Rails::default()).is_none());
        // Half past the newest sample: scaled to the window, interpolated.
        let w = r.integrate_with(0.25, 0.27, Rails::default()).expect("partly covered");
        assert!(close(w.coverage, 0.5));
        assert_eq!(w.confidence, Confidence::Interpolated);
        assert!(close(w.joules.ane_w, 1.5 * 0.02));
    }

    #[test]
    fn ring_drops_oldest() {
        let mut r = PowerRing::new(3);
        for i in 0..5 {
            r.push(sample(i as f64, i as f64 + 1.0, 1.0, 0.0, 0.0));
        }
        assert_eq!(r.len(), 3);
        assert!(r.integrate_with(0.0, 1.0, Rails::default()).is_none());
        assert_eq!(r.latest_end(), Some(5.0));
    }

    const STREAM: &str = "\
*** Sampled system activity (Fri Oct  9 07:49:07 2026 -0400) (43.78ms elapsed) ***
CPU Power: 37235 mW
GPU Power: 46 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 37281 mW
**** GPU usage ****
GPU idle residency:  94.36%
GPU Power: 46 mW
*** Sampled system activity (Fri Oct  9 07:49:07 2026 -0400) (43.53ms elapsed) ***
CPU Power: 1.5 W
GPU Power: 900 mW
ANE Power: 250 mW
*** Sampled system activity (Fri Oct  9 07:49:07 2026 -0400) (20.00ms elapsed) ***
GPU Power: 7 mW
";

    #[test]
    fn parses_cpu_gpu_ane_rails() {
        let s = parse_rails(STREAM);
        assert_eq!(s.len(), 3);
        assert_eq!(s[0].elapsed_ms, 43.78);
        assert!(close(s[0].rails.cpu_w, 37.235));
        assert!(close(s[0].rails.gpu_w, 0.046));
        assert_eq!(s[0].combined_w, Some(37.281));
        // No combined line: completes at the next header; " W" units.
        assert!(close(s[1].rails.cpu_w, 1.5));
        assert!(close(s[1].rails.gpu_w, 0.9));
        assert!(close(s[1].rails.ane_w, 0.25));
        assert_eq!(s[1].combined_w, None);
        assert!(close(s[2].rails.gpu_w, 0.007));
    }

    #[test]
    #[cfg(not(target_os = "macos"))]
    fn no_global_sampler_off_macos() {
        assert!(PowerSampler::global().is_none());
    }
}
