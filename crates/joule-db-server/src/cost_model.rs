//! Measured-cost routing.
//!
//! The fabric picks a processor for a job by what the same kind of job
//! actually cost on this machine. A [`CostModel`] keeps, per
//! `(op kind, size bucket, device/backend)`, the observed joules and seconds
//! (exponentially weighted, so a changed machine is relearned), and
//! [`CostModel::choose`] routes to the cheapest known candidate by energy,
//! with time as the tie-breaker.
//!
//! Energy means measured incremental joules (above the idle baseline, see
//! [`crate::power_meter`]) when every compared candidate has a measured
//! reading; otherwise the flat-watts estimate, which ranks exactly like time.
//! Each entry records its `energy_source`.
//!
//! Exploration is bounded: with no history the static preference runs
//! (`no_history`); once one candidate in a bucket is measured, each
//! unmeasured candidate is tried exactly once (`explore`); after that one
//! call in every [`CostModel::explore_every`] per bucket re-measures the
//! runner-up (`explore`). Everything else is `cost_model_<device>_cheaper`.
//!
//! Routing never refuses: there is always a chosen candidate and a fallback
//! that is a real processor (the other candidate, or the CPU itself on a
//! CPU-only host). Energy is reported, never used to deny work.
//!
//! The model is per process and can persist to a small JSON file (e.g.
//! under the database path) so a restarted server keeps what it learned.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Mutex;

/// One processor a job can run on.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Candidate {
    /// `cpu`, `gpu`, `npu`, ... (a real processor).
    pub device: String,
    /// Concrete path, e.g. `cpu`, `wgpu-metal`, `wgpu-vulkan`.
    pub backend: String,
}

impl Candidate {
    pub fn new(device: impl Into<String>, backend: impl Into<String>) -> Self {
        Self { device: device.into(), backend: backend.into() }
    }

    pub fn cpu() -> Self {
        Self::new("cpu", "cpu")
    }
}

/// What a routing decision rests on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteReason {
    /// Nothing measured in this bucket yet: the static preference runs.
    NoHistory,
    /// Measuring a candidate (first time in this bucket, or periodic).
    Explore,
    /// Measured cheapest (by joules, then time).
    CostModelCheaper,
    /// Only one candidate exists.
    OnlyCandidate,
}

/// A routing decision.
#[derive(Debug, Clone)]
pub struct Route {
    pub chosen: Candidate,
    /// Next real processor if `chosen` cannot run the job.
    pub fallback: Candidate,
    pub reason: RouteReason,
    /// The first candidate passed in (the static preference).
    pub requested: Candidate,
    /// Size bucket used for the lookup.
    pub bucket: u32,
    /// How measured candidates were compared (`None` for `no_history`,
    /// first-time `explore`, `only_candidate`).
    pub ranked_by: Option<RankedBy>,
}

impl Route {
    /// `route_reason` string for receipts, e.g. `cost_model_gpu_cheaper`.
    pub fn reason_label(&self) -> String {
        match self.reason {
            RouteReason::NoHistory => "no_history".into(),
            RouteReason::Explore => "explore".into(),
            RouteReason::OnlyCandidate => "only_candidate".into(),
            RouteReason::CostModelCheaper => format!("cost_model_{}_cheaper", self.chosen.device),
        }
    }
}

/// The receipt of a cost-routed job.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RouteReceipt {
    /// Op kind the cost model keys on, e.g. `sigql:iir`.
    pub op: String,
    /// `ceil(log2(size))`.
    pub size_bucket: u32,
    /// Static preference (the first candidate), e.g. `gpu`.
    pub requested_device: String,
    /// Processor that produced the result.
    pub device: String,
    /// Path on that processor, e.g. `wgpu-metal` or `cpu`.
    pub backend: String,
    /// Next real processor (never empty).
    pub fallback: String,
    /// `cost_model_cpu_cheaper`, `cost_model_gpu_cheaper`, `explore`,
    /// `no_history`, `only_candidate`; `+<device>_error` when the chosen
    /// device failed and the fallback ran.
    pub route_reason: String,
    /// Energy of this run: total measured joules over the job window (all
    /// rails / package), or `watts x seconds` for an estimate.
    pub joules: f64,
    /// Joules above the idle baseline (equals `joules` for an estimate).
    #[serde(default)]
    pub incremental_joules: f64,
    pub seconds: f64,
    /// `joules / seconds`.
    pub watts: f64,
    /// `powermetrics`, `rapl`, `emi`, or `estimate`.
    pub energy_source: String,
    /// `measured`, `interpolated`, or `estimate`.
    #[serde(default)]
    pub energy_confidence: String,
    /// Mean per-device watts over the job window (measured), or the estimate
    /// watts on the device that ran.
    #[serde(default)]
    pub cpu_watts: Option<f64>,
    #[serde(default)]
    pub gpu_watts: Option<f64>,
    #[serde(default)]
    pub ane_watts: Option<f64>,
    /// Idle baseline subtracted for `incremental_joules`, watts.
    #[serde(default)]
    pub baseline_watts: Option<f64>,
    /// What the cost model ranked by for this decision: `incremental_joules`
    /// (all candidates measured) or `time` (some estimate).
    #[serde(default)]
    pub ranked_by: String,
    /// One-time setup this call paid (GPU pipeline compile), not included
    /// in `joules` / `seconds`.
    pub setup_seconds: Option<f64>,
}

impl RouteReceipt {
    /// One-line, key=value rendering for logs and test output.
    pub fn line(&self) -> String {
        let w = |v: Option<f64>| v.map(|x| format!("{x:.3}")).unwrap_or_else(|| "none".into());
        format!(
            "op={} bucket={} requested_device={} device={} backend={} fallback={} route_reason={} ranked_by={} joules={:.6e} incremental_joules={:.6e} seconds={:.6} watts={:.2} energy_source={} energy_confidence={} cpu_w={} gpu_w={} ane_w={} baseline_w={} setup_s={}",
            self.op,
            self.size_bucket,
            self.requested_device,
            self.device,
            self.backend,
            self.fallback,
            self.route_reason,
            self.ranked_by,
            self.joules,
            self.incremental_joules,
            self.seconds,
            self.watts,
            self.energy_source,
            self.energy_confidence,
            w(self.cpu_watts),
            w(self.gpu_watts),
            w(self.ane_watts),
            w(self.baseline_watts),
            self.setup_seconds.map(|s| format!("{s:.4}")).unwrap_or_else(|| "none".into())
        )
    }
}

/// Observed cost of one `(op, bucket, candidate)`.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Observed {
    pub runs: u64,
    /// Running joules: measured totals, or `watts x seconds` estimates.
    pub joules: f64,
    pub seconds: f64,
    /// Running measured incremental joules; `None` until a measured run.
    #[serde(default)]
    pub incremental_joules: Option<f64>,
    /// Source of the latest observation.
    #[serde(default = "estimate_label")]
    pub energy_source: String,
}

fn estimate_label() -> String {
    "estimate".into()
}

/// How [`CostModel::choose`] compared candidates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankedBy {
    /// Every candidate had measured incremental joules.
    IncrementalJoules,
    /// Some candidate only had an estimate: flat watts, i.e. time.
    Time,
}

impl RankedBy {
    pub fn label(&self) -> &'static str {
        match self {
            RankedBy::IncrementalJoules => "incremental_joules",
            RankedBy::Time => "time",
        }
    }
}

/// Order two observations: by measured incremental joules when both have
/// them (2% tie band, then time), else by time.
fn compare(a: &Observed, b: &Observed, by: RankedBy) -> std::cmp::Ordering {
    match (by, a.incremental_joules, b.incremental_joules) {
        (RankedBy::IncrementalJoules, Some(x), Some(y)) => {
            let scale = x.abs().max(y.abs()).max(f64::MIN_POSITIVE);
            if (x - y).abs() <= ENERGY_TIE * scale {
                a.seconds.total_cmp(&b.seconds)
            } else {
                x.total_cmp(&y)
            }
        }
        _ => a.seconds.total_cmp(&b.seconds),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Key {
    op: String,
    bucket: u32,
    candidate: Candidate,
}

#[derive(Default)]
struct State {
    costs: HashMap<Key, Observed>,
    /// Decisions per `(op, bucket)`, for the periodic explore schedule.
    calls: HashMap<(String, u32), u64>,
}

fn ranked_by(measured: &[(Candidate, Observed)]) -> RankedBy {
    if measured.iter().all(|(_, o)| o.incremental_joules.is_some()) {
        RankedBy::IncrementalJoules
    } else {
        RankedBy::Time
    }
}

/// Size bucket: `ceil(log2(size))`, so 64K and 1M land in buckets 16 and 20.
pub fn size_bucket(size: usize) -> u32 {
    usize::BITS - size.max(1).saturating_sub(1).leading_zeros()
}

/// Relative difference within which two energies count as a tie.
const ENERGY_TIE: f64 = 0.02;
/// Weight of a new observation in the running cost.
const EWMA: f64 = 0.3;

/// Per-process measured-cost model.
pub struct CostModel {
    state: Mutex<State>,
    /// Re-measure the runner-up once every this many decisions per bucket
    /// (0 disables periodic exploration). Default 32.
    pub explore_every: u64,
    path: Option<PathBuf>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct Persisted {
    op: String,
    bucket: u32,
    device: String,
    backend: String,
    cost: Observed,
}

impl CostModel {
    pub fn new() -> Self {
        Self { state: Mutex::new(State::default()), explore_every: 32, path: None }
    }

    /// A model that loads from and saves to `path` (best effort: an
    /// unreadable file starts empty, a failed save is ignored).
    pub fn with_file(path: impl Into<PathBuf>) -> Self {
        let mut model = Self::new();
        let path = path.into();
        if let Ok(text) = std::fs::read_to_string(&path) {
            if let Ok(rows) = serde_json::from_str::<Vec<Persisted>>(&text) {
                let state = model.state.get_mut().unwrap_or_else(|p| p.into_inner());
                for row in rows {
                    state.costs.insert(
                        Key { op: row.op, bucket: row.bucket, candidate: Candidate::new(row.device, row.backend) },
                        row.cost,
                    );
                }
            }
        }
        model.path = Some(path);
        model
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Record (or inject) an estimated cost (`joules = watts x seconds`).
    pub fn record(&self, op: &str, size: usize, candidate: &Candidate, seconds: f64, joules: f64) {
        self.record_cost(op, size, candidate, seconds, joules, None, "estimate");
    }

    /// Record a job's energy as returned by the power meter.
    pub fn record_energy(&self, op: &str, size: usize, candidate: &Candidate, e: &crate::power_meter::JobEnergy) {
        let incremental = e.is_measured().then_some(e.incremental_joules);
        self.record_cost(op, size, candidate, e.seconds, e.joules, incremental, &e.source);
    }

    /// Record (or inject) a cost; `incremental` is `Some` only for a
    /// measured reading.
    #[allow(clippy::too_many_arguments)]
    pub fn record_cost(
        &self,
        op: &str,
        size: usize,
        candidate: &Candidate,
        seconds: f64,
        joules: f64,
        incremental: Option<f64>,
        source: &str,
    ) {
        let key = Key { op: op.to_string(), bucket: size_bucket(size), candidate: candidate.clone() };
        {
            let mut state = self.lock();
            let entry = state.costs.entry(key).or_insert(Observed {
                runs: 0,
                joules,
                seconds,
                incremental_joules: None,
                energy_source: source.to_string(),
            });
            if entry.runs > 0 {
                entry.joules += EWMA * (joules - entry.joules);
                entry.seconds += EWMA * (seconds - entry.seconds);
            }
            if let Some(inc) = incremental {
                entry.incremental_joules = Some(match entry.incremental_joules {
                    Some(prev) => prev + EWMA * (inc - prev),
                    None => inc,
                });
            }
            entry.energy_source = source.to_string();
            entry.runs += 1;
        }
        self.save();
    }

    /// Cheapest measured candidate by the energy rule and by time alone,
    /// without counting a decision. `None` until every candidate is measured.
    pub fn rankings(&self, op: &str, size: usize, candidates: &[Candidate]) -> Option<(Candidate, Candidate, RankedBy)> {
        let bucket = size_bucket(size);
        let state = self.lock();
        let measured: Option<Vec<(Candidate, Observed)>> = candidates
            .iter()
            .map(|c| {
                state
                    .costs
                    .get(&Key { op: op.to_string(), bucket, candidate: c.clone() })
                    .map(|o| (c.clone(), o.clone()))
            })
            .collect();
        let measured = measured.filter(|m| !m.is_empty())?;
        let by = ranked_by(&measured);
        let energy = measured.iter().min_by(|(_, a), (_, b)| compare(a, b, by))?.0.clone();
        let time = measured.iter().min_by(|(_, a), (_, b)| compare(a, b, RankedBy::Time))?.0.clone();
        Some((energy, time, by))
    }

    /// The running cost of `candidate` for `op` at `size`, if measured.
    pub fn observed(&self, op: &str, size: usize, candidate: &Candidate) -> Option<Observed> {
        let key = Key { op: op.to_string(), bucket: size_bucket(size), candidate: candidate.clone() };
        self.lock().costs.get(&key).cloned()
    }

    /// Pick a candidate for `op` at `size`. `candidates[0]` is the static
    /// preference. An empty list routes to the CPU: there is always a
    /// processor.
    pub fn choose(&self, op: &str, size: usize, candidates: &[Candidate]) -> Route {
        let bucket = size_bucket(size);
        let candidates: Vec<Candidate> =
            if candidates.is_empty() { vec![Candidate::cpu()] } else { candidates.to_vec() };
        let requested = candidates[0].clone();
        let fallback_for = |chosen: &Candidate| {
            candidates
                .iter()
                .find(|c| c.device != chosen.device)
                .cloned()
                .unwrap_or_else(|| {
                    // Same device only (or one candidate): the CPU is the
                    // next real processor, or the CPU itself on a CPU-only host.
                    Candidate::cpu()
                })
        };
        let route = |chosen: Candidate, reason| Route {
            fallback: fallback_for(&chosen),
            chosen,
            reason,
            requested: requested.clone(),
            bucket,
            ranked_by: None,
        };
        if candidates.len() == 1 {
            return route(requested.clone(), RouteReason::OnlyCandidate);
        }
        let mut state = self.lock();
        let call = {
            let n = state.calls.entry((op.to_string(), bucket)).or_insert(0);
            *n += 1;
            *n
        };
        let measured: Vec<(Candidate, Observed)> = candidates
            .iter()
            .filter_map(|c| {
                state
                    .costs
                    .get(&Key { op: op.to_string(), bucket, candidate: c.clone() })
                    .map(|o| (c.clone(), o.clone()))
            })
            .collect();
        if measured.is_empty() {
            return route(requested.clone(), RouteReason::NoHistory);
        }
        if let Some(unmeasured) = candidates.iter().find(|c| !measured.iter().any(|(m, _)| m == *c)) {
            return route(unmeasured.clone(), RouteReason::Explore);
        }
        let mut ranked = measured;
        let by = ranked_by(&ranked);
        ranked.sort_by(|(_, a), (_, b)| compare(a, b, by));
        drop(state);
        if self.explore_every > 0 && call % self.explore_every == 0 {
            return Route { ranked_by: Some(by), ..route(ranked[1].0.clone(), RouteReason::Explore) };
        }
        Route { ranked_by: Some(by), ..route(ranked[0].0.clone(), RouteReason::CostModelCheaper) }
    }

    fn save(&self) {
        let Some(path) = &self.path else { return };
        let rows: Vec<Persisted> = self
            .lock()
            .costs
            .iter()
            .map(|(k, v)| Persisted {
                op: k.op.clone(),
                bucket: k.bucket,
                device: k.candidate.device.clone(),
                backend: k.candidate.backend.clone(),
                cost: v.clone(),
            })
            .collect();
        if let Ok(text) = serde_json::to_string(&rows) {
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, text).is_ok() {
                let _ = std::fs::rename(&tmp, path);
            }
        }
    }
}

impl Default for CostModel {
    fn default() -> Self {
        Self::new()
    }
}

/// The process-wide model. Persists to `JOULE_COST_MODEL_PATH` when set,
/// or to the file passed to [`init_persistent`] before first use (the
/// server passes `<db path>/cost_model.json`); in memory otherwise.
pub fn global() -> &'static CostModel {
    GLOBAL.get_or_init(|| match std::env::var_os("JOULE_COST_MODEL_PATH") {
        Some(path) => CostModel::with_file(path),
        None => CostModel::new(),
    })
}

/// File name of the persisted model under a database path.
pub const FILE: &str = "cost_model.json";

static GLOBAL: std::sync::OnceLock<CostModel> = std::sync::OnceLock::new();

/// Make the process-wide model load from and save to `path`. Returns
/// `false` (and changes nothing) when the model is already in use.
pub fn init_persistent(path: impl Into<PathBuf>) -> bool {
    let path = path.into();
    GLOBAL.set(CostModel::with_file(path)).is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu() -> Candidate {
        Candidate::new("gpu", "wgpu-metal")
    }

    fn real(c: &Candidate) -> bool {
        matches!(c.device.as_str(), "cpu" | "gpu" | "npu" | "tpu" | "lpu")
    }

    #[test]
    fn buckets_are_log2() {
        assert_eq!(size_bucket(0), 0);
        assert_eq!(size_bucket(1), 0);
        assert_eq!(size_bucket(65536), 16);
        assert_eq!(size_bucket(65537), 17);
        assert_eq!(size_bucket(1 << 20), 20);
    }

    #[test]
    fn injected_costs_route_to_the_cheaper_device_by_energy() {
        let mut model = CostModel::new();
        model.explore_every = 0;
        let both = [gpu(), Candidate::cpu()];
        // Small jobs: CPU cheaper. Large jobs: GPU cheaper.
        model.record("sigql:iir", 65536, &Candidate::cpu(), 0.0004, 0.002);
        model.record("sigql:iir", 65536, &gpu(), 0.003, 0.015);
        model.record("sigql:iir", 1 << 20, &Candidate::cpu(), 0.010, 0.050);
        model.record("sigql:iir", 1 << 20, &gpu(), 0.004, 0.020);
        let small = model.choose("sigql:iir", 65536, &both);
        assert_eq!(small.chosen, Candidate::cpu());
        assert_eq!(small.reason_label(), "cost_model_cpu_cheaper");
        assert_eq!(small.fallback, gpu());
        assert_eq!(small.requested, gpu());
        let large = model.choose("sigql:iir", 1 << 20, &both);
        assert_eq!(large.chosen, gpu());
        assert_eq!(large.reason_label(), "cost_model_gpu_cheaper");
        assert_eq!(large.fallback, Candidate::cpu());
        // Measured energy is primary: faster but hungrier loses.
        model.record_cost("op", 100, &Candidate::cpu(), 0.001, 1.5, Some(1.0), "powermetrics");
        model.record_cost("op", 100, &gpu(), 0.0005, 2.5, Some(2.0), "powermetrics");
        let r = model.choose("op", 100, &both);
        assert_eq!(r.chosen, Candidate::cpu());
        assert_eq!(r.ranked_by, Some(RankedBy::IncrementalJoules));
        // Within the energy tie band, time decides.
        model.record_cost("tie", 100, &Candidate::cpu(), 0.002, 1.0, Some(1.0), "powermetrics");
        model.record_cost("tie", 100, &gpu(), 0.001, 1.0, Some(1.01), "powermetrics");
        assert_eq!(model.choose("tie", 100, &both).chosen, gpu());
        // Estimates are flat watts x time: they rank like time.
        model.record("est", 100, &Candidate::cpu(), 0.001, 1.0);
        model.record("est", 100, &gpu(), 0.0005, 2.0);
        let r = model.choose("est", 100, &both);
        assert_eq!(r.chosen, gpu());
        assert_eq!(r.ranked_by, Some(RankedBy::Time));
    }

    #[test]
    fn exploration_is_bounded() {
        let mut model = CostModel::new();
        model.explore_every = 4;
        let both = [gpu(), Candidate::cpu()];
        let r = model.choose("x", 1000, &both);
        assert_eq!((r.chosen.clone(), r.reason_label().as_str()), (gpu(), "no_history"));
        model.record("x", 1000, &gpu(), 0.01, 0.05);
        // The unmeasured CPU is explored once...
        let r = model.choose("x", 1000, &both);
        assert_eq!((r.chosen.clone(), r.reason_label().as_str()), (Candidate::cpu(), "explore"));
        model.record("x", 1000, &Candidate::cpu(), 0.001, 0.005);
        // ...then the cheaper one wins, except one call in four re-measures
        // the runner-up.
        let reasons: Vec<String> = (0..8).map(|_| model.choose("x", 1000, &both).reason_label()).collect();
        let explores = reasons.iter().filter(|r| *r == "explore").count();
        assert_eq!(explores, 2, "{reasons:?}");
        assert_eq!(reasons.iter().filter(|r| *r == "cost_model_cpu_cheaper").count(), 6);
        // A different bucket has its own history.
        assert_eq!(model.choose("x", 1 << 20, &both).reason_label(), "no_history");
    }

    #[test]
    fn fallback_is_always_a_real_processor() {
        let model = CostModel::new();
        for candidates in [vec![], vec![Candidate::cpu()], vec![gpu()], vec![gpu(), Candidate::cpu()], vec![Candidate::cpu(), gpu()]] {
            let r = model.choose("y", 10, &candidates);
            assert!(real(&r.chosen), "{r:?}");
            assert!(real(&r.fallback), "{r:?}");
            assert!(real(&r.requested), "{r:?}");
            assert!(!r.reason_label().is_empty());
        }
        assert_eq!(model.choose("y", 10, &[]).reason_label(), "only_candidate");
    }

    #[test]
    fn costs_persist_across_processes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cost_model.json");
        {
            let model = CostModel::with_file(&path);
            model.record("z", 4096, &gpu(), 0.002, 0.01);
            model.record("z", 4096, &Candidate::cpu(), 0.001, 0.004);
        }
        let mut model = CostModel::with_file(&path);
        model.explore_every = 0;
        let r = model.choose("z", 4096, &[gpu(), Candidate::cpu()]);
        assert_eq!(r.reason_label(), "cost_model_cpu_cheaper");
        assert_eq!(model.observed("z", 4096, &gpu()).unwrap().runs, 1);
    }

    #[test]
    fn measured_incremental_joules_can_overrule_time() {
        let mut model = CostModel::new();
        model.explore_every = 0;
        let cpu = Candidate::cpu();
        let both = [gpu(), cpu.clone()];
        // GPU slower (1.5 ms vs 1.2 ms) but draws far less above idle.
        model.record_cost("sigql:iir", 65536, &cpu, 0.0012, 0.060, Some(0.030), "powermetrics");
        model.record_cost("sigql:iir", 65536, &gpu(), 0.0015, 0.050, Some(0.006), "powermetrics");
        let (by_energy, by_time, by) = model.rankings("sigql:iir", 65536, &both).expect("measured");
        assert_eq!((by_energy, by_time, by), (gpu(), cpu.clone(), RankedBy::IncrementalJoules));
        let r = model.choose("sigql:iir", 65536, &both);
        assert_eq!(r.chosen, gpu());
        assert_eq!(r.reason_label(), "cost_model_gpu_cheaper");
        // One candidate only estimated: no mixing of measured and estimated
        // joules; time decides.
        model.record("sigql:iir", 1 << 20, &cpu, 0.010, 0.05);
        model.record_cost("sigql:iir", 1 << 20, &gpu(), 0.012, 0.03, Some(0.001), "powermetrics");
        let r = model.choose("sigql:iir", 1 << 20, &both);
        assert_eq!((r.chosen, r.ranked_by), (cpu.clone(), Some(RankedBy::Time)));
        // A later measured run upgrades the entry.
        model.record_cost("sigql:iir", 1 << 20, &cpu, 0.010, 0.2, Some(0.1), "powermetrics");
        let r = model.choose("sigql:iir", 1 << 20, &both);
        assert_eq!((r.chosen, r.ranked_by), (gpu(), Some(RankedBy::IncrementalJoules)));
        let o = model.observed("sigql:iir", 1 << 20, &cpu).expect("observed");
        assert_eq!(o.energy_source, "powermetrics");
    }
}
