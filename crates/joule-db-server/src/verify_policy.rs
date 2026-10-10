//! Sampled verification of accelerator results.
//!
//! Per `(op, shape bucket, machine, device/backend)`: verify every result
//! for the first `first` runs, then a deterministic jittered sample of
//! about one in `every`. A mismatch resets that key to verify-every-run
//! (the next `first` runs) and is recorded; the caller still returns the
//! CPU result. `JOULE_VERIFY=always` verifies everything;
//! `JOULE_VERIFY_FIRST` / `JOULE_VERIFY_EVERY` set the schedule
//! (defaults 8 and 16).

use std::collections::HashMap;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Always,
    Sampled,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::Always => "always",
            Mode::Sampled => "sampled",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Policy {
    pub mode: Mode,
    pub first: u64,
    pub every: u64,
}

impl Default for Policy {
    fn default() -> Self {
        Policy { mode: Mode::Sampled, first: 8, every: 16 }
    }
}

impl Policy {
    pub fn from_env() -> Self {
        let d = Policy::default();
        let num = |k: &str, v: u64| std::env::var(k).ok().and_then(|s| s.parse::<u64>().ok()).unwrap_or(v);
        Policy {
            mode: match std::env::var("JOULE_VERIFY").as_deref() {
                Ok("always") | Ok("1") | Ok("all") => Mode::Always,
                _ => d.mode,
            },
            first: num("JOULE_VERIFY_FIRST", d.first),
            every: num("JOULE_VERIFY_EVERY", d.every).max(1),
        }
    }
}

#[derive(Debug, Default, Clone)]
struct Bucket {
    /// Runs seen since the last reset.
    runs: u64,
    /// Next sampled run (after the first `first`).
    next: u64,
}

/// One recorded mismatch.
#[derive(Debug, Clone, PartialEq)]
pub struct Mismatch {
    pub key: String,
    pub run: u64,
}

pub struct Verifier {
    pub policy: Policy,
    buckets: Mutex<HashMap<String, Bucket>>,
    mismatches: Mutex<Vec<Mismatch>>,
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

fn hash(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x100_0000_01b3))
}

/// This machine's name, part of every key.
pub fn machine() -> &'static str {
    static M: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    M.get_or_init(|| {
        std::env::var("HOSTNAME")
            .ok()
            .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
            .or_else(|| {
                std::process::Command::new("hostname")
                    .output()
                    .ok()
                    .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
            })
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "local".into())
    })
}

/// `op|bucket|machine|device/backend`.
pub fn key(op: &str, bucket: &str, device_backend: &str) -> String {
    format!("{op}|{bucket}|{}|{device_backend}", machine())
}

impl Verifier {
    pub fn new(policy: Policy) -> Self {
        Verifier { policy, buckets: Mutex::new(HashMap::new()), mismatches: Mutex::new(Vec::new()) }
    }

    pub fn global() -> &'static Verifier {
        static G: std::sync::OnceLock<Verifier> = std::sync::OnceLock::new();
        G.get_or_init(|| Verifier::new(Policy::from_env()))
    }

    /// Gap to the next sample: `every` with deterministic jitter in
    /// `[every/2, every*3/2]`.
    fn gap(&self, key: &str, run: u64) -> u64 {
        let e = self.policy.every;
        if e <= 1 {
            return 1;
        }
        let half = e / 2;
        (e - half + mix(hash(key) ^ run) % (2 * half + 1)).max(1)
    }

    /// Count a run for `key` and say whether to verify it.
    pub fn should_verify(&self, key: &str) -> bool {
        if self.policy.mode == Mode::Always {
            return true;
        }
        let mut map = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        let b = map.entry(key.to_string()).or_default();
        let run = b.runs;
        b.runs += 1;
        if run < self.policy.first {
            if run + 1 == self.policy.first {
                b.next = run + self.gap(key, run);
            }
            return true;
        }
        if run >= b.next {
            b.next = run + self.gap(key, run);
            return true;
        }
        false
    }

    /// A verified result disagreed: back to verify-every-run for `key`.
    pub fn record_mismatch(&self, key: &str) {
        let mut map = self.buckets.lock().unwrap_or_else(|p| p.into_inner());
        let b = map.entry(key.to_string()).or_default();
        let run = b.runs.saturating_sub(1);
        *b = Bucket::default();
        self.mismatches
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(Mismatch { key: key.to_string(), run });
    }

    pub fn mismatches(&self) -> Vec<Mismatch> {
        self.mismatches.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    pub fn label(&self) -> &'static str {
        self.policy.mode.label()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schedule(v: &Verifier, key: &str, runs: usize) -> Vec<bool> {
        (0..runs).map(|_| v.should_verify(key)).collect()
    }

    #[test]
    fn first_runs_then_deterministic_jittered_sample() {
        let p = Policy { mode: Mode::Sampled, first: 8, every: 16 };
        let a = schedule(&Verifier::new(p), "sum|b20|m|gpu/metal", 400);
        let b = schedule(&Verifier::new(p), "sum|b20|m|gpu/metal", 400);
        assert_eq!(a, b, "deterministic");
        assert!(a[..8].iter().all(|v| *v));
        let later: Vec<usize> = a.iter().enumerate().skip(8).filter(|(_, v)| **v).map(|(i, _)| i).collect();
        // About 1 in 16 with gaps in [8, 24].
        assert!((18..=32).contains(&later.len()), "{}", later.len());
        let mut prev = 7;
        for i in &later {
            assert!((8..=24).contains(&(i - prev)), "gap {} at {i}", i - prev);
            prev = *i;
        }
        // Gaps are jittered, not a fixed stride; other keys get another schedule.
        let gaps: std::collections::HashSet<usize> = later.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.len() > 2);
        assert_ne!(a, schedule(&Verifier::new(p), "hdc|b|m|npu/coreml-ane", 400));
    }

    #[test]
    fn mismatch_resets_to_verify_every_run_and_is_recorded() {
        let v = Verifier::new(Policy { mode: Mode::Sampled, first: 4, every: 16 });
        let k = "k";
        let _ = schedule(&v, k, 40);
        v.record_mismatch(k);
        assert!(schedule(&v, k, 4).iter().all(|x| *x));
        assert_eq!(v.mismatches(), vec![Mismatch { key: "k".into(), run: 39 }]);
        // Other keys are untouched.
        assert!(v.should_verify("other"));
    }

    #[test]
    fn always_mode_verifies_everything() {
        let v = Verifier::new(Policy { mode: Mode::Always, first: 0, every: 16 });
        assert!(schedule(&v, "k", 50).iter().all(|x| *x));
        assert_eq!(v.label(), "always");
    }
}
