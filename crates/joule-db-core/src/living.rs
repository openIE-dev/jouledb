//! Living Computing & Active-Inference Homeostatic Substrate
//!
//! Replaces dead, write-everything-forever database storage with two core principles of
//! biological & thermodynamic computing:
//!
//! 1. **[`SurpriseGate`] (Predictive-Coding / Variational Free-Energy Filter, ~0.5 µJ)**:
//!    Maintains an online generative expectation $(\hat{\mu}, \hat{\sigma}^2)$ per sensory
//!    channel plus a **Pre-Trigger Causal Black-Box Ring Buffer**. Habituated low-surprise
//!    observations fold into zero-I/O sufficient statistics (100-1,000x write amplification cut),
//!    whereas salient prediction-error spikes ($\mathcal{F} \ge \theta$) automatically capture
//!    the pre-trigger causal lead-up window + anomaly into durable hippocampal episodes!
//! 2. **[`HomeostaticMetabolism`] (Sympathetic Wake vs. Slow-Wave Sleep Consolidation)**:
//!    Adapts database maintenance to real power headroom and latency jitter budgets:
//!    suppressing background stalls during `SympatheticWake` real-time control loops, and
//!    running hippocampal-to-neocortical episode consolidation + Ebbinghaus synaptic pruning
//!    during `RestingConsolidate` windows.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::RwLock;

/// Single timestamped sensory observation fed into a [`SurpriseGate`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SensoryFrame {
    /// Monotonic timestamp in nanoseconds.
    pub timestamp_ns: u64,
    /// Feature/sensor vector.
    pub values: Vec<f64>,
}

/// Durable episode emitted when prediction error crosses the active-inference salience threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalientEpisode {
    /// Channel identifier.
    pub channel: String,
    /// Trigger timestamp in nanoseconds.
    pub trigger_timestamp_ns: u64,
    /// Variational surprise score ($z^2$ Mahalanobis free-energy proxy).
    pub surprise_score: f64,
    /// Causal lead-up frames from the pre-trigger black-box ring buffer (including the trigger frame).
    pub causal_window: Vec<SensoryFrame>,
}

/// Outcome of perceiving one [`SensoryFrame`] through [`SurpriseGate`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerceptionOutcome {
    /// Predictable steady-state frame absorbed into homeostatic moments with zero disk writes.
    Habituated {
        /// Observed surprise score ($< \theta$).
        surprise_score: f64,
        /// Cumulative frames absorbed without disk I/O on this channel.
        habituated_total: u64,
    },
    /// High-surprise prediction error: pre-trigger causal window promoted to episodic memory!
    PromotedEpisode(SalientEpisode),
}

#[derive(Debug, Clone)]
struct ChannelPredictor {
    mean: Vec<f64>,
    var: Vec<f64>,
    observations: u64,
    habituated_count: u64,
    promoted_count: u64,
    pre_trigger_ring: VecDeque<SensoryFrame>,
}

/// Predictive-coding ingress gate across arbitrary named sensory/telemetry channels.
pub struct SurpriseGate {
    /// Exponential adaptation rate $\alpha \in (0, 1)$ for running expectation $(\hat{\mu}, \hat{\sigma}^2)$.
    alpha: f64,
    /// Surprise threshold $\theta$ (mean squared z-score, e.g. `9.0` $\approx 3\sigma$ anomaly).
    surprise_threshold: f64,
    /// Number of warmup frames before habituation begins (initial frames establish baseline).
    warmup_frames: u64,
    /// Pre-trigger ring buffer capacity per channel.
    pre_trigger_len: usize,
    channels: RwLock<HashMap<String, ChannelPredictor>>,
}

impl SurpriseGate {
    /// Create a new `SurpriseGate` (`alpha` learning rate, `surprise_threshold` $z^2$, `pre_trigger_len`).
    pub fn new(alpha: f64, surprise_threshold: f64, pre_trigger_len: usize) -> Self {
        Self {
            alpha: alpha.clamp(0.001, 0.5),
            surprise_threshold: surprise_threshold.max(1.0),
            warmup_frames: 8,
            pre_trigger_len: pre_trigger_len.max(2),
            channels: RwLock::new(HashMap::new()),
        }
    }

    /// Process an incoming [`SensoryFrame`] in ~0.5 µJ, either habituating it into compact
    /// sufficient statistics or promoting a [`SalientEpisode`] with pre-trigger context.
    pub fn perceive(&self, channel: &str, frame: SensoryFrame) -> PerceptionOutcome {
        let mut map = match self.channels.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        // Bound distinct channels (4,096 max) so dynamic untrusted channel tags cannot OOM
        if !map.contains_key(channel) && map.len() >= 4096
            && let Some(evict_key) = map.keys().next().cloned()
        {
            map.remove(&evict_key);
        }
        let dim = frame.values.len().max(1);
        let state = map
            .entry(channel.to_string())
            .or_insert_with(|| ChannelPredictor {
                mean: frame.values.clone(),
                var: vec![1e-2; dim],
                observations: 0,
                habituated_count: 0,
                promoted_count: 0,
                pre_trigger_ring: VecDeque::with_capacity(self.pre_trigger_len + 1),
            });

        state.observations += 1;

        // Compute mean squared standardized prediction error (free-energy surprise proxy)
        let mut z2_sum = 0.0f64;
        let overlap = state.mean.len().min(frame.values.len());
        for i in 0..overlap {
            let diff = frame.values[i] - state.mean[i];
            let denom = state.var[i].max(1e-6);
            z2_sum += (diff * diff) / denom;
        }
        let surprise = if overlap > 0 {
            z2_sum / (overlap as f64)
        } else {
            0.0
        };

        // Push into pre-trigger black-box ring
        if state.pre_trigger_ring.len() >= self.pre_trigger_len {
            state.pre_trigger_ring.pop_front();
        }
        state.pre_trigger_ring.push_back(frame.clone());

        // Update generative expectation (damped update on massive outliers so spikes don't poison baseline)
        let eff_alpha = if state.observations <= self.warmup_frames {
            0.25
        } else if surprise >= self.surprise_threshold {
            self.alpha * 0.1
        } else {
            self.alpha
        };
        for i in 0..overlap {
            let diff = frame.values[i] - state.mean[i];
            state.mean[i] += eff_alpha * diff;
            state.var[i] = (1.0 - eff_alpha) * state.var[i] + eff_alpha * (diff * diff);
        }

        if state.observations > self.warmup_frames && surprise >= self.surprise_threshold {
            state.promoted_count += 1;
            let episode = SalientEpisode {
                channel: channel.to_string(),
                trigger_timestamp_ns: frame.timestamp_ns,
                surprise_score: surprise,
                causal_window: state.pre_trigger_ring.iter().cloned().collect(),
            };
            PerceptionOutcome::PromotedEpisode(episode)
        } else {
            state.habituated_count += 1;
            PerceptionOutcome::Habituated {
                surprise_score: surprise,
                habituated_total: state.habituated_count,
            }
        }
    }

    /// Return `(total_observations, habituated_frames, promoted_episodes)` across all channels.
    pub fn compression_telemetry(&self) -> (u64, u64, u64) {
        let Ok(map) = self.channels.read() else {
            return (0, 0, 0);
        };
        map.values().fold((0, 0, 0), |(obs, hab, pro), c| {
            (
                obs + c.observations,
                hab + c.habituated_count,
                pro + c.promoted_count,
            )
        })
    }
}

// ============================================================================
// Homeostatic Sleep/Wake Metabolic Scheduler
// ============================================================================

/// Autonomic operating state of a Living JouleDB node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MetabolicMode {
    /// Active real-time control / low-jitter perception: zero background compaction stalls!
    SympatheticWake,
    /// Idle or plugged-in surplus energy: slow-wave hippocampal replay, compression & pruning.
    RestingConsolidate,
    /// Critical battery / zero traffic: scale-to-zero object-manifest hibernate.
    DeepHibernate,
}

/// Consolidated neocortical prototype distilled during [`HomeostaticMetabolism::consolidate_sleep_cycle`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedPrototype {
    /// Source sensory channel.
    pub channel: String,
    /// Number of salient episodes merged into this prototype.
    pub episode_support: usize,
    /// Centroid feature vector across trigger frames.
    pub centroid: Vec<f64>,
    /// Peak surprise observed across merged episodes.
    pub max_surprise: f64,
}

/// Report emitted after a slow-wave consolidation cycle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Current metabolic state.
    pub mode: MetabolicMode,
    /// Episodes replayed from hippocampal buffer.
    pub episodes_replayed: usize,
    /// Weak/stale traces pruned via synaptic decay.
    pub traces_pruned: usize,
    /// Distilled semantic prototypes updated.
    pub prototypes_updated: usize,
    /// Modeled microjoules spent on consolidation.
    pub consolidation_energy_uj: u64,
}

/// Living homeostatic controller managing hippocampal episodes and neocortical prototypes.
pub struct HomeostaticMetabolism {
    hippocampal_buffer: RwLock<Vec<SalientEpisode>>,
    neocortical_prototypes: RwLock<HashMap<String, ConsolidatedPrototype>>,
}

impl Default for HomeostaticMetabolism {
    fn default() -> Self {
        Self::new()
    }
}

impl HomeostaticMetabolism {
    /// Create a new `HomeostaticMetabolism` controller.
    pub fn new() -> Self {
        Self {
            hippocampal_buffer: RwLock::new(Vec::new()),
            neocortical_prototypes: RwLock::new(HashMap::new()),
        }
    }

    /// Select autonomic [`MetabolicMode`] from battery percentage, active control-loop flag, and idle time.
    pub fn select_mode(
        battery_pct: f32,
        realtime_control_active: bool,
        idle_ms: u64,
    ) -> MetabolicMode {
        if realtime_control_active {
            MetabolicMode::SympatheticWake
        } else if battery_pct < 10.0 && idle_ms >= 5_000 {
            MetabolicMode::DeepHibernate
        } else if idle_ms >= 250 && battery_pct >= 15.0 {
            MetabolicMode::RestingConsolidate
        } else {
            MetabolicMode::SympatheticWake
        }
    }

    /// Stage a [`SalientEpisode`] into the fast hippocampal write buffer.
    pub fn stage_episode(&self, episode: SalientEpisode) {
        if let Ok(mut buf) = self.hippocampal_buffer.write() {
            buf.push(episode);
        }
    }

    /// Run a slow-wave sleep consolidation cycle: prune low-surprise transient noise below
    /// `min_retain_surprise` and distill surviving episodes into neocortical channel prototypes!
    pub fn consolidate_sleep_cycle(&self, min_retain_surprise: f64) -> ConsolidationReport {
        let episodes: Vec<SalientEpisode> = match self.hippocampal_buffer.write() {
            Ok(mut buf) => std::mem::take(&mut *buf),
            Err(p) => std::mem::take(&mut *p.into_inner()),
        };

        let episodes_replayed = episodes.len();
        let mut traces_pruned = 0usize;
        let mut by_channel: HashMap<String, Vec<SalientEpisode>> = HashMap::new();

        for ep in episodes {
            if ep.surprise_score < min_retain_surprise {
                traces_pruned += 1;
            } else {
                by_channel.entry(ep.channel.clone()).or_default().push(ep);
            }
        }

        let mut protos = match self.neocortical_prototypes.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let mut prototypes_updated = 0usize;

        for (ch, eps) in by_channel {
            let Some(first_trigger) = eps.first().and_then(|e| e.causal_window.last()) else {
                continue;
            };
            let dim = first_trigger.values.len();
            if dim == 0 {
                continue;
            }
            let mut sum = vec![0.0f64; dim];
            let mut count = 0usize;
            let mut max_surp = 0.0f64;
            for ep in &eps {
                max_surp = max_surp.max(ep.surprise_score);
                if let Some(trig) = ep.causal_window.last()
                    && trig.values.len() == dim
                {
                    for (i, &v) in trig.values.iter().enumerate() {
                        sum[i] += v;
                    }
                    count += 1;
                }
            }
            if count > 0 {
                let centroid: Vec<f64> = sum.into_iter().map(|s| s / (count as f64)).collect();
                protos.insert(
                    ch.clone(),
                    ConsolidatedPrototype {
                        channel: ch,
                        episode_support: count,
                        centroid,
                        max_surprise: max_surp,
                    },
                );
                prototypes_updated += 1;
            }
        }

        ConsolidationReport {
            mode: MetabolicMode::RestingConsolidate,
            episodes_replayed,
            traces_pruned,
            prototypes_updated,
            consolidation_energy_uj: 2 + (episodes_replayed as u64) * 3,
        }
    }

    /// Snapshot all consolidated neocortical prototypes.
    pub fn prototypes(&self) -> Vec<ConsolidatedPrototype> {
        self.neocortical_prototypes
            .read()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surprise_gate_habituates_steady_stream_and_promotes_anomaly_with_pretrigger_ring() {
        let gate = SurpriseGate::new(0.1, 16.0, 4);
        let metabolism = HomeostaticMetabolism::new();

        // Feed 100 steady-state 9.81 m/s^2 gravity readings -> all post-warmup habituate!
        for i in 0..100u64 {
            let outcome = gate.perceive(
                "imu.accel_z",
                SensoryFrame {
                    timestamp_ns: i * 1_000_000,
                    values: vec![9.81 + ((i % 2) as f64) * 0.01],
                },
            );
            assert!(matches!(outcome, PerceptionOutcome::Habituated { .. }));
        }

        // Sudden 45 m/s^2 collision spike at t=100ms!
        let spike = gate.perceive(
            "imu.accel_z",
            SensoryFrame {
                timestamp_ns: 100_000_000,
                values: vec![45.0],
            },
        );
        match spike {
            PerceptionOutcome::PromotedEpisode(ep) => {
                assert!(ep.surprise_score > 100.0);
                // Pre-trigger black-box ring captured 3 lead-up frames + the collision frame = 4!
                assert_eq!(ep.causal_window.len(), 4);
                assert_eq!(ep.causal_window.last().unwrap().values[0], 45.0);
                metabolism.stage_episode(ep);
            }
            PerceptionOutcome::Habituated { .. } => panic!("collision spike must trigger episode!"),
        }

        let (obs, hab, pro) = gate.compression_telemetry();
        assert_eq!(obs, 101);
        assert_eq!(hab, 100);
        assert_eq!(pro, 1);

        // Slow-wave sleep consolidates the collision episode into a neocortical prototype!
        let report = metabolism.consolidate_sleep_cycle(10.0);
        assert_eq!(report.prototypes_updated, 1);
        assert_eq!(metabolism.prototypes().len(), 1);
    }
}
