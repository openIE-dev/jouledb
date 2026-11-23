//! Embodied & Physical-AI Substrate (`HLC` + `Entity-Component-Time` + `SE(3)` Transform Tree)
//!
//! Moves JouleDB beyond 1970s discrete CRUD rows into the continuous, multi-rate, spatially
//! grounded physical world required by robotics, autonomous machines, AR/VR, and neural interfaces:
//!
//! 1. **[`HybridLogicalClock`] (`HLC`)**: Spanner/Cockroach/Zenoh-class causal physical+logical
//!    timestamps preserving strict happens-before ordering across unsynchronized edge sensors
//!    without atomic clocks.
//! 2. **[`EctStore`] (Rerun-style `Entity-Component-Time` + `kdb+`-class `ASOF` & `WINDOW` joins)**:
//!    Ingests multi-rate sensor/neural/actuator streams (`1000 Hz` IMU, `2000 Hz` EMG/EEG,
//!    `60 Hz` camera, `20 Hz` LiDAR) with LabStreamingLayer (`LSL`) clock-offset compensation,
//!    typed uncertainty (`value ± std_dev`), and $O(\log N)$ strictly causal `ASOF` alignment
//!    (never leaking future samples $t' > t$).
//! 3. **[`TransformTree`] (`SE(3)` Rigid-Body Frame Graph — ROS2 `tf2`-in-DB)**:
//!    Time-indexed 3D translations + unit quaternions with temporal SLERP/LERP interpolation and
//!    multi-hop kinematic chain composition (`world -> base_link -> arm -> gripper -> camera`).

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};
use std::sync::RwLock;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// 1. Hybrid Logical Clock (HLC)
// ============================================================================

/// Causal Hybrid Logical Timestamp `(wall_time_ns, logical, node_id)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct HlcTimestamp {
    /// Monotonic physical wall-clock estimate in nanoseconds since Unix epoch.
    pub wall_time_ns: u64,
    /// Logical tie-breaker incremented when multiple events occur at the same `wall_time_ns`.
    pub logical: u32,
    /// Originating sensor/ECU/node identifier.
    pub node_id: u32,
}

/// Thread-safe Hybrid Logical Clock enforcing bounded-drift causal ordering.
pub struct HybridLogicalClock {
    node_id: u32,
    max_drift_ns: u64,
    /// Packed `(wall_time_ns_top_48_or_split)` — stored as two atomics under a light spin/mutex
    /// or `RwLock<(u64, u32)>` for exact 96-bit atomicity.
    state: RwLock<(u64, u32)>,
}

impl HybridLogicalClock {
    /// Construct a new `HybridLogicalClock` for `node_id` rejecting remote clocks ahead by `> max_drift_ns`.
    pub fn new(node_id: u32, max_drift_ns: u64) -> Self {
        Self {
            node_id,
            max_drift_ns,
            state: RwLock::new((0, 0)),
        }
    }

    /// Mint a strictly monotonic local [`HlcTimestamp`] given observed local `physical_ns`.
    pub fn tick(&self, physical_ns: u64) -> HlcTimestamp {
        let mut guard = match self.state.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        if physical_ns > guard.0 {
            guard.0 = physical_ns;
            guard.1 = 0;
        } else {
            guard.1 = guard.1.saturating_add(1);
        }
        HlcTimestamp {
            wall_time_ns: guard.0,
            logical: guard.1,
            node_id: self.node_id,
        }
    }

    /// Merge a remote [`HlcTimestamp`] upon message receipt, advancing causal time or returning
    /// `Err` if the remote node's physical clock exceeds `physical_ns + max_drift_ns`.
    pub fn observe(
        &self,
        remote: HlcTimestamp,
        physical_ns: u64,
    ) -> Result<HlcTimestamp, String> {
        if remote.wall_time_ns > physical_ns.saturating_add(self.max_drift_ns) {
            return Err(format!(
                "HLC clock drift exceeded: remote={} ns > physical={} ns + max_drift={} ns",
                remote.wall_time_ns, physical_ns, self.max_drift_ns
            ));
        }
        let mut guard = match self.state.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let prev_wall = guard.0;
        let prev_log = guard.1;
        let next_wall = prev_wall.max(remote.wall_time_ns).max(physical_ns);
        let next_log = if next_wall == prev_wall && next_wall == remote.wall_time_ns {
            prev_log.max(remote.logical).saturating_add(1)
        } else if next_wall == prev_wall {
            prev_log.saturating_add(1)
        } else if next_wall == remote.wall_time_ns {
            remote.logical.saturating_add(1)
        } else {
            0
        };
        *guard = (next_wall, next_log);
        Ok(HlcTimestamp {
            wall_time_ns: next_wall,
            logical: next_log,
            node_id: self.node_id,
        })
    }
}

// ============================================================================
// 2. Entity-Component-Time (ECT) + kdb+-style Causal ASOF & Window Joins
// ============================================================================

/// Single time-indexed multi-dimensional component sample with physical uncertainty.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EctSample {
    /// Calibrated causal timestamp in nanoseconds (`raw_sensor_ns + clock_offset_ns`).
    pub timestamp_ns: u64,
    /// Dense numeric vector payload (e.g. `[ax, ay, az]` IMU or 8-band EEG power).
    pub values: Vec<f64>,
    /// 1-sigma measurement uncertainty (`0.0` if exact).
    pub std_dev: f64,
}

/// Join specification for [`EctStore::asof_align`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsofJoinSpec {
    /// Target entity path (e.g. `"robot/imu"` or `"human/bci"`).
    pub entity: String,
    /// Target component name (e.g. `"accel"` or `"motor_intent"`).
    pub component: String,
    /// Maximum allowed age `t_anchor - t_sample` in nanoseconds before treating as stale (`None`).
    pub max_staleness_ns: u64,
}

/// Single causally aligned multi-rate row returned by [`EctStore::asof_align`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsofAlignedRow {
    /// Anchor timestamp in nanoseconds.
    pub anchor_timestamp_ns: u64,
    /// Anchor component payload.
    pub anchor_values: Vec<f64>,
    /// Aligned right-hand-side component samples (`None` if no causal sample `<= anchor_timestamp_ns` within `max_staleness_ns`).
    pub aligned: Vec<Option<EctSample>>,
}

/// Summary statistics over a causal `[end_ns - window_ns, end_ns]` window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalWindowSummary {
    /// Number of samples in the causal window.
    pub sample_count: usize,
    /// Per-dimension arithmetic mean.
    pub mean: Vec<f64>,
    /// Per-dimension sample variance.
    pub variance: Vec<f64>,
    /// Propagated root-mean-square measurement uncertainty.
    pub combined_std_dev: f64,
}

/// Rerun-inspired `Entity-Component-Time` columnar ring store with `kdb+`-class `ASOF` alignment.
pub struct EctStore {
    max_samples_per_stream: usize,
    /// Optional LSL-style clock skew offset per entity (`calibrated_ns = raw_ns + offset_ns`).
    clock_offsets_ns: RwLock<HashMap<String, i64>>,
    streams: RwLock<HashMap<(String, String), Vec<EctSample>>>,
    ingested_samples: AtomicU64,
}

impl EctStore {
    /// Create a new `EctStore` retaining up to `max_samples_per_stream` per `(entity, component)`.
    pub fn new(max_samples_per_stream: usize) -> Self {
        Self {
            max_samples_per_stream: max_samples_per_stream.max(16),
            clock_offsets_ns: RwLock::new(HashMap::new()),
            streams: RwLock::new(HashMap::new()),
            ingested_samples: AtomicU64::new(0),
        }
    }

    /// Register a LabStreamingLayer (`LSL`)-style clock offset in nanoseconds for `entity`.
    pub fn set_clock_offset_ns(&self, entity: &str, offset_ns: i64) {
        if let Ok(mut map) = self.clock_offsets_ns.write() {
            map.insert(entity.to_string(), offset_ns);
        }
    }

    /// Ingest a raw timestamped component vector for `(entity, component)`.
    pub fn ingest(
        &self,
        entity: &str,
        component: &str,
        raw_timestamp_ns: u64,
        values: Vec<f64>,
        std_dev: f64,
    ) {
        let offset = self
            .clock_offsets_ns
            .read()
            .ok()
            .and_then(|m| m.get(entity).copied())
            .unwrap_or(0);
        let calibrated_ns = if offset >= 0 {
            raw_timestamp_ns.saturating_add(offset as u64)
        } else {
            raw_timestamp_ns.saturating_sub(offset.unsigned_abs())
        };

        let sample = EctSample {
            timestamp_ns: calibrated_ns,
            values,
            std_dev: std_dev.abs(),
        };

        if let Ok(mut map) = self.streams.write() {
            let vec = map
                .entry((entity.to_string(), component.to_string()))
                .or_default();
            // Fast path: append in-order; otherwise binary-search insert to keep sorted!
            if vec.last().is_none_or(|last| last.timestamp_ns <= calibrated_ns) {
                vec.push(sample);
            } else {
                let pos = vec.partition_point(|s| s.timestamp_ns <= calibrated_ns);
                vec.insert(pos, sample);
            }
            if vec.len() > self.max_samples_per_stream {
                let drop_n = vec.len() - self.max_samples_per_stream;
                vec.drain(0..drop_n);
            }
        }
        self.ingested_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// `kdb+`-style strictly causal `ASOF JOIN` (`aj`): for every sample of
    /// `(anchor_entity, anchor_component)` in `[start_ns, end_ns]`, match each `spec` in `specs`
    /// to the latest sample with `t_rhs <= t_anchor` and `t_anchor - t_rhs <= max_staleness_ns`.
    ///
    /// **Causality guarantee:** Future samples (`t_rhs > t_anchor`) are NEVER visible!
    pub fn asof_align(
        &self,
        anchor_entity: &str,
        anchor_component: &str,
        start_ns: u64,
        end_ns: u64,
        specs: &[AsofJoinSpec],
    ) -> Vec<AsofAlignedRow> {
        let Ok(map) = self.streams.read() else {
            return Vec::new();
        };
        let Some(anchor_stream) =
            map.get(&(anchor_entity.to_string(), anchor_component.to_string()))
        else {
            return Vec::new();
        };

        let lo = anchor_stream.partition_point(|s| s.timestamp_ns < start_ns);
        let hi = anchor_stream.partition_point(|s| s.timestamp_ns <= end_ns);

        let mut out = Vec::with_capacity(hi.saturating_sub(lo));
        for anchor in &anchor_stream[lo..hi] {
            let t = anchor.timestamp_ns;
            let mut aligned = Vec::with_capacity(specs.len());
            for spec in specs {
                let matched = map
                    .get(&(spec.entity.clone(), spec.component.clone()))
                    .and_then(|rhs| {
                        let idx = rhs.partition_point(|s| s.timestamp_ns <= t);
                        if idx == 0 {
                            None
                        } else {
                            let cand = &rhs[idx - 1];
                            if t.saturating_sub(cand.timestamp_ns) <= spec.max_staleness_ns {
                                Some(cand.clone())
                            } else {
                                None
                            }
                        }
                    });
                aligned.push(matched);
            }
            out.push(AsofAlignedRow {
                anchor_timestamp_ns: t,
                anchor_values: anchor.values.clone(),
                aligned,
            });
        }
        out
    }

    /// Compute causal window statistics over `[end_ns - window_ns, end_ns]` (`kdb+` `wj`).
    pub fn causal_window_summary(
        &self,
        entity: &str,
        component: &str,
        end_ns: u64,
        window_ns: u64,
    ) -> Option<CausalWindowSummary> {
        let map = self.streams.read().ok()?;
        let stream = map.get(&(entity.to_string(), component.to_string()))?;
        let start_ns = end_ns.saturating_sub(window_ns);
        let lo = stream.partition_point(|s| s.timestamp_ns < start_ns);
        let hi = stream.partition_point(|s| s.timestamp_ns <= end_ns);
        let slice = &stream[lo..hi];
        if slice.is_empty() {
            return None;
        }
        let dim = slice[0].values.len();
        let n = slice.len() as f64;
        let mut mean = vec![0.0f64; dim];
        let mut sq_err_sum = 0.0f64;
        for s in slice {
            for (d, &v) in s.values.iter().take(dim).enumerate() {
                mean[d] += v / n;
            }
            sq_err_sum += s.std_dev * s.std_dev;
        }
        let mut variance = vec![0.0f64; dim];
        for s in slice {
            for (d, &v) in s.values.iter().take(dim).enumerate() {
                let diff = v - mean[d];
                variance[d] += (diff * diff) / n;
            }
        }
        Some(CausalWindowSummary {
            sample_count: slice.len(),
            mean,
            variance,
            combined_std_dev: (sq_err_sum / (n * n)).sqrt(),
        })
    }

    /// Total samples ingested across all entity-component channels.
    pub fn total_ingested(&self) -> u64 {
        self.ingested_samples.load(Ordering::Relaxed)
    }
}

// ============================================================================
// 3. SE(3) Rigid-Body Spatial Transform Tree (`tf2`-in-DB)
// ============================================================================

/// 3D rigid-body pose in `SE(3)`: translation `[x, y, z]` + unit quaternion `[qx, qy, qz, qw]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Transform3D {
    /// Translation vector `[tx, ty, tz]` in meters.
    pub translation: [f64; 3],
    /// Unit rotation quaternion `[qx, qy, qz, qw]` (`qw` scalar last).
    pub rotation: [f64; 4],
}

impl Transform3D {
    /// Identity `SE(3)` transform.
    pub const fn identity() -> Self {
        Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Normalize the quaternion to unit length.
    pub fn normalized(mut self) -> Self {
        let [x, y, z, w] = self.rotation;
        let norm = (x * x + y * y + z * z + w * w).sqrt();
        if norm > 1e-12 {
            self.rotation = [x / norm, y / norm, z / norm, w / norm];
        } else {
            self.rotation = [0.0, 0.0, 0.0, 1.0];
        }
        self
    }

    /// Rotate a 3D vector `v` by this transform's unit quaternion.
    pub fn rotate_vec(&self, v: [f64; 3]) -> [f64; 3] {
        let [qx, qy, qz, qw] = self.rotation;
        // t = 2 * cross(q.xyz, v)
        let tx = 2.0 * (qy * v[2] - qz * v[1]);
        let ty = 2.0 * (qz * v[0] - qx * v[2]);
        let tz = 2.0 * (qx * v[1] - qy * v[0]);
        // v' = v + qw * t + cross(q.xyz, t)
        [
            v[0] + qw * tx + (qy * tz - qz * ty),
            v[1] + qw * ty + (qz * tx - qx * tz),
            v[2] + qw * tz + (qx * ty - qy * tx),
        ]
    }

    /// Apply rigid-body transform `R * p + t` to a 3D point.
    pub fn transform_point(&self, p: [f64; 3]) -> [f64; 3] {
        let r = self.rotate_vec(p);
        [
            r[0] + self.translation[0],
            r[1] + self.translation[1],
            r[2] + self.translation[2],
        ]
    }

    /// Compose `self * other` ($T_{AC} = T_{AB} \circ T_{BC}$).
    pub fn compose(&self, other: &Self) -> Self {
        let [ax, ay, az, aw] = self.rotation;
        let [bx, by, bz, bw] = other.rotation;
        let rot = [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ];
        let t_other_in_self = self.rotate_vec(other.translation);
        Self {
            translation: [
                self.translation[0] + t_other_in_self[0],
                self.translation[1] + t_other_in_self[1],
                self.translation[2] + t_other_in_self[2],
            ],
            rotation: rot,
        }
        .normalized()
    }

    /// Rigid-body inverse $T^{-1}$ such that `T.compose(&T.inverse()) == identity`.
    pub fn inverse(&self) -> Self {
        let [qx, qy, qz, qw] = self.rotation;
        let inv_rot = Self {
            translation: [0.0, 0.0, 0.0],
            rotation: [-qx, -qy, -qz, qw],
        };
        let neg_t = inv_rot.rotate_vec([
            -self.translation[0],
            -self.translation[1],
            -self.translation[2],
        ]);
        Self {
            translation: neg_t,
            rotation: inv_rot.rotation,
        }
    }

    /// Interpolate smoothly (`LERP` translation + shortest-arc `SLERP` rotation) at `alpha in [0, 1]`.
    pub fn interpolate(&self, other: &Self, alpha: f64) -> Self {
        let a = alpha.clamp(0.0, 1.0);
        let translation = [
            self.translation[0] + a * (other.translation[0] - self.translation[0]),
            self.translation[1] + a * (other.translation[1] - self.translation[1]),
            self.translation[2] + a * (other.translation[2] - self.translation[2]),
        ];

        let q1 = self.rotation;
        let mut q2 = other.rotation;
        let mut dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];
        if dot < 0.0 {
            dot = -dot;
            q2 = [-q2[0], -q2[1], -q2[2], -q2[3]];
        }

        let rotation = if dot > 0.9995 {
            [
                q1[0] + a * (q2[0] - q1[0]),
                q1[1] + a * (q2[1] - q1[1]),
                q1[2] + a * (q2[2] - q1[2]),
                q1[3] + a * (q2[3] - q1[3]),
            ]
        } else {
            let theta_0 = dot.clamp(-1.0, 1.0).acos();
            let theta = theta_0 * a;
            let sin_theta = theta.sin();
            let sin_theta_0 = theta_0.sin();
            let s0 = theta.cos() - dot * sin_theta / sin_theta_0;
            let s1 = sin_theta / sin_theta_0;
            [
                s0 * q1[0] + s1 * q2[0],
                s0 * q1[1] + s1 * q2[1],
                s0 * q1[2] + s1 * q2[2],
                s0 * q1[3] + s1 * q2[3],
            ]
        };

        Self {
            translation,
            rotation,
        }
        .normalized()
    }
}

/// Time-indexed `SE(3)` coordinate-frame transform graph (`ROS2 tf2`-in-DB).
#[derive(Default)]
pub struct TransformTree {
    /// `(parent_frame, child_frame) -> timestamp_ns -> Transform3D` (`T_parent_child`).
    edges: RwLock<HashMap<(String, String), BTreeMap<u64, Transform3D>>>,
    /// Undirected adjacency `frame -> Vec<(neighbor_frame, is_forward_parent_to_child)>`.
    adj: RwLock<HashMap<String, Vec<(String, bool)>>>,
}

impl TransformTree {
    /// Construct an empty `TransformTree`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record transform $T_{\text{parent} \leftarrow \text{child}}$ at `timestamp_ns`
    /// (maps points in `child_frame` coordinates into `parent_frame` coordinates).
    pub fn upsert_transform(
        &self,
        parent_frame: &str,
        child_frame: &str,
        timestamp_ns: u64,
        transform: Transform3D,
    ) {
        if let Ok(mut edges) = self.edges.write() {
            let key = (parent_frame.to_string(), child_frame.to_string());
            let is_new_edge = !edges.contains_key(&key);
            if is_new_edge && edges.len() >= 1024 {
                // Bound total distinct frame-pair edges to prevent unbounded memory growth
                return;
            }
            let timeline = edges.entry(key).or_default();
            timeline.insert(timestamp_ns, transform.normalized());
            while timeline.len() > 4096 {
                if let Some((&oldest_ts, _)) = timeline.iter().next() {
                    timeline.remove(&oldest_ts);
                } else {
                    break;
                }
            }
            drop(edges);

            if is_new_edge
                && let Ok(mut adj) = self.adj.write()
            {
                adj.entry(parent_frame.to_string())
                    .or_default()
                    .push((child_frame.to_string(), true));
                adj.entry(child_frame.to_string())
                    .or_default()
                    .push((parent_frame.to_string(), false));
            }
        }
    }

    fn sample_edge(
        timeline: &BTreeMap<u64, Transform3D>,
        at_time_ns: u64,
    ) -> Option<Transform3D> {
        let prev = timeline.range(..=at_time_ns).next_back();
        let next = timeline.range(at_time_ns..).next();
        match (prev, next) {
            (Some((&t0, tf0)), Some((&t1, tf1))) => {
                if t0 == t1 {
                    Some(*tf0)
                } else {
                    let alpha = (at_time_ns - t0) as f64 / (t1 - t0) as f64;
                    Some(tf0.interpolate(tf1, alpha))
                }
            }
            (Some((_, tf0)), None) => Some(*tf0), // Hold latest known pose causally
            (None, Some((_, tf1))) => Some(*tf1),
            (None, None) => None,
        }
    }

    /// Resolve $T_{\text{target} \leftarrow \text{source}}(t)$ across multi-link kinematic chains
    /// so `tf.transform_point(p_in_source)` yields coordinates in `target_frame` at `at_time_ns`.
    pub fn lookup_transform(
        &self,
        target_frame: &str,
        source_frame: &str,
        at_time_ns: u64,
    ) -> Option<Transform3D> {
        if target_frame == source_frame {
            return Some(Transform3D::identity());
        }
        let adj = self.adj.read().ok()?;
        let edges = self.edges.read().ok()?;

        // BFS from `target_frame` to `source_frame`, accumulating `T_{target <- cur}`
        let mut visited: HashMap<String, Transform3D> = HashMap::new();
        let mut queue: VecDeque<String> = VecDeque::new();
        visited.insert(target_frame.to_string(), Transform3D::identity());
        queue.push_back(target_frame.to_string());

        while let Some(cur) = queue.pop_front() {
            let t_target_cur = *visited.get(&cur)?;
            if let Some(neighbors) = adj.get(&cur) {
                for (nbr, cur_is_parent) in neighbors {
                    if visited.contains_key(nbr) {
                        continue;
                    }
                    let step_tf = if *cur_is_parent {
                        // Edge stored is `(cur, nbr)` = `T_{cur <- nbr}`
                        let tl = edges.get(&(cur.clone(), nbr.clone()))?;
                        Self::sample_edge(tl, at_time_ns)?
                    } else {
                        // Edge stored is `(nbr, cur)` = `T_{nbr <- cur}`, so `T_{cur <- nbr} = inv`
                        let tl = edges.get(&(nbr.clone(), cur.clone()))?;
                        Self::sample_edge(tl, at_time_ns)?.inverse()
                    };
                    let t_target_nbr = t_target_cur.compose(&step_tf);
                    if nbr == source_frame {
                        return Some(t_target_nbr);
                    }
                    visited.insert(nbr.clone(), t_target_nbr);
                    queue.push_back(nbr.clone());
                }
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hlc_monotonic_and_drift_guard() {
        let hlc_a = HybridLogicalClock::new(1, 5_000_000); // 5 ms max drift
        let t1 = hlc_a.tick(1_000_000);
        let t2 = hlc_a.tick(999_000); // NTP backward jump — HLC MUST stay strictly > t1!
        assert!(t2 > t1);

        let remote_ok = HlcTimestamp {
            wall_time_ns: 1_002_000,
            logical: 7,
            node_id: 2,
        };
        let t3 = hlc_a.observe(remote_ok, 1_000_000).unwrap();
        assert!(t3 > remote_ok);

        let remote_runaway = HlcTimestamp {
            wall_time_ns: 20_000_000,
            logical: 0,
            node_id: 9,
        };
        assert!(hlc_a.observe(remote_runaway, 1_000_000).is_err());
    }

    #[test]
    fn test_ect_strictly_causal_asof_join_and_lsl_clock_offset() {
        let store = EctStore::new(1024);
        // Wearable BCI clock lags robot master clock by +200 ns -> compensate!
        store.set_clock_offset_ns("human/bci", 200);

        // Camera frames at 1000 ns and 2000 ns
        store.ingest("robot/cam", "frame_id", 1_000, vec![1.0], 0.0);
        store.ingest("robot/cam", "frame_id", 2_000, vec![2.0], 0.0);

        // IMU at 950 ns, 1050 ns (AFTER cam 1000!), 1980 ns
        store.ingest("robot/imu", "accel_z", 950, vec![9.81], 0.02);
        store.ingest("robot/imu", "accel_z", 1_050, vec![12.40], 0.02);
        store.ingest("robot/imu", "accel_z", 1_980, vec![9.79], 0.02);

        // BCI raw 790 ns -> calibrated 990 ns (<= 1000 ns!)
        store.ingest("human/bci", "motor_alpha", 790, vec![0.88], 0.05);

        let aligned = store.asof_align(
            "robot/cam",
            "frame_id",
            0,
            3_000,
            &[
                AsofJoinSpec {
                    entity: "robot/imu".into(),
                    component: "accel_z".into(),
                    max_staleness_ns: 500,
                },
                AsofJoinSpec {
                    entity: "human/bci".into(),
                    component: "motor_alpha".into(),
                    max_staleness_ns: 500,
                },
            ],
        );

        assert_eq!(aligned.len(), 2);
        // At t=1000, IMU MUST match 950 ns (9.81), NEVER peeking at future 1050 ns (12.40)!
        assert_eq!(aligned[0].aligned[0].as_ref().unwrap().timestamp_ns, 950);
        assert_eq!(aligned[0].aligned[1].as_ref().unwrap().timestamp_ns, 990);
        // At t=2000, IMU matches 1980 ns, while BCI (990 ns, age 1010 > 500) is marked stale `None`!
        assert_eq!(aligned[1].aligned[0].as_ref().unwrap().timestamp_ns, 1_980);
        assert!(aligned[1].aligned[1].is_none());
    }

    #[test]
    fn test_se3_kinematic_chain_interpolation_and_inverse() {
        let tf = TransformTree::new();
        // base_link moves along +X from x=1.0 (t=0) to x=3.0 (t=1000) in `world`
        tf.upsert_transform(
            "world",
            "base_link",
            0,
            Transform3D {
                translation: [1.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        );
        tf.upsert_transform(
            "world",
            "base_link",
            1_000,
            Transform3D {
                translation: [3.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        );
        // gripper is +0.5m along +Z from `base_link`
        tf.upsert_transform(
            "base_link",
            "gripper",
            0,
            Transform3D {
                translation: [0.0, 0.0, 0.5],
                rotation: [0.0, 0.0, 0.0, 1.0],
            },
        );

        // Lookup `T_{world <- gripper}` at midpoint t=500 -> interpolated x=2.0, z=0.5!
        let t_wg = tf.lookup_transform("world", "gripper", 500).unwrap();
        let p_world = t_wg.transform_point([0.0, 0.0, 0.0]);
        assert!((p_world[0] - 2.0).abs() < 1e-9);
        assert!((p_world[2] - 0.5).abs() < 1e-9);

        // Lookup inverse direction `T_{gripper <- world}` maps `[2.0, 0.0, 0.5]` back to origin!
        let t_gw = tf.lookup_transform("gripper", "world", 500).unwrap();
        let p_grip = t_gw.transform_point(p_world);
        assert!(p_grip[0].abs() < 1e-9 && p_grip[1].abs() < 1e-9 && p_grip[2].abs() < 1e-9);
    }
}
