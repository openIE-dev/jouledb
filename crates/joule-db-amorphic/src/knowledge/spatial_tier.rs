//! Spatial tier of the materializer cascade.
//!
//! This is the **whole point** of the spatial-intelligence integration:
//! a tier that resolves 3D spatial queries via index probes (kd-tree /
//! R-tree / octree) at picojoule cost — *before* the cascade falls back
//! to LLM at millijoule cost.
//!
//! ## Where it sits in the cascade
//!
//! Materializer cascade (per `materializer.rs`):
//!
//! ```text
//!   1. Cache              (LUT, ~0.1 µJ)
//!   2. Skill              (LUT, ~0.1 µJ)
//!   3. Pattern-Lang       (LUT, ~1   µJ)
//!   4. Eigenbasis         (compute, varies)
//! → 4.5 SPATIAL-3D        (compute, ~500 pJ–5 µJ)   ← NEW
//!   5. Holographic        (compute, varies)
//!   6. flowR              (compute, varies)
//!   7. Neural             (~1 mJ per LLM token)
//! ```
//!
//! Spatial-3D slots in **before** Holographic / flowR / Neural because the
//! probe is deterministic, bounded, and orders of magnitude cheaper than
//! any of those tiers. A `NEAREST_GRASPABLE(scene, point)` query is one
//! kd-tree descent (~500 pJ); the LLM equivalent is "find me the nearest
//! object I can pick up given this point cloud" which is thousands of
//! tokens at ~mJ each. Same answer, ~10⁷× the energy.
//!
//! ## Typed entry point — not NL parsing
//!
//! The string-in/string-out `Materializer::materialize` API works for
//! casual NL queries, but spatial queries are **structured** — they have
//! coordinates, bboxes, k values. Adding NL parsing here would couple the
//! energy story to a separate parsing problem. Instead this module exposes
//! `Materializer::materialize_spatial(SpatialQuery)` — a typed entry that
//! returns a typed `SpatialResult` *and* a `MaterializeResult` with the
//! cascade source, energy, and metrics tracked.
//!
//! NL parsing can land on top later as a thin shim that produces a
//! `SpatialQuery` from text — that's the right factoring.

use std::time::Instant;

use joule_db_core::index::{KdTree3, Octree3, RTree3};
use joule_db_core::types::spatial::{Bbox3, Point3};

use super::materializer::{
    EntropyLevel, MaterializeResult, Materializer, Source,
};

// ============================================================================
// Scene
// ============================================================================

/// A spatial scene the cascade can probe.
///
/// Holds three indexes over the same conceptual data, each tuned for a
/// different query class:
///
/// - `points`: kd-tree over `Point3` keys → nearest / k-NN over object
///   centroids and point clouds.
/// - `extents`: R-tree over `Bbox3` keys → "what overlaps this region",
///   "what contains this point", "ray vs scene" pre-pass.
/// - `octree`: octree over `Point3` → hierarchical / LOD descent and
///   bounded-radius nearest-neighbor for streaming workloads.
///
/// Each index stores a `u32` object ID rather than a payload; the caller
/// keeps an external table mapping IDs → objects (string label, properties,
/// etc.). This keeps the index types thin and the energy cost per probe
/// dominated by tree descent, not payload copying.
#[derive(Debug, Clone)]
pub struct SpatialScene {
    points: KdTree3<u32>,
    extents: RTree3<u32>,
    octree: Octree3<u32>,
    /// External labels keyed by object ID. The cascade returns the label
    /// in `SpatialResult::Hits` so the caller doesn't need a side lookup.
    labels: Vec<String>,
}

/// One object in a `SpatialScene`.
#[derive(Debug, Clone)]
pub struct SpatialObject {
    pub label: String,
    /// Centroid of the object (used as the kd-tree / octree key).
    pub centroid: Point3,
    /// Bounding box of the object (used as the R-tree key).
    pub bbox: Bbox3,
}

impl SpatialScene {
    /// Build a scene from a list of objects + a world bbox for the octree.
    ///
    /// `world` should fully contain every object; out-of-bounds points are
    /// dropped from the octree (other indexes are unaffected).
    pub fn build(world: Bbox3, objects: Vec<SpatialObject>) -> Self {
        // Assign sequential IDs and produce parallel arrays for each index.
        let mut labels = Vec::with_capacity(objects.len());
        let mut kd_pts: Vec<(Point3, u32)> = Vec::with_capacity(objects.len());
        let mut rt_boxes: Vec<(Bbox3, u32)> = Vec::with_capacity(objects.len());
        let mut oct_pts: Vec<(Point3, u32)> = Vec::with_capacity(objects.len());

        for (i, obj) in objects.into_iter().enumerate() {
            let id = i as u32;
            labels.push(obj.label);
            kd_pts.push((obj.centroid, id));
            rt_boxes.push((obj.bbox, id));
            oct_pts.push((obj.centroid, id));
        }

        Self {
            points: KdTree3::build(kd_pts),
            extents: RTree3::build(rt_boxes),
            octree: Octree3::build(world, oct_pts),
            labels,
        }
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Look up the label for an object ID.
    pub fn label(&self, id: u32) -> Option<&str> {
        self.labels.get(id as usize).map(String::as_str)
    }
}

// ============================================================================
// Query / result
// ============================================================================

/// A typed spatial query the cascade can resolve at picojoule cost.
///
/// Each variant maps to a specific index probe — see `flowG::Spatial3dOp`
/// for the full operation set this draws from.
#[derive(Debug, Clone)]
pub enum SpatialQuery {
    /// Single nearest object to `point` — kd-tree probe.
    /// Maps to `Spatial3dOp::KnnQuery3d` with k=1.
    Nearest { point: Point3 },

    /// k nearest objects to `point` — kd-tree probe.
    /// Maps to `Spatial3dOp::KnnQuery3d`.
    Knn { point: Point3, k: usize },

    /// All objects whose centroid lies in `bbox` — kd-tree range descent.
    /// Maps to `Spatial3dOp::RangeQuery3d`.
    PointsInBox { bbox: Bbox3 },

    /// All objects whose bbox overlaps `bbox` — R-tree probe.
    /// Maps to `Spatial3dOp::SceneNeighborQuery`.
    BoxesIntersecting { bbox: Bbox3 },

    /// All objects whose bbox contains `point` — R-tree probe.
    /// Maps to `Spatial3dOp::SceneRelationEval` ("CONTAINS" predicate).
    BoxesContaining { point: Point3 },

    /// Bounded-radius nearest neighbor via the octree (LOD-friendly,
    /// for streaming or partial scenes). Maps to `Spatial3dOp::OctreeQuery`.
    NearestWithin { point: Point3, max_dist: f64 },
}

/// Spatial query result — typed counterpart to the materializer's string output.
#[derive(Debug, Clone)]
pub enum SpatialResult {
    /// `Nearest` / `NearestWithin` succeeded.
    Hit { id: u32, label: String, distance: f64 },
    /// k-NN or range query — list of `(id, label)` pairs in result order.
    Hits(Vec<(u32, String)>),
    /// Query did not match (e.g. `NearestWithin` outside `max_dist`).
    Empty,
}

// ============================================================================
// Energy model
// ============================================================================

/// Per-probe energy cost in joules. Mirrors the calibration in
/// `inv-ai-codegraph::flowg::energy::Spatial3dOp` (~300 pJ baseline,
/// scaling with index descent depth).
///
/// These are deliberately conservative — even at 10× the calibrated cost,
/// the spatial tier is still 5+ orders of magnitude cheaper than a single
/// LLM token. The whole point is to make picking the spatial tier always
/// the right call when the query has a structured spatial answer.
mod energy {
    /// kd-tree nearest / k-NN — O(log N) descent, ~300 pJ per probe.
    pub const KD_PROBE_J: f64 = 0.000_000_000_300;
    /// R-tree intersect / contains — O(log_M N) descent, ~500 pJ per probe.
    pub const RTREE_PROBE_J: f64 = 0.000_000_000_500;
    /// Octree LOD descent + bounded NN — O(log_8 N), ~500 pJ per probe.
    pub const OCTREE_PROBE_J: f64 = 0.000_000_000_500;
    /// Per-result-row marshalling cost (label clone + Vec push).
    pub const PER_ROW_J: f64 = 0.000_000_000_010;
}

// ============================================================================
// Resolution
// ============================================================================

impl SpatialScene {
    /// Resolve a `SpatialQuery` against this scene. Returns the typed
    /// result *and* the picojoule cost so the materializer can attribute
    /// it correctly in its metrics.
    pub fn resolve(&self, query: &SpatialQuery) -> (SpatialResult, f64) {
        match query {
            SpatialQuery::Nearest { point } => match self.points.nearest(*point) {
                Some((_, &id, distance)) => {
                    let label = self.labels[id as usize].clone();
                    (
                        SpatialResult::Hit { id, label, distance },
                        energy::KD_PROBE_J + energy::PER_ROW_J,
                    )
                }
                None => (SpatialResult::Empty, energy::KD_PROBE_J),
            },

            SpatialQuery::Knn { point, k } => {
                let raw = self.points.knn(*point, *k);
                let hits: Vec<(u32, String)> = raw
                    .iter()
                    .map(|&(_, &id, _)| (id, self.labels[id as usize].clone()))
                    .collect();
                let n = hits.len() as f64;
                (
                    SpatialResult::Hits(hits),
                    energy::KD_PROBE_J + n * energy::PER_ROW_J,
                )
            }

            SpatialQuery::PointsInBox { bbox } => {
                let raw = self.points.range(*bbox);
                let hits: Vec<(u32, String)> = raw
                    .iter()
                    .map(|&(_, &id)| (id, self.labels[id as usize].clone()))
                    .collect();
                let n = hits.len() as f64;
                (
                    SpatialResult::Hits(hits),
                    energy::KD_PROBE_J + n * energy::PER_ROW_J,
                )
            }

            SpatialQuery::BoxesIntersecting { bbox } => {
                let raw = self.extents.intersects(*bbox);
                let hits: Vec<(u32, String)> = raw
                    .iter()
                    .map(|&(_, &id)| (id, self.labels[id as usize].clone()))
                    .collect();
                let n = hits.len() as f64;
                (
                    SpatialResult::Hits(hits),
                    energy::RTREE_PROBE_J + n * energy::PER_ROW_J,
                )
            }

            SpatialQuery::BoxesContaining { point } => {
                let raw = self.extents.contains_point(*point);
                let hits: Vec<(u32, String)> = raw
                    .iter()
                    .map(|&(_, &id)| (id, self.labels[id as usize].clone()))
                    .collect();
                let n = hits.len() as f64;
                (
                    SpatialResult::Hits(hits),
                    energy::RTREE_PROBE_J + n * energy::PER_ROW_J,
                )
            }

            SpatialQuery::NearestWithin { point, max_dist } => {
                match self.octree.nearest_within(*point, *max_dist) {
                    Some((_, &id, distance)) => {
                        let label = self.labels[id as usize].clone();
                        (
                            SpatialResult::Hit { id, label, distance },
                            energy::OCTREE_PROBE_J + energy::PER_ROW_J,
                        )
                    }
                    None => (SpatialResult::Empty, energy::OCTREE_PROBE_J),
                }
            }
        }
    }
}

// ============================================================================
// Materializer integration
// ============================================================================

/// Output format used when bridging `SpatialResult` into the string-typed
/// `MaterializeResult` for cascade-wide metrics. Callers using the typed
/// path (`materialize_spatial_full`) get both the typed result and the
/// `MaterializeResult` directly.
fn format_spatial_result(scene_size: usize, r: &SpatialResult) -> String {
    match r {
        SpatialResult::Hit { id, label, distance } => {
            format!("nearest: id={} label={} distance={:.6}", id, label, distance)
        }
        SpatialResult::Hits(hits) => {
            let summary: Vec<String> = hits
                .iter()
                .take(5)
                .map(|(id, label)| format!("{}={}", id, label))
                .collect();
            let more = if hits.len() > 5 {
                format!(" (+{} more)", hits.len() - 5)
            } else {
                String::new()
            };
            format!(
                "spatial_hits: {}/{} [{}]{}",
                hits.len(),
                scene_size,
                summary.join(", "),
                more
            )
        }
        SpatialResult::Empty => "spatial_empty".to_string(),
    }
}

impl Materializer {
    /// Attach a spatial scene to this materializer.
    ///
    /// The cascade can now resolve `SpatialQuery` at picojoule cost via
    /// `materialize_spatial`. Replaces any previously-attached scene.
    pub fn set_spatial_scene(&mut self, scene: SpatialScene) {
        self.spatial_scene = Some(scene);
    }

    /// Whether a spatial scene is attached.
    pub fn has_spatial_scene(&self) -> bool {
        self.spatial_scene.is_some()
    }

    /// Resolve a typed `SpatialQuery` through the cascade.
    ///
    /// Returns `None` if no spatial scene is attached. Records the result
    /// in the materializer's metrics under `Source::Spatial3d` so the
    /// "% avoided neural" headline number reflects spatial wins.
    pub fn materialize_spatial(
        &mut self,
        query: &SpatialQuery,
    ) -> Option<(SpatialResult, MaterializeResult)> {
        let scene = self.spatial_scene.as_ref()?;
        let start = Instant::now();
        self.metrics.total += 1;

        // Spatial queries are always Low entropy from the cascade's
        // perspective: deterministic, bounded, structured. They never
        // need novel composition.
        let entropy = EntropyLevel::Low;
        *self
            .metrics
            .by_entropy
            .entry(format!("{:?}", entropy))
            .or_insert(0) += 1;

        let (result, energy_j) = scene.resolve(query);
        self.metrics.total_energy += energy_j;
        *self
            .metrics
            .by_source
            .entry("spatial_3d".into())
            .or_insert(0) += 1;

        let output = format_spatial_result(scene.len(), &result);
        let actual_pj = (energy_j * 1e12).max(0.0) as u64;
        let mat_result = MaterializeResult {
            output,
            source: Source::Spatial3d,
            entropy,
            verified: true,
            energy_joules: energy_j,
            elapsed_us: start.elapsed().as_micros() as u64,
            receipt: super::energy_receipt::EnergyReceipt::for_tier(
                Source::Spatial3d,
                actual_pj,
            ),
        };

        Some((result, mat_result))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f64, y: f64, z: f64) -> Point3 {
        Point3::new(x, y, z)
    }
    fn b(min: (f64, f64, f64), max: (f64, f64, f64)) -> Bbox3 {
        Bbox3::new(p(min.0, min.1, min.2), p(max.0, max.1, max.2))
    }

    fn demo_scene() -> SpatialScene {
        let world = b((0.0, 0.0, 0.0), (100.0, 100.0, 100.0));
        let objs = vec![
            SpatialObject {
                label: "mug".into(),
                centroid: p(10.0, 10.0, 5.0),
                bbox: b((9.5, 9.5, 0.0), (10.5, 10.5, 10.0)),
            },
            SpatialObject {
                label: "kettle".into(),
                centroid: p(20.0, 10.0, 7.0),
                bbox: b((19.0, 9.0, 0.0), (21.0, 11.0, 14.0)),
            },
            SpatialObject {
                label: "lamp".into(),
                centroid: p(50.0, 50.0, 30.0),
                bbox: b((49.0, 49.0, 0.0), (51.0, 51.0, 60.0)),
            },
            SpatialObject {
                label: "table".into(),
                centroid: p(50.0, 50.0, 5.0),
                bbox: b((30.0, 30.0, 0.0), (70.0, 70.0, 10.0)),
            },
        ];
        SpatialScene::build(world, objs)
    }

    #[test]
    fn nearest_returns_closest_object() {
        let scene = demo_scene();
        let (r, _) = scene.resolve(&SpatialQuery::Nearest { point: p(11.0, 10.0, 5.0) });
        match r {
            SpatialResult::Hit { label, distance, .. } => {
                assert_eq!(label, "mug");
                assert!((distance - 1.0).abs() < 1e-9);
            }
            other => panic!("expected Hit, got {:?}", other),
        }
    }

    #[test]
    fn knn_returns_k_in_distance_order() {
        let scene = demo_scene();
        let (r, _) = scene.resolve(&SpatialQuery::Knn { point: p(15.0, 10.0, 6.0), k: 2 });
        match r {
            SpatialResult::Hits(hits) => {
                assert_eq!(hits.len(), 2);
                let labels: Vec<&str> = hits.iter().map(|(_, l)| l.as_str()).collect();
                // mug and kettle are both close to (15,10,6); mug at dist √(25+0+1)=5.099,
                // kettle at √(25+0+1)=5.099 — both equidistant. Just check they're the two.
                assert!(labels.contains(&"mug"));
                assert!(labels.contains(&"kettle"));
            }
            other => panic!("expected Hits, got {:?}", other),
        }
    }

    #[test]
    fn boxes_containing_point_uses_rtree() {
        let scene = demo_scene();
        // (50, 50, 5) is inside the table bbox AND the lamp bbox (lamp z spans 0..60).
        let (r, _) = scene.resolve(&SpatialQuery::BoxesContaining { point: p(50.0, 50.0, 5.0) });
        match r {
            SpatialResult::Hits(hits) => {
                let mut labels: Vec<&str> = hits.iter().map(|(_, l)| l.as_str()).collect();
                labels.sort();
                assert_eq!(labels, vec!["lamp", "table"]);
            }
            other => panic!("expected Hits, got {:?}", other),
        }
    }

    #[test]
    fn boxes_intersecting_query_uses_rtree() {
        let scene = demo_scene();
        // Box around the table area should overlap table + lamp + nothing else.
        let q = b((40.0, 40.0, 0.0), (60.0, 60.0, 8.0));
        let (r, _) = scene.resolve(&SpatialQuery::BoxesIntersecting { bbox: q });
        match r {
            SpatialResult::Hits(hits) => {
                let mut labels: Vec<&str> = hits.iter().map(|(_, l)| l.as_str()).collect();
                labels.sort();
                assert_eq!(labels, vec!["lamp", "table"]);
            }
            other => panic!("expected Hits, got {:?}", other),
        }
    }

    #[test]
    fn nearest_within_uses_octree() {
        let scene = demo_scene();
        // Tight radius around the mug.
        let (r, _) = scene.resolve(&SpatialQuery::NearestWithin {
            point: p(10.0, 10.0, 5.0),
            max_dist: 0.5,
        });
        match r {
            SpatialResult::Hit { label, .. } => assert_eq!(label, "mug"),
            other => panic!("expected Hit, got {:?}", other),
        }

        // Far away from anything.
        let (r, _) = scene.resolve(&SpatialQuery::NearestWithin {
            point: p(99.0, 99.0, 99.0),
            max_dist: 5.0,
        });
        assert!(matches!(r, SpatialResult::Empty));
    }

    #[test]
    fn energy_per_probe_is_picojoule_scale() {
        let scene = demo_scene();
        let (_, e) = scene.resolve(&SpatialQuery::Nearest { point: p(11.0, 10.0, 5.0) });
        // Single probe must be < 10 nJ — 10⁵× cheaper than 1 mJ LLM token.
        assert!(e < 10e-9, "spatial probe energy {} J too high", e);
        // And nonzero.
        assert!(e > 0.0);
    }

    #[test]
    fn materializer_records_spatial_source() {
        let mut m = Materializer::new();
        m.set_spatial_scene(demo_scene());
        assert!(m.has_spatial_scene());

        let (_, mat) = m
            .materialize_spatial(&SpatialQuery::Nearest { point: p(11.0, 10.0, 5.0) })
            .unwrap();
        assert_eq!(mat.source, Source::Spatial3d);
        assert_eq!(mat.entropy, EntropyLevel::Low);
        assert!(mat.verified);
        assert!(mat.energy_joules < 10e-9);
        assert!(mat.output.contains("mug"));

        // Metrics tracked.
        assert_eq!(m.metrics.total, 1);
        assert_eq!(*m.metrics.by_source.get("spatial_3d").unwrap(), 1);
    }

    #[test]
    fn materialize_spatial_returns_none_without_scene() {
        let mut m = Materializer::new();
        let r = m.materialize_spatial(&SpatialQuery::Nearest { point: p(0.0, 0.0, 0.0) });
        assert!(r.is_none());
    }

    #[test]
    fn cascade_99pct_neural_avoidance_with_spatial_workload() {
        // The headline benchmark: a robotic-perception-shaped workload
        // (lots of "what's near me", "what's in the gripper envelope")
        // resolved entirely at the spatial tier. Compare against the
        // baseline of going straight to LLM at ~1 mJ per query.
        let mut m = Materializer::new();
        m.set_spatial_scene(demo_scene());

        // 100 spatial probes covering the four query types.
        for i in 0..25 {
            let f = i as f64 * 0.5;
            let _ = m.materialize_spatial(&SpatialQuery::Nearest {
                point: p(10.0 + f, 10.0, 5.0),
            });
            let _ = m.materialize_spatial(&SpatialQuery::Knn {
                point: p(50.0, 50.0, 5.0 + f),
                k: 3,
            });
            let _ = m.materialize_spatial(&SpatialQuery::BoxesContaining {
                point: p(50.0, 50.0, f),
            });
            let _ = m.materialize_spatial(&SpatialQuery::BoxesIntersecting {
                bbox: b((10.0, 10.0, 0.0), (20.0 + f, 20.0, 10.0)),
            });
        }

        assert_eq!(m.metrics.total, 100);
        // 100% spatial wins — neural avoidance must be 100%.
        assert_eq!(m.metrics.pct_avoided_neural(), 100.0);
        // Total energy must be << 100 LLM-token equivalents.
        // 100 LLM tokens at 1 mJ each = 100 mJ = 0.1 J.
        // 100 spatial probes at <10 nJ each = <1 µJ = 1e-6 J.
        // → at least 5 orders of magnitude.
        assert!(m.metrics.total_energy < 1e-5,
            "total spatial energy {} J should be << 0.1 J LLM equivalent",
            m.metrics.total_energy);

        // Per-probe average must be in the picojoule–nanojoule range.
        let avg = m.metrics.total_energy / m.metrics.total as f64;
        assert!(avg < 100e-9, "avg per-probe energy {} J too high", avg);
    }
}
