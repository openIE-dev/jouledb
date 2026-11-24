//! Unified Embodied AI, Human-Machine Interface (`SigQL`), Living Metabolism & `DBSP` HTTP Plane
//!
//! Exposes JouleDB's Next-Era Living & Physical-AI organs directly on `Server::router()`:
//! - **Embodied `ECT` + `kdb+` Causal `ASOF` Alignment**:
//!   `POST /api/v1/embodied/ingest`, `POST /api/v1/embodied/asof`
//! - **`SE(3)` Kinematic Transform Tree (`tf2`-in-DB)**:
//!   `POST /api/v1/embodied/tf/upsert`, `POST /api/v1/embodied/tf/lookup`
//! - **Living Computing (`SurpriseGate` + Slow-Wave Consolidation)**:
//!   `POST /api/v1/living/perceive`, `POST /api/v1/living/consolidate`
//! - **Human-to-Machine Interface (`SigQL` Frequency & Uncertainty Stream Decoder)**:
//!   `POST /api/v1/hmi/sigql`

use axum::{Json, Router, extract::State, http::StatusCode, response::IntoResponse, routing::post};
use joule_db_core::embodied::{
    AsofJoinSpec, EctStore, HybridLogicalClock, Transform3D, TransformTree,
};
use joule_db_core::living::{HomeostaticMetabolism, PerceptionOutcome, SensoryFrame, SurpriseGate};
use serde::Deserialize;
use sigql::compile::Target;
use sigql::runtime::{OutputValue, Runtime, RuntimeConfig};
use sigql::types::DynSignal;
use std::sync::Arc;

/// Shared state for Embodied AI, HMI (`SigQL`), and Living Computing endpoints.
pub struct LivingEmbodiedRuntime {
    pub hlc: HybridLogicalClock,
    pub ect: EctStore,
    pub tf_tree: TransformTree,
    pub surprise_gate: SurpriseGate,
    pub metabolism: HomeostaticMetabolism,
}

impl Default for LivingEmbodiedRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl LivingEmbodiedRuntime {
    pub fn new() -> Self {
        Self {
            hlc: HybridLogicalClock::new(1, 50_000_000), // 50 ms max distributed clock skew
            ect: EctStore::new(16_384),
            tf_tree: TransformTree::new(),
            surprise_gate: SurpriseGate::new(0.10, 12.0, 8),
            metabolism: HomeostaticMetabolism::new(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct EctIngestReq {
    pub entity: String,
    pub component: String,
    pub timestamp_ns: u64,
    pub values: Vec<f64>,
    #[serde(default)]
    pub std_dev: f64,
}

#[derive(Debug, Deserialize)]
pub struct EctAsofReq {
    pub anchor_entity: String,
    pub anchor_component: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub joins: Vec<AsofJoinSpec>,
}

#[derive(Debug, Deserialize)]
pub struct TfUpsertReq {
    pub parent_frame: String,
    pub child_frame: String,
    pub timestamp_ns: u64,
    pub translation: [f64; 3],
    pub rotation: [f64; 4],
}

#[derive(Debug, Deserialize)]
pub struct TfLookupReq {
    pub target_frame: String,
    pub source_frame: String,
    pub at_time_ns: u64,
    #[serde(default)]
    pub point_in_source: Option<[f64; 3]>,
}

#[derive(Debug, Deserialize)]
pub struct PerceiveReq {
    pub channel: String,
    pub timestamp_ns: u64,
    pub values: Vec<f64>,
}

#[derive(Debug, Deserialize)]
pub struct ConsolidateReq {
    #[serde(default = "default_min_surprise")]
    pub min_retain_surprise: f64,
}

fn default_min_surprise() -> f64 {
    10.0
}

#[derive(Debug, Deserialize)]
pub struct HmiSigqlReq {
    pub signal_name: String,
    pub sample_rate_hz: f64,
    pub samples: Vec<f64>,
    pub query: String,
}

/// Build the Axum sub-router for Embodied AI, Living Computing, and HMI (`SigQL`).
pub fn living_embodied_routes(rt: Arc<LivingEmbodiedRuntime>) -> Router {
    Router::new()
        .route(
            "/api/v1/embodied/ingest",
            post(
                |State(rt): State<Arc<LivingEmbodiedRuntime>>, Json(req): Json<EctIngestReq>| async move {
                    let hlc_ts = rt.hlc.tick(req.timestamp_ns);
                    rt.ect.ingest(
                        &req.entity,
                        &req.component,
                        req.timestamp_ns,
                        req.values,
                        req.std_dev,
                    );
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "ok": true,
                            "hlc": hlc_ts,
                            "total_ingested": rt.ect.total_ingested(),
                        })),
                    )
                        .into_response()
                },
            ),
        )
        .route(
            "/api/v1/embodied/asof",
            post(
                |State(rt): State<Arc<LivingEmbodiedRuntime>>, Json(req): Json<EctAsofReq>| async move {
                    let aligned = rt.ect.asof_align(
                        &req.anchor_entity,
                        &req.anchor_component,
                        req.start_ns,
                        req.end_ns,
                        &req.joins,
                    );
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "ok": true,
                            "rows": aligned,
                            "energy_uj": 1 + (aligned.len() as u64) / 8,
                        })),
                    )
                        .into_response()
                },
            ),
        )
        .route(
            "/api/v1/embodied/tf/upsert",
            post(
                |State(rt): State<Arc<LivingEmbodiedRuntime>>, Json(req): Json<TfUpsertReq>| async move {
                    rt.tf_tree.upsert_transform(
                        &req.parent_frame,
                        &req.child_frame,
                        req.timestamp_ns,
                        Transform3D {
                            translation: req.translation,
                            rotation: req.rotation,
                        },
                    );
                    (StatusCode::OK, Json(serde_json::json!({ "ok": true }))).into_response()
                },
            ),
        )
        .route(
            "/api/v1/embodied/tf/lookup",
            post(
                |State(rt): State<Arc<LivingEmbodiedRuntime>>, Json(req): Json<TfLookupReq>| async move {
                    match rt.tf_tree.lookup_transform(
                        &req.target_frame,
                        &req.source_frame,
                        req.at_time_ns,
                    ) {
                        Some(tf) => {
                            let transformed_point =
                                req.point_in_source.map(|p| tf.transform_point(p));
                            (
                                StatusCode::OK,
                                Json(serde_json::json!({
                                    "ok": true,
                                    "transform": tf,
                                    "transformed_point": transformed_point,
                                })),
                            )
                                .into_response()
                        }
                        None => (
                            StatusCode::NOT_FOUND,
                            Json(serde_json::json!({ "ok": false, "error": "transform path not found" })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/living/perceive",
            post(
                |State(rt): State<Arc<LivingEmbodiedRuntime>>, Json(req): Json<PerceiveReq>| async move {
                    let outcome = rt.surprise_gate.perceive(
                        &req.channel,
                        SensoryFrame {
                            timestamp_ns: req.timestamp_ns,
                            values: req.values,
                        },
                    );
                    if let PerceptionOutcome::PromotedEpisode(ref ep) = outcome {
                        rt.metabolism.stage_episode(ep.clone());
                    }
                    let (obs, hab, pro) = rt.surprise_gate.compression_telemetry();
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "ok": true,
                            "outcome": outcome,
                            "telemetry": {
                                "observations": obs,
                                "habituated": hab,
                                "promoted_episodes": pro,
                            }
                        })),
                    )
                        .into_response()
                },
            ),
        )
        .route(
            "/api/v1/living/consolidate",
            post(
                |State(rt): State<Arc<LivingEmbodiedRuntime>>, Json(req): Json<ConsolidateReq>| async move {
                    let report = rt
                        .metabolism
                        .consolidate_sleep_cycle(req.min_retain_surprise);
                    let prototypes = rt.metabolism.prototypes();
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "ok": true,
                            "report": report,
                            "prototypes": prototypes,
                        })),
                    )
                        .into_response()
                },
            ),
        )
        .route(
            "/api/v1/hmi/sigql",
            post(|Json(req): Json<HmiSigqlReq>| async move {
                let parsed = match sigql::parse(&req.query) {
                    Ok(q) => q,
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response();
                    }
                };
                let sample_rate_u32 = req.sample_rate_hz.max(1.0).round() as u32;
                let mut compiler = sigql::Compiler::new(sigql::CompilerConfig {
                    target: Target::Simd,
                    default_sample_rate: sample_rate_u32,
                    default_fft_size: req.samples.len().next_power_of_two().max(64),
                    ..sigql::CompilerConfig::default()
                });
                let plan = match compiler.compile(&parsed) {
                    Ok(p) => p,
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response();
                    }
                };
                let mut runtime = Runtime::new(RuntimeConfig {
                    default_sample_rate: sample_rate_u32,
                    ..RuntimeConfig::default()
                });
                let sig = DynSignal::new(&req.signal_name, req.samples, sample_rate_u32, 0);
                runtime.register_signal(&req.signal_name, sig);

                match runtime.execute(&plan) {
                    Ok(res) => {
                        let mut fields = serde_json::Map::new();
                        for (name, val) in res.outputs {
                            let encoded = match val {
                                OutputValue::Scalar(uv) => serde_json::json!({
                                    "kind": "scalar",
                                    "value": uv.value,
                                    "confidence": uv.confidence,
                                    "lower_bound": uv.lower_bound,
                                    "upper_bound": uv.upper_bound,
                                    "n_samples": uv.n_samples,
                                }),
                                OutputValue::Signal(s) => serde_json::json!({
                                    "kind": "signal",
                                    "sample_rate": s.sample_rate,
                                    "len": s.samples.len(),
                                    "channel": s.channel.as_str(),
                                }),
                                OutputValue::Spectrum(sp) => serde_json::json!({
                                    "kind": "spectrum",
                                    "bins": sp.len(),
                                    "magnitudes": sp,
                                }),
                            };
                            fields.insert(name.to_string(), encoded);
                        }
                        (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "stats": {
                                    "execution_time_ns": res.stats.execution_time_ns,
                                    "samples_processed": res.stats.samples_processed,
                                    "memory_used": res.stats.memory_used,
                                },
                                "output": {
                                    "fields": fields,
                                },
                            })),
                        )
                            .into_response()
                    }
                    Err(e) => (
                        StatusCode::UNPROCESSABLE_ENTITY,
                        Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                    )
                        .into_response(),
                }
            }),
        )
        .with_state(rt)
}
