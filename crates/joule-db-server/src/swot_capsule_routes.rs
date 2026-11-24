//! Live HTTP Surface for:
//! 1. **`JouleCapsule` — Post-Docker Zero-Bloat OCI & Compose Engine** (`/api/v1/capsules/*`,
//!    including pure-Rust HTTPS OCI Distribution v2 registry pull without `dockerd`!)
//! 2. **360° Global Database SWOT Parity Suite** (`/api/v1/hybrid/*`, `/api/v1/graphrag/*`,
//!    `/api/v1/semantic-layer/*`, `/api/v1/langcache/*`).

use crate::query::{QueryExecutor, QueryRequest};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use base64::Engine;
use joule_db_branch::{
    agentfs::JouleAgentFs,
    capsule::{
        CapsuleEngine, ComposeSpec, OciImageConfig, OciImageIndex, OciImageRef, OciManifestV2,
        SpawnCapsuleRequest,
    },
    joulespace::{
        ActorKind, CloudSandboxHandoffBundle, JouleSpaceEngine, LwwRoomEntry, MeshPeer,
        NetworkEgressPolicy, RoomParticipant, ThreadMessage, ThreadStatus,
    },
    manager::BranchManager,
};
use joule_db_query::hybrid_graphrag::{
    CertifiedMetric, HippoRagEngine, HybridDocument, MetricQueryRequest, RrfConfig,
    SemanticCatalog, SemanticEntity, SemanticResponseCache, TriModalRrfIndex, VerifiedJoinEdge,
};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

/// Shared runtime state for `JouleCapsule`, `JouleSpace`, + 360° Global Ecosystem Parity endpoints.
pub struct SwotCapsuleRuntime {
    /// Post-Docker OCI + CoW `JouleCapsule` engine.
    pub capsules: CapsuleEngine,
    /// Post-Git (`Zed Delta` + `Jujutsu`), Docker Sandboxes, LiveKit Room & Mesh `JouleSpace` engine.
    pub space: JouleSpaceEngine,
    /// Scratch directory for ephemeral rootfs materialization during `exec_in_capsule`.
    pub scratch_dir: PathBuf,
    /// Tri-Modal (`BM25 ⋈ RaBitQ ⋈ HDC`) + ACORN-2-hop index.
    pub hybrid_index: RwLock<TriModalRrfIndex>,
    /// Governed Semantic Metric & Verified Join-Path catalog.
    pub semantic_catalog: RwLock<SemanticCatalog>,
    /// LSN-consistent `LangCache`-class semantic response cache.
    pub lang_cache: RwLock<SemanticResponseCache>,
    /// SQL query executor for running compiled governed semantic-layer SQL.
    pub executor: Arc<dyn QueryExecutor>,
    /// HTTP client for pure-Rust OCI Distribution v2 registry pulls.
    pub http_client: reqwest::Client,
}

impl SwotCapsuleRuntime {
    /// Construct a new runtime sharing the server's `BranchManager` and `QueryExecutor`.
    pub fn new(
        branch_manager: Arc<BranchManager>,
        executor: Arc<dyn QueryExecutor>,
        db_path: &str,
    ) -> Arc<Self> {
        let agentfs = Arc::new(JouleAgentFs::new(branch_manager));
        let scratch_dir = PathBuf::from(format!("{db_path}/capsule-scratch"));
        let _ = std::fs::create_dir_all(&scratch_dir);
        Arc::new(Self {
            capsules: CapsuleEngine::new(agentfs.clone()),
            space: JouleSpaceEngine::new(agentfs),
            scratch_dir,
            hybrid_index: RwLock::new(TriModalRrfIndex::new(64)),
            semantic_catalog: RwLock::new(SemanticCatalog::new()),
            lang_cache: RwLock::new(SemanticResponseCache::new()),
            executor,
            http_client: reqwest::Client::new(),
        })
    }
}

#[derive(Debug, Deserialize)]
struct BuildJoulefileReq {
    tag: String,
    joulefile: String,
    /// Map of `relative_path -> utf8_or_base64_content`
    #[serde(default)]
    context_files: BTreeMap<String, String>,
    #[serde(default)]
    base64_encoded: bool,
}

#[derive(Debug, Deserialize)]
struct IngestLayersReq {
    image_ref: String,
    #[serde(default)]
    config: OciImageConfig,
    /// Base64-encoded `.tar` or `.tar.gz` layers (base -> top)
    layers_base64: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct PullOciReq {
    image_ref: String,
}

#[derive(Debug, Deserialize)]
struct ForkCapsuleReq {
    new_capsule_id: String,
    #[serde(default)]
    energy_budget_uj: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct ExecCapsuleReq {
    command: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct HybridSearchReq {
    query_text: String,
    #[serde(default)]
    query_embedding: Option<Vec<f32>>,
    #[serde(default)]
    query_role_fillers: Vec<(String, String)>,
    #[serde(default)]
    required_attributes: BTreeMap<String, String>,
    #[serde(default)]
    entry_seed_id: Option<u64>,
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct GraphRagReq {
    query_text: String,
    #[serde(default)]
    query_embedding: Option<Vec<f32>>,
    #[serde(default)]
    query_role_fillers: Vec<(String, String)>,
    #[serde(default)]
    seed_k: Option<usize>,
    #[serde(default)]
    top_k: Option<usize>,
}

#[derive(Debug, Deserialize)]
struct RegisterSemanticModelReq {
    #[serde(default)]
    entities: Vec<SemanticEntity>,
    #[serde(default)]
    joins: Vec<VerifiedJoinEdge>,
    #[serde(default)]
    metrics: Vec<CertifiedMetric>,
}

#[derive(Debug, Deserialize)]
struct ExecuteSemanticMetricReq {
    request: MetricQueryRequest,
    #[serde(default)]
    execute_sql: bool,
}

#[derive(Debug, Deserialize)]
struct LangCachePutReq {
    tenant_id: String,
    branch_id: String,
    current_lsn: u64,
    prompt: String,
    response_payload: serde_json::Value,
    original_cost_uj: u64,
}

#[derive(Debug, Deserialize)]
struct LangCacheGetReq {
    tenant_id: String,
    branch_id: String,
    current_lsn: u64,
    prompt: String,
    #[serde(default)]
    min_cosine: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct SpaceEditReq {
    branch: String,
    actor: String,
    actor_kind: ActorKind,
    path: String,
    content: String,
    summary: String,
    #[serde(default)]
    prompt_intent: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SpaceUndoReq {
    branch: String,
    actor: String,
}

#[derive(Debug, Deserialize)]
struct SpaceMergeReq {
    branch: String,
    path: String,
    base: String,
    side_a_label: String,
    side_a: String,
    side_b_label: String,
    side_b: String,
}

#[derive(Debug, Deserialize)]
struct SpaceResolveConflictReq {
    actor: String,
    resolved_content: String,
}

#[derive(Debug, Deserialize)]
struct SpaceThreadReq {
    thread_id: String,
    branch: String,
    title: String,
    #[serde(default)]
    anchor_path: Option<String>,
    #[serde(default)]
    anchor_op_id: Option<String>,
    #[serde(default)]
    status: Option<ThreadStatus>,
    message: ThreadMessage,
}

#[derive(Debug, Deserialize)]
struct SandboxPolicyReq {
    branch: String,
    policy: NetworkEgressPolicy,
}

#[derive(Debug, Deserialize)]
struct SandboxEgressCheckReq {
    branch: String,
    host: String,
}

#[derive(Debug, Deserialize)]
struct RoomJoinReq {
    room_id: String,
    linked_branch: String,
    participant: RoomParticipant,
}

#[derive(Debug, Deserialize)]
struct RoomTurnReq {
    room_id: String,
    turn_id: String,
    agent_id: String,
    budgeted_uj: u64,
}

#[derive(Debug, Deserialize)]
struct RoomBargeInReq {
    room_id: String,
    interrupter_id: String,
    reason: String,
}

#[derive(Debug, Deserialize)]
struct RoomCrdtReq {
    room_id: String,
    updates: BTreeMap<String, LwwRoomEntry>,
}

/// Mount all `/api/v1/capsules/*`, `/api/v1/space/*`, and 360° SWOT parity routes.
pub fn swot_capsule_router(rt: Arc<SwotCapsuleRuntime>) -> Router {
    Router::new()
        // ── Post-Docker `JouleCapsule` & OCI Routes ──────────────────────────
        .route(
            "/api/v1/capsules/images",
            get(|State(rt): State<Arc<SwotCapsuleRuntime>>| async move {
                (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "ok": true,
                        "images": rt.capsules.list_images(),
                        "cas_dedup": rt.capsules.cas_stats(),
                    })),
                )
            }),
        )
        .route(
            "/api/v1/capsules/images/build",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<BuildJoulefileReq>| async move {
                    let mut ctx = BTreeMap::new();
                    for (k, v) in req.context_files {
                        let bytes = if req.base64_encoded {
                            base64::engine::general_purpose::STANDARD
                                .decode(v.as_bytes())
                                .unwrap_or_default()
                        } else {
                            v.into_bytes()
                        };
                        ctx.insert(k, bytes);
                    }
                    match rt.capsules.build_from_joulefile(&req.tag, &req.joulefile, &ctx) {
                        Ok(img) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "image": img,
                                "cas_dedup": rt.capsules.cas_stats(),
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/capsules/images/ingest",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<IngestLayersReq>| async move {
                    let mut layers = Vec::with_capacity(req.layers_base64.len());
                    for b64 in &req.layers_base64 {
                        match base64::engine::general_purpose::STANDARD.decode(b64.as_bytes()) {
                            Ok(bytes) => layers.push(bytes),
                            Err(e) => {
                                return (
                                    StatusCode::BAD_REQUEST,
                                    Json(serde_json::json!({ "ok": false, "error": format!("invalid base64 layer: {e}") })),
                                )
                                    .into_response();
                            }
                        }
                    }
                    match rt.capsules.ingest_oci_image(&req.image_ref, req.config, &layers) {
                        Ok(img) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "image": img,
                                "cas_dedup": rt.capsules.cas_stats(),
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/capsules/images/pull",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<PullOciReq>| async move {
                    match pull_oci_v2_native(&rt, &req.image_ref).await {
                        Ok(img) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "image": img,
                                "cas_dedup": rt.capsules.cas_stats(),
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_GATEWAY,
                            Json(serde_json::json!({ "ok": false, "error": e })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/capsules",
            get(|State(rt): State<Arc<SwotCapsuleRuntime>>| async move {
                let list = rt.capsules.list_capsules();
                (
                    StatusCode::OK,
                    Json(serde_json::json!({
                        "ok": true,
                        "count": list.len(),
                        "capsules": list,
                    })),
                )
            }),
        )
        .route(
            "/api/v1/capsules/spawn",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<SpawnCapsuleRequest>| async move {
                    match rt.capsules.spawn_capsule(req) {
                        Ok(cap) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "capsule": cap })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/capsules/{id}/fork",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Path(id): Path<String>,
                 Json(req): Json<ForkCapsuleReq>| async move {
                    match rt
                        .capsules
                        .fork_capsule(&id, &req.new_capsule_id, req.energy_budget_uj)
                    {
                        Ok(cap) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "capsule": cap })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/capsules/{id}/exec",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Path(id): Path<String>,
                 Json(req): Json<ExecCapsuleReq>| async move {
                    match rt
                        .capsules
                        .exec_in_capsule(&id, &req.command, &rt.scratch_dir)
                    {
                        Ok(log) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "exec": log })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/capsules/{id}/diff/{other}",
            get(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Path((id, other)): Path<(String, String)>| async move {
                    match rt.capsules.diff_capsules(&id, &other) {
                        Ok(entries) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "base_capsule": id,
                                "target_capsule": other,
                                "entries": entries,
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/capsules/compose/up",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(spec): Json<ComposeSpec>| async move {
                    match rt.capsules.compose_up(&spec) {
                        Ok(plan) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "compose": plan })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        // ── 360° Global SWOT Parity Routes ───────────────────────────────────
        .route(
            "/api/v1/hybrid/upsert",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(doc): Json<HybridDocument>| async move {
                    let id = doc.id;
                    if let Ok(mut idx) = rt.hybrid_index.write() {
                        idx.upsert(doc);
                    }
                    (StatusCode::OK, Json(serde_json::json!({ "ok": true, "id": id })))
                },
            ),
        )
        .route(
            "/api/v1/hybrid/search",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<HybridSearchReq>| async move {
                    let hits = rt
                        .hybrid_index
                        .read()
                        .map(|idx| {
                            idx.search(
                                &req.query_text,
                                req.query_embedding.as_deref(),
                                &req.query_role_fillers,
                                &req.required_attributes,
                                req.entry_seed_id,
                                req.top_k.unwrap_or(10),
                                RrfConfig::default(),
                            )
                        })
                        .unwrap_or_default();
                    (StatusCode::OK, Json(serde_json::json!({ "ok": true, "hits": hits })))
                },
            ),
        )
        .route(
            "/api/v1/graphrag/retrieve",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<GraphRagReq>| async move {
                    let subgraph = rt
                        .hybrid_index
                        .read()
                        .map(|idx| {
                            HippoRagEngine::retrieve_subgraph(
                                &idx,
                                &req.query_text,
                                req.query_embedding.as_deref(),
                                &req.query_role_fillers,
                                req.seed_k.unwrap_or(3),
                                req.top_k.unwrap_or(10),
                            )
                        })
                        .unwrap_or_default();
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({ "ok": true, "subgraph": subgraph })),
                    )
                },
            ),
        )
        .route(
            "/api/v1/semantic-layer/register",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Json(req): Json<RegisterSemanticModelReq>| async move {
                    if let Ok(mut cat) = rt.semantic_catalog.write() {
                        for e in req.entities {
                            cat.register_entity(e);
                        }
                        for j in req.joins {
                            cat.register_join(j);
                        }
                        for m in req.metrics {
                            cat.register_metric(m);
                        }
                    }
                    (StatusCode::OK, Json(serde_json::json!({ "ok": true })))
                },
            ),
        )
        .route(
            "/api/v1/semantic-layer/query",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Json(req): Json<ExecuteSemanticMetricReq>| async move {
                    let compiled = rt
                        .semantic_catalog
                        .read()
                        .map_err(|_| "semantic catalog lock poisoned".to_string())
                        .and_then(|cat| cat.compile_to_sql(&req.request));
                    let sql = match compiled {
                        Ok(s) => s,
                        Err(e) => {
                            return (
                                StatusCode::BAD_REQUEST,
                                Json(serde_json::json!({ "ok": false, "error": e })),
                            )
                                .into_response();
                        }
                    };
                    if !req.execute_sql {
                        return (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "compiled_sql": sql })),
                        )
                            .into_response();
                    }
                    let qreq = QueryRequest {
                        sql: sql.clone(),
                        params: Default::default(),
                        args: Vec::new(),
                        explain: false,
                        limit: None,
                        session_id: None,
                        query_timeout_ms: None,
                        branch_id: None,
                        // Row-level tenant isolation is already compiled into `sql` via `tenant_column`
                        tenant_id: None,
                    };
                    match rt.executor.execute(&qreq) {
                        Ok(qres) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "compiled_sql": sql,
                                "columns": qres.columns,
                                "rows": qres.rows,
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::UNPROCESSABLE_ENTITY,
                            Json(serde_json::json!({
                                "ok": false,
                                "compiled_sql": sql,
                                "error": e.message,
                            })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/langcache/put",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<LangCachePutReq>| async move {
                    if let Ok(mut c) = rt.lang_cache.write() {
                        c.put(
                            &req.tenant_id,
                            &req.branch_id,
                            req.current_lsn,
                            &req.prompt,
                            req.response_payload,
                            req.original_cost_uj,
                        );
                    }
                    let stats = rt.lang_cache.read().map(|c| c.stats()).unwrap_or_default();
                    (StatusCode::OK, Json(serde_json::json!({ "ok": true, "stats": stats })))
                },
            ),
        )
        .route(
            "/api/v1/langcache/get",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<LangCacheGetReq>| async move {
                    let hit = rt.lang_cache.write().ok().and_then(|mut c| {
                        c.get(
                            &req.tenant_id,
                            &req.branch_id,
                            req.current_lsn,
                            &req.prompt,
                            req.min_cosine.unwrap_or(0.82),
                        )
                    });
                    let stats = rt.lang_cache.read().map(|c| c.stats()).unwrap_or_default();
                    match hit {
                        Some((payload, cosine, saved_uj)) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "hit": true,
                                "cosine": cosine,
                                "joules_saved_uj": saved_uj,
                                "response": payload,
                                "stats": stats,
                            })),
                        ),
                        None => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "hit": false,
                                "stats": stats,
                            })),
                        ),
                    }
                },
            ),
        )
        // ── Post-Silo `JouleSpace` Routes (Zed Delta + Jujutsu + Docker Sandbox + LiveKit + Mesh) ──
        .route(
            "/api/v1/space/edit",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<SpaceEditReq>| async move {
                    match rt.space.apply_continuous_edit(
                        &req.branch,
                        &req.actor,
                        req.actor_kind,
                        &req.path,
                        &req.content,
                        &req.summary,
                        req.prompt_intent,
                    ) {
                        Ok((op, scrubbed)) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "op": op,
                                "secrets_scrubbed": scrubbed,
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/undo",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<SpaceUndoReq>| async move {
                    match rt.space.undo_last_op(&req.branch, &req.actor) {
                        Ok(op) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "undo_op": op })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/oplog/{branch}",
            get(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Path(branch): Path<String>| async move {
                    let ops = rt.space.list_ops(Some(&branch));
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({ "ok": true, "branch": branch, "ops": ops })),
                    )
                },
            ),
        )
        .route(
            "/api/v1/space/merge",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<SpaceMergeReq>| async move {
                    match rt.space.merge_concurrent_file(
                        &req.branch,
                        &req.path,
                        &req.base,
                        &req.side_a_label,
                        &req.side_a,
                        &req.side_b_label,
                        &req.side_b,
                    ) {
                        Ok(Ok(merged)) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "clean_merge": true,
                                "merged_content": merged,
                            })),
                        )
                            .into_response(),
                        Ok(Err(conflict)) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "clean_merge": false,
                                "first_class_conflict": conflict,
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/conflicts/{id}/resolve",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Path(id): Path<String>,
                 Json(req): Json<SpaceResolveConflictReq>| async move {
                    match rt.space.resolve_conflict(&id, &req.actor, &req.resolved_content) {
                        Ok(c) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "conflict": c })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/threads",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<SpaceThreadReq>| async move {
                    match rt.space.upsert_delta_thread(
                        &req.thread_id,
                        &req.branch,
                        &req.title,
                        req.anchor_path,
                        req.anchor_op_id,
                        req.status,
                        req.message,
                    ) {
                        Ok(th) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "thread": th })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/threads/{branch}",
            get(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Path(branch): Path<String>| async move {
                    let threads = rt.space.list_threads(Some(&branch));
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({ "ok": true, "threads": threads })),
                    )
                },
            ),
        )
        .route(
            "/api/v1/space/git-export/{branch}",
            get(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Path(branch): Path<String>| async move {
                    match rt.space.export_git_fast_import(&branch, "refs/heads/main") {
                        Ok(stream) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "fast_import": stream })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/sandbox/policy",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<SandboxPolicyReq>| async move {
                    rt.space.set_sandbox_policy(&req.branch, req.policy);
                    (StatusCode::OK, Json(serde_json::json!({ "ok": true })))
                },
            ),
        )
        .route(
            "/api/v1/space/sandbox/egress-check",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Json(req): Json<SandboxEgressCheckReq>| async move {
                    let allowed = rt.space.check_sandbox_egress(&req.branch, &req.host);
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({ "ok": true, "allowed": allowed })),
                    )
                },
            ),
        )
        .route(
            "/api/v1/space/sandbox/handoff/{branch}",
            get(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Path(branch): Path<String>| async move {
                    match rt.space.export_cloud_handoff(&branch) {
                        Ok(bundle) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "handoff": bundle })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/sandbox/handoff/import",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Json(bundle): Json<CloudSandboxHandoffBundle>| async move {
                    match rt.space.import_cloud_handoff(bundle) {
                        Ok(restored_files) => (
                            StatusCode::OK,
                            Json(serde_json::json!({
                                "ok": true,
                                "restored_files": restored_files,
                            })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/rooms/join",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<RoomJoinReq>| async move {
                    match rt
                        .space
                        .join_or_update_room(&req.room_id, &req.linked_branch, req.participant)
                    {
                        Ok(room) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "room": room })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/rooms/turn",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<RoomTurnReq>| async move {
                    match rt.space.start_agent_turn(
                        &req.room_id,
                        &req.turn_id,
                        &req.agent_id,
                        req.budgeted_uj,
                    ) {
                        Ok(turn) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "turn": turn })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/rooms/barge-in",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Json(req): Json<RoomBargeInReq>| async move {
                    match rt
                        .space
                        .trigger_barge_in(&req.room_id, &req.interrupter_id, &req.reason)
                    {
                        Ok(receipt) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "barge_in": receipt })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/rooms/crdt",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(req): Json<RoomCrdtReq>| async move {
                    match rt.space.merge_room_crdt_state(&req.room_id, req.updates) {
                        Ok(room) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "room": room })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .route(
            "/api/v1/space/mesh/peers",
            post(
                |State(rt): State<Arc<SwotCapsuleRuntime>>, Json(peer): Json<MeshPeer>| async move {
                    rt.space.register_mesh_peer(peer);
                    let peers = rt.space.list_mesh_peers();
                    (
                        StatusCode::OK,
                        Json(serde_json::json!({ "ok": true, "peers": peers })),
                    )
                },
            )
            .get(|State(rt): State<Arc<SwotCapsuleRuntime>>| async move {
                let peers = rt.space.list_mesh_peers();
                (
                    StatusCode::OK,
                    Json(serde_json::json!({ "ok": true, "peers": peers })),
                )
            }),
        )
        .route(
            "/api/v1/space/sync/{branch}/{since_seq}",
            get(
                |State(rt): State<Arc<SwotCapsuleRuntime>>,
                 Path((branch, since_seq)): Path<(String, u64)>| async move {
                    match rt.space.pull_delta_since(&branch, since_seq) {
                        Ok(batch) => (
                            StatusCode::OK,
                            Json(serde_json::json!({ "ok": true, "delta": batch })),
                        )
                            .into_response(),
                        Err(e) => (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e.to_string() })),
                        )
                            .into_response(),
                    }
                },
            ),
        )
        .with_state(rt)
}

/// Pure-Rust OCI Distribution v2 registry pull (`WWW-Authenticate: Bearer` + manifest + `.tar.gz` layers).
async fn pull_oci_v2_native(
    rt: &SwotCapsuleRuntime,
    raw_ref: &str,
) -> Result<joule_db_branch::capsule::OciImageRecord, String> {
    let parsed = OciImageRef::parse(raw_ref).map_err(|e| e.to_string())?;
    let scheme = if parsed.registry.starts_with("localhost") || parsed.registry.starts_with("127.")
    {
        "http"
    } else {
        "https"
    };
    let ref_or_digest = parsed.digest.as_deref().unwrap_or(&parsed.tag);
    let manifest_url = format!(
        "{scheme}://{}/v2/{}/manifests/{ref_or_digest}",
        parsed.registry, parsed.repository
    );

    let accept_hdr = "application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.list.v2+json, application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json";

    let first = rt
        .http_client
        .get(&manifest_url)
        .header("Accept", accept_hdr)
        .send()
        .await
        .map_err(|e| format!("registry connect error: {e}"))?;

    let token = if first.status() == StatusCode::UNAUTHORIZED {
        if let Some(www) = first
            .headers()
            .get("www-authenticate")
            .and_then(|v| v.to_str().ok())
        {
            fetch_bearer_token(&rt.http_client, www).await?
        } else {
            return Err("registry returned 401 without WWW-Authenticate".into());
        }
    } else {
        None
    };

    let fetch_with_auth = |url: String, accept: &'static str| {
        let client = rt.http_client.clone();
        let tok = token.clone();
        async move {
            let mut req = client.get(&url).header("Accept", accept);
            if let Some(t) = tok {
                req = req.bearer_auth(t);
            }
            req.send()
                .await
                .map_err(|e| format!("GET {url} failed: {e}"))
        }
    };

    let resp = fetch_with_auth(manifest_url, accept_hdr).await?;
    if !resp.status().is_success() {
        return Err(format!("manifest fetch returned HTTP {}", resp.status()));
    }
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let body_bytes = resp
        .bytes()
        .await
        .map_err(|e| format!("manifest body error: {e}"))?;

    // If multi-arch index, select host arch (`arm64` on aarch64, else `amd64`)
    let manifest: OciManifestV2 = if ctype.contains("index") || ctype.contains("manifest.list") {
        let index: OciImageIndex = serde_json::from_slice(&body_bytes)
            .map_err(|e| format!("invalid OCI image index JSON: {e}"))?;
        let pref_arch = if cfg!(target_arch = "aarch64") {
            "arm64"
        } else {
            "amd64"
        };
        let sub_digest = index
            .select_digest(pref_arch)
            .ok_or_else(|| "empty OCI image index".to_string())?;
        let sub_url = format!(
            "{scheme}://{}/v2/{}/manifests/{sub_digest}",
            parsed.registry, parsed.repository
        );
        let sub_resp = fetch_with_auth(
            sub_url,
            "application/vnd.oci.image.manifest.v1+json, application/vnd.docker.distribution.manifest.v2+json",
        )
        .await?;
        let sub_bytes = sub_resp
            .bytes()
            .await
            .map_err(|e| format!("platform manifest read error: {e}"))?;
        serde_json::from_slice(&sub_bytes)
            .map_err(|e| format!("invalid platform manifest JSON: {e}"))?
    } else {
        serde_json::from_slice(&body_bytes)
            .map_err(|e| format!("invalid OCI manifest JSON: {e}"))?
    };

    let mut layers = Vec::with_capacity(manifest.layers.len());
    for layer_desc in &manifest.layers {
        let blob_url = format!(
            "{scheme}://{}/v2/{}/blobs/{}",
            parsed.registry, parsed.repository, layer_desc.digest
        );
        let lresp = fetch_with_auth(blob_url, "*/*").await?;
        if !lresp.status().is_success() {
            return Err(format!("layer blob fetch returned HTTP {}", lresp.status()));
        }
        let lbytes = lresp
            .bytes()
            .await
            .map_err(|e| format!("layer stream error: {e}"))?;
        layers.push(lbytes.to_vec());
    }

    rt.capsules
        .ingest_oci_image(raw_ref, OciImageConfig::default(), &layers)
        .map_err(|e| e.to_string())
}

async fn fetch_bearer_token(
    client: &reqwest::Client,
    www_auth: &str,
) -> Result<Option<String>, String> {
    let Some(rest) = www_auth.strip_prefix("Bearer ") else {
        return Ok(None);
    };
    let mut realm = None;
    let mut service = None;
    let mut scope = None;
    for part in rest.split(',') {
        if let Some((k, v)) = part.trim().split_once('=') {
            let val = v.trim_matches('"').to_string();
            match k.trim() {
                "realm" => realm = Some(val),
                "service" => service = Some(val),
                "scope" => scope = Some(val),
                _ => {}
            }
        }
    }
    let Some(realm_url) = realm else {
        return Ok(None);
    };
    let mut query = Vec::new();
    if let Some(s) = service {
        query.push(("service", s));
    }
    if let Some(sc) = scope {
        query.push(("scope", sc));
    }
    let resp = client
        .get(&realm_url)
        .query(&query)
        .send()
        .await
        .map_err(|e| format!("OCI token exchange failed: {e}"))?;
    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("invalid OCI token JSON: {e}"))?;
    Ok(json
        .get("token")
        .or_else(|| json.get("access_token"))
        .and_then(|t| t.as_str())
        .map(String::from))
}
