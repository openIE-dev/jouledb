//! LangGraph HTTP Handlers
//!
//! REST API for JouleDB's LangGraph-compatible checkpoint and message stores.
//! Exposes checkpoint persistence and semantic message search via HTTP.

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use joule_db_langgraph::checkpoint::JouleCheckpointStore;
use joule_db_langgraph::messages::{JouleMessageStore, Message, MessageRole};
use std::sync::Arc;

use crate::lock_util::{read_lock, write_lock};

/// Shared state for LangGraph routes.
#[derive(Clone)]
pub struct LangGraphState {
    pub checkpoint_store: Arc<std::sync::RwLock<JouleCheckpointStore>>,
    pub message_store: Arc<std::sync::RwLock<JouleMessageStore>>,
}

impl LangGraphState {
    pub fn new() -> Self {
        Self {
            checkpoint_store: Arc::new(std::sync::RwLock::new(JouleCheckpointStore::new())),
            message_store: Arc::new(std::sync::RwLock::new(JouleMessageStore::new())),
        }
    }
}

// ── Checkpoint Handlers ────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct PutCheckpointRequest {
    pub checkpoint_id: String,
    pub state: serde_json::Value,
    pub step: Option<u64>,
    pub tags: Option<Vec<String>>,
}

/// PUT/POST a checkpoint for a thread.
pub async fn put_checkpoint(
    State(state): State<LangGraphState>,
    Path(thread_id): Path<String>,
    Json(body): Json<PutCheckpointRequest>,
) -> impl IntoResponse {
    let mut store = write_lock(&state.checkpoint_store);
    let metadata = {
        let mut m = joule_db_langgraph::checkpoint::CheckpointMetadata::new(
            &thread_id,
            &body.checkpoint_id,
        );
        if let Some(step) = body.step {
            m = m.with_step(step);
        }
        if let Some(tags) = body.tags {
            m = m.with_tags(tags);
        }
        m
    };
    match store.put_checkpoint_with_metadata(
        &thread_id,
        &body.checkpoint_id,
        &body.state,
        Some(metadata),
    ) {
        Ok(()) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "checkpoint_id": body.checkpoint_id,
                "thread_id": thread_id,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// GET the latest checkpoint for a thread.
pub async fn get_latest_checkpoint(
    State(state): State<LangGraphState>,
    Path(thread_id): Path<String>,
) -> impl IntoResponse {
    let store = read_lock(&state.checkpoint_store);
    match store.get_checkpoint(&thread_id) {
        Some(cp) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "checkpoint_id": cp.metadata.checkpoint_id,
                "thread_id": cp.metadata.thread_id,
                "state": cp.state,
                "step": cp.metadata.step,
                "tags": cp.metadata.tags,
                "created_at": cp.metadata.created_at.to_rfc3339(),
            })),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(
                serde_json::json!({ "error": format!("No checkpoint for thread '{}'", thread_id) }),
            ),
        )
            .into_response(),
    }
}

/// GET a specific checkpoint by ID.
pub async fn get_checkpoint_by_id(
    State(state): State<LangGraphState>,
    Path((_thread_id, cp_id)): Path<(String, String)>,
) -> impl IntoResponse {
    let store = read_lock(&state.checkpoint_store);
    match store.get_checkpoint_by_id(&cp_id) {
        Some(cp) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "checkpoint_id": cp.metadata.checkpoint_id,
                "thread_id": cp.metadata.thread_id,
                "state": cp.state,
                "step": cp.metadata.step,
                "tags": cp.metadata.tags,
                "created_at": cp.metadata.created_at.to_rfc3339(),
            })),
        )
            .into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "error": format!("Checkpoint '{}' not found", cp_id) })),
        )
            .into_response(),
    }
}

// ── Message Handlers ───────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct AddMessageRequest {
    pub role: String,
    pub content: String,
    pub name: Option<String>,
    pub tool_call_id: Option<String>,
}

/// POST a message to a thread.
pub async fn add_message(
    State(state): State<LangGraphState>,
    Path(thread_id): Path<String>,
    Json(body): Json<AddMessageRequest>,
) -> impl IntoResponse {
    let role = match body.role.as_str() {
        "user" => MessageRole::User,
        "assistant" => MessageRole::Assistant,
        "system" => MessageRole::System,
        "tool" => MessageRole::Tool,
        _ => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": format!("Invalid role: {}", body.role) })),
            )
                .into_response();
        }
    };

    let mut msg = Message::new(&thread_id, role, &body.content);
    if let Some(name) = body.name {
        msg = msg.with_name(name);
    }
    if let Some(tool_call_id) = body.tool_call_id {
        msg = msg.with_metadata("tool_call_id", tool_call_id);
    }

    let mut store = write_lock(&state.message_store);
    match store.add_message(msg) {
        Ok(id) => (
            StatusCode::CREATED,
            Json(serde_json::json!({
                "message_id": id,
                "thread_id": thread_id,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

/// GET all messages for a thread.
pub async fn get_messages(
    State(state): State<LangGraphState>,
    Path(thread_id): Path<String>,
) -> impl IntoResponse {
    let store = read_lock(&state.message_store);
    let messages = store.get_messages(&thread_id);
    let json_messages: Vec<serde_json::Value> = messages
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "thread_id": m.thread_id,
                "role": m.role.as_str(),
                "content": m.content,
                "created_at": m.created_at.to_rfc3339(),
                "name": m.name,
                "metadata": m.metadata,
            })
        })
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({ "messages": json_messages })),
    )
        .into_response()
}

// ── Search Handler ─────────────────────────────────────────────────────

#[derive(serde::Deserialize)]
pub struct SearchRequest {
    pub query: String,
    pub k: Option<usize>,
    pub thread_id: Option<String>,
}

/// POST search for semantically similar messages.
pub async fn search_similar(
    State(state): State<LangGraphState>,
    Json(body): Json<SearchRequest>,
) -> impl IntoResponse {
    let store = read_lock(&state.message_store);
    let k = body.k.unwrap_or(5);
    let results = if let Some(tid) = &body.thread_id {
        store.search_similar_in_thread(tid, &body.query, k)
    } else {
        store.search_similar(&body.query, k)
    };

    let json_results: Vec<serde_json::Value> = results
        .iter()
        .map(|m| {
            serde_json::json!({
                "id": m.id,
                "thread_id": m.thread_id,
                "role": m.role.as_str(),
                "content": m.content,
                "created_at": m.created_at.to_rfc3339(),
            })
        })
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({ "results": json_results })),
    )
        .into_response()
}

/// Build the LangGraph route group.
pub fn langgraph_routes(state: LangGraphState) -> axum::Router {
    use axum::routing::{get, post};

    axum::Router::new()
        .route(
            "/api/v1/langgraph/checkpoint/{thread_id}",
            post(put_checkpoint).get(get_latest_checkpoint),
        )
        .route(
            "/api/v1/langgraph/checkpoint/{thread_id}/{cp_id}",
            get(get_checkpoint_by_id),
        )
        .route("/api/v1/langgraph/message/{thread_id}", post(add_message))
        .route("/api/v1/langgraph/messages/{thread_id}", get(get_messages))
        .route("/api/v1/langgraph/search", post(search_similar))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn test_app() -> axum::Router {
        langgraph_routes(LangGraphState::new())
    }

    #[tokio::test]
    async fn test_put_and_get_checkpoint() {
        let app = test_app();

        // PUT a checkpoint
        let body = serde_json::json!({
            "checkpoint_id": "cp1",
            "state": {"counter": 42},
            "step": 1
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/langgraph/checkpoint/thread1")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // GET latest checkpoint
        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/langgraph/checkpoint/thread1")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["checkpoint_id"], "cp1");
        assert_eq!(json["state"]["counter"], 42);
    }

    #[tokio::test]
    async fn test_get_checkpoint_not_found() {
        let app = test_app();
        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/langgraph/checkpoint/nonexistent")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn test_get_checkpoint_by_id() {
        let app = test_app();

        // Create checkpoint
        let body = serde_json::json!({
            "checkpoint_id": "specific-cp",
            "state": {"key": "value"}
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/langgraph/checkpoint/t1")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        app.clone().oneshot(req).await.unwrap();

        // Get by ID
        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/langgraph/checkpoint/t1/specific-cp")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn test_add_and_get_messages() {
        let app = test_app();

        // Add a message
        let body = serde_json::json!({
            "role": "user",
            "content": "Hello, world!"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/langgraph/message/thread1")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::CREATED);

        // Get messages
        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/langgraph/messages/thread1")
            .body(Body::empty())
            .unwrap();
        let resp = app.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["messages"].as_array().unwrap().len(), 1);
        assert_eq!(json["messages"][0]["content"], "Hello, world!");
    }

    #[tokio::test]
    async fn test_add_message_invalid_role() {
        let app = test_app();
        let body = serde_json::json!({
            "role": "invalid_role",
            "content": "test"
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/langgraph/message/thread1")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_search_empty() {
        let app = test_app();
        let body = serde_json::json!({
            "query": "hello",
            "k": 5
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/langgraph/search")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&body).unwrap()))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["results"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_get_empty_messages() {
        let app = test_app();
        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/langgraph/messages/empty_thread")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert!(json["messages"].as_array().unwrap().is_empty());
    }

    #[tokio::test]
    async fn test_multiple_messages_ordered() {
        let app = test_app();

        for content in &["First", "Second", "Third"] {
            let body = serde_json::json!({
                "role": "user",
                "content": content
            });
            let req = Request::builder()
                .method("POST")
                .uri("/api/v1/langgraph/message/thread1")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap();
            app.clone().oneshot(req).await.unwrap();
        }

        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/langgraph/messages/thread1")
            .body(Body::empty())
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let msgs = json["messages"].as_array().unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0]["content"], "First");
        assert_eq!(msgs[2]["content"], "Third");
    }
}
