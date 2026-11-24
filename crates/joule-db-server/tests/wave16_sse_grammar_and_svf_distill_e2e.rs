//! Wave 16 End-to-End Integration Suite:
//! - Single-Pass Teacher->Student Coordinate-Ridge `SVF` + Per-Row Gain Self-Distillation (`distill_svf: true`)
//! - Native SSE Token Streaming (`stream: true`, `text/event-stream`, per-token `x_jouledb_token_nj`) on
//!   OpenAI `POST /v1/chat/completions` AND Anthropic `POST /v1/messages`
//! - Zero-Hallucination Schema-Constrained JSON (`response_format`) & Tool-Call Grammar (`tools` -> `tool_calls` / `tool_use`)

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::checkpoint_ingest::{
    SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use tower::ServiceExt;

fn boot_test_server(dir: &std::path::Path) -> Server {
    let cfg = ServerConfig {
        db_path: dir.to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    Server::new(cfg).expect("boot server")
}

async fn post_raw(server: &Server, uri: &str, body: JsonValue) -> (StatusCode, String, String) {
    let app = server.router();
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("post req");
    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let ctype = resp
        .headers()
        .get("content-type")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body");
    (status, ctype, String::from_utf8_lossy(&bytes).to_string())
}

async fn post_json(server: &Server, uri: &str, body: JsonValue) -> (StatusCode, JsonValue) {
    let (status, _, txt) = post_raw(server, uri, body).await;
    let parsed: JsonValue = serde_json::from_str(&txt).expect("parse json");
    (status, parsed)
}

#[tokio::test]
async fn test_svf_self_distillation_sse_token_streaming_and_grammar_tool_calls_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(tmp.path());

    // 1. Verify Self-Distilled `SVF` calibration (`distill_svf: true`) during `.safetensors` ingestion
    let entries = vec![
        SafetensorBuildEntry {
            name: "model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: (0..1024).map(|i| ((i as f32) * 0.13).sin() * 1.4).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: (0..1024).map(|i| ((i as f32) * 0.19).cos() * 1.2).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, 32],
            values_f32: (0..512).map(|i| ((i as f32) * 0.23).sin() * 0.9).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![16, 32],
            values_f32: (0..512).map(|i| ((i as f32) * 0.29).cos() * 0.9).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 32],
            values_f32: (0..1024).map(|i| ((i as f32) * 0.31).sin() * 0.8).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: (0..2048).map(|i| ((i as f32) * 0.17).cos() * 0.7).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![64, 32],
            values_f32: (0..2048).map(|i| ((i as f32) * 0.11).sin() * 0.7).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32, 64],
            values_f32: (0..2048).map(|i| ((i as f32) * 0.07).cos() * 0.7).collect(),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![1.0; 32],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![1.0; 32],
        },
        SafetensorBuildEntry {
            name: "model.norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![32],
            values_f32: vec![1.0; 32],
        },
    ];
    let st_bytes = build_safetensors_bytes(&BTreeMap::new(), &entries).expect("safetensors");
    let st_file = tmp.path().join("qat-model.safetensors");
    std::fs::write(&st_file, st_bytes).expect("write st");

    let (status, ing) = post_json(
        &server,
        "/api/v1/synapse/ingest/safetensors",
        json!({
            "model_id": "qat-causal-1l",
            "file_path": st_file.to_string_lossy(),
            "target_rank": 16,
            "max_outliers": 4,
            "distill_svf": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "ing: {ing}");
    assert_eq!(ing["report"]["svf_distilled"], true);
    let pre_mse = ing["report"]["pre_distill_probe_mse"]
        .as_f64()
        .unwrap_or(1.0);
    let post_mse = ing["report"]["post_distill_probe_mse"]
        .as_f64()
        .unwrap_or(2.0);
    assert!(
        post_mse <= pre_mse,
        "expected post_mse ({post_mse}) <= pre_mse ({pre_mse})"
    );

    // 2. OpenAI `POST /v1/chat/completions` with `response_format` (`json_schema`) -> 100% valid JSON!
    let schema = json!({
        "name": "fabric_placement",
        "schema": {
            "type": "object",
            "properties": {
                "tier": { "type": "string", "enum": ["mcu", "uma", "cloud"] },
                "budget_mj": { "type": "integer" },
                "sql": { "type": "string" }
            },
            "required": ["tier", "budget_mj", "sql"]
        }
    });
    let (status, structured) = post_json(
        &server,
        "/v1/chat/completions",
        json!({
            "model": "qat-causal-1l",
            "messages": [{
                "role": "user",
                "content": "Dispatch on uma with 750 mJ budget: SELECT id, joules FROM telemetry LIMIT 3"
            }],
            "response_format": {
                "type": "json_schema",
                "json_schema": schema
            }
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "structured: {structured}");
    assert_eq!(
        structured["x_jouledb_energy"]["cascade_tier"],
        "tier4_xgrammar_constrained_json"
    );
    let content_str = structured["choices"][0]["message"]["content"]
        .as_str()
        .expect("content string");
    let parsed_content: JsonValue = serde_json::from_str(content_str).expect("valid json content");
    assert_eq!(parsed_content["tier"], "uma");
    assert_eq!(parsed_content["budget_mj"], 750);
    assert_eq!(
        parsed_content["sql"],
        "SELECT id, joules FROM telemetry LIMIT 3"
    );

    // 3. OpenAI `POST /v1/chat/completions` with `tools` -> native `tool_calls`!
    let (status, tool_resp) = post_json(
        &server,
        "/v1/chat/completions",
        json!({
            "model": "qat-causal-1l",
            "messages": [{
                "role": "user",
                "content": "Run SQL query on cloud with 900 mJ: SELECT COUNT(*) FROM synapse_tensors"
            }],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "run_metered_sql",
                    "description": "Execute SQL query on a target fabric tier with mJ budget",
                    "parameters": schema["schema"]
                }
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "tool_resp: {tool_resp}");
    assert_eq!(tool_resp["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(
        tool_resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "run_metered_sql"
    );
    let tool_args: JsonValue = serde_json::from_str(
        tool_resp["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap_or("{}"),
    )
    .expect("valid tool args");
    assert_eq!(tool_args["tier"], "cloud");
    assert_eq!(tool_args["budget_mj"], 900);

    // 4. OpenAI `POST /v1/chat/completions` with `stream: true` -> `text/event-stream` + per-token `nJ` + `[DONE]`!
    let (status, ctype, sse_text) = post_raw(
        &server,
        "/v1/chat/completions",
        json!({
            "model": "jouledb-cascade-v1",
            "stream": true,
            "messages": [{ "role": "user", "content": "Stream thermodynamic chunks" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(ctype.contains("text/event-stream"));
    assert!(sse_text.contains("data: [DONE]"));
    let mut sum_tok_nj = 0u64;
    let mut assembled = String::new();
    for line in sse_text.lines() {
        if let Some(rest) = line.strip_prefix("data: ")
            && rest.trim() != "[DONE]"
            && let Ok(ev) = serde_json::from_str::<JsonValue>(rest)
        {
            if let Some(nj) = ev.get("x_jouledb_token_nj").and_then(|x| x.as_u64()) {
                sum_tok_nj += nj;
            }
            if let Some(s) = ev["choices"][0]["delta"]["content"].as_str() {
                assembled.push_str(s);
            }
        }
    }
    assert!(sum_tok_nj > 0);
    assert!(!assembled.is_empty());

    // 5. Anthropic `POST /v1/messages` with `stream: true` AND with `tools` (`tool_use`)!
    let (status, actype, anth_sse) = post_raw(
        &server,
        "/v1/messages",
        json!({
            "model": "jouledb-cascade-v1",
            "max_tokens": 64,
            "stream": true,
            "messages": [{ "role": "user", "content": "Hello Anthropic SSE" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert!(actype.contains("text/event-stream"));
    assert!(anth_sse.contains("event: message_start"));
    assert!(anth_sse.contains("event: content_block_delta"));
    assert!(anth_sse.contains("event: message_stop"));

    let (status, anth_tool) = post_json(
        &server,
        "/v1/messages",
        json!({
            "model": "qat-causal-1l",
            "max_tokens": 64,
            "messages": [{
                "role": "user",
                "content": "Execute SQL on mcu with 120 mJ: SELECT 1"
            }],
            "tools": [{
                "name": "run_metered_sql",
                "description": "Execute SQL query on a target fabric tier",
                "input_schema": schema["schema"]
            }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "anth_tool: {anth_tool}");
    assert_eq!(anth_tool["stop_reason"], "tool_use");
    assert_eq!(anth_tool["content"][0]["type"], "tool_use");
    assert_eq!(anth_tool["content"][0]["name"], "run_metered_sql");
    assert_eq!(anth_tool["content"][0]["input"]["tier"], "mcu");
    assert_eq!(anth_tool["content"][0]["input"]["budget_mj"], 120);
}
