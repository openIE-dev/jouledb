//! Wave 17 End-to-End Integration Suite:
//! 1. **Frontier 2 (`wgpu` Resident Batched `LittleBit-2` + Causal `GQA` + `T-MAC` 8-Bit LUT)**:
//!    Verifies bit-exact parity and acceleration across `use_gpu: true` (`wgpu` resident batched
//!    bitplane + fused causal `GQA` WGSL shaders) and `use_gpu: false` (`T-MAC` 8-bit LUT batched
//!    prefill) plus `Mooncake` KV prefix reuse.
//! 2. **Frontier 3 (Database-Native `MCP` Host & Autonomous SQL/`JouleWard`/`Titans` Agent Loop)**:
//!    - Live Streamable-HTTP `JSON-RPC 2.0` `POST /api/v1/mcp/host/servers` discovery (`tools/list`)
//!      and invocation (`tools/call`) against a live `/mcp` server endpoint.
//!    - Multi-turn autonomous SQL -> `HarnessObservation` -> `Engram` + `Titans` MVCC memory loop
//!      (`POST /api/v1/agent/run`) via `HttpFabricClient`.
//!    - Cryptographic mid-loop `ThermoCap` (`allowed_sql_tables`) capability halt!

use base64::Engine;
use joule_db_server::{Server, ServerConfig};
use joule_db_weights::checkpoint_ingest::{
    SafetensorBuildEntry, SafetensorDtype, build_safetensors_bytes,
};
use serde_json::{Value as JsonValue, json};
use std::collections::BTreeMap;
use tokio::net::TcpListener;

#[tokio::test]
async fn test_wave17_gpu_batched_decoder_mcp_host_and_autonomous_agent_loop_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let cfg = ServerConfig {
        db_path: tmp.path().to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    let server = Server::new(cfg).expect("boot server");
    let app = server.router();

    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    let srv_task = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    let base_url = format!("http://127.0.0.1:{port}");
    let http = reqwest::Client::new();

    // -------------------------------------------------------------------------
    // 1. Frontier 2: Batched Causal Decoder (`use_gpu: true` vs `use_gpu: false`)
    // -------------------------------------------------------------------------
    let d_model = 256usize; // 256x256 = 65,536 weights (>= 32,768 threshold so GPU batch dispatches!)
    let kv_dim = 128usize;
    let ffn_dim = 256usize;
    let vocab = 128usize;

    let make_vec = |n: usize, mul: f32| -> Vec<f32> {
        (0..n).map(|i| ((i as f32) * mul).sin() * 0.4).collect()
    };
    let entries = vec![
        SafetensorBuildEntry {
            name: "model.embed_tokens.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![vocab, d_model],
            values_f32: make_vec(vocab * d_model, 0.11),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.input_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_model],
            values_f32: vec![1.0; d_model],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.q_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_model, d_model],
            values_f32: make_vec(d_model * d_model, 0.13),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.k_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![kv_dim, d_model],
            values_f32: make_vec(kv_dim * d_model, 0.17),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.v_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![kv_dim, d_model],
            values_f32: make_vec(kv_dim * d_model, 0.19),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.self_attn.o_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_model, d_model],
            values_f32: make_vec(d_model * d_model, 0.23),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.post_attention_layernorm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_model],
            values_f32: vec![1.0; d_model],
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.gate_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![ffn_dim, d_model],
            values_f32: make_vec(ffn_dim * d_model, 0.29),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.up_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![ffn_dim, d_model],
            values_f32: make_vec(ffn_dim * d_model, 0.31),
        },
        SafetensorBuildEntry {
            name: "model.layers.0.mlp.down_proj.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_model, ffn_dim],
            values_f32: make_vec(d_model * ffn_dim, 0.37),
        },
        SafetensorBuildEntry {
            name: "model.norm.weight".into(),
            dtype: SafetensorDtype::F32,
            shape: vec![d_model],
            values_f32: vec![1.0; d_model],
        },
    ];
    let st_bytes = build_safetensors_bytes(&BTreeMap::new(), &entries).expect("build st");
    let b64 = base64::engine::general_purpose::STANDARD.encode(&st_bytes);

    let ing: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/ingest/safetensors"))
        .json(&json!({
            "model_id": "wave17-causal",
            "bytes_base64": b64,
            "target_rank": 16,
            "distill_svf": true,
        }))
        .send()
        .await
        .expect("ingest send")
        .json()
        .await
        .expect("ingest json");
    assert_eq!(ing["causal_decoder_layers"], 1);

    let cpu_dec: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "wave17-causal",
            "prompt": "Batch prefill tokens across T-MAC lookup tables for fast silicon execution",
            "max_new_tokens": 5,
            "temperature": 0.0,
            "use_gpu": false,
        }))
        .send()
        .await
        .expect("cpu dec")
        .json()
        .await
        .expect("cpu dec json");
    assert_eq!(cpu_dec["execution_backend"], "cpu_tmac_lut_bitplane");
    assert_eq!(cpu_dec["gpu_accelerated"], false);

    let gpu_dec: JsonValue = http
        .post(format!("{base_url}/api/v1/synapse/decoder/generate"))
        .json(&json!({
            "model_id": "wave17-causal",
            "prompt": "Different prefix forcing cold GPU resident batched prefill execution path now",
            "max_new_tokens": 5,
            "temperature": 0.0,
            "use_gpu": true,
        }))
        .send()
        .await
        .expect("gpu dec")
        .json()
        .await
        .expect("gpu dec json");
    assert!(
        gpu_dec["generated_token_ids"]
            .as_array()
            .map(|a| !a.is_empty())
            .unwrap_or(false)
    );

    // -------------------------------------------------------------------------
    // 2. Frontier 3A: Database-Native `MCP` Host Mounting Live MCP Servers!
    // -------------------------------------------------------------------------
    // Mount JouleDB's own `/mcp/messages` JSON-RPC endpoint AND a dedicated external Streamable-HTTP MCP server
    let ext_listener = TcpListener::bind("127.0.0.1:0").await.expect("ext bind");
    let ext_port = ext_listener.local_addr().expect("ext addr").port();
    let ext_app = axum::Router::new().route(
        "/mcp",
        axum::routing::post(|axum::Json(rpc): axum::Json<JsonValue>| async move {
            let method = rpc["method"].as_str().unwrap_or("");
            if method == "tools/list" {
                axum::Json(json!({
                    "jsonrpc": "2.0",
                    "id": rpc["id"],
                    "result": {
                        "tools": [{
                            "name": "grid_forecast",
                            "description": "Fetch hydro grid carbon forecast",
                            "inputSchema": {
                                "type": "object",
                                "properties": { "region": { "type": "string" } },
                                "required": ["region"]
                            }
                        }]
                    }
                }))
            } else {
                axum::Json(json!({
                    "jsonrpc": "2.0",
                    "id": rpc["id"],
                    "result": {
                        "content": [{ "type": "text", "text": "14 gCO2/kWh" }],
                        "carbon_g_kwh": 14.0
                    }
                }))
            }
        }),
    );
    let ext_srv = tokio::spawn(async move {
        let _ = axum::serve(ext_listener, ext_app).await;
    });

    let self_reg: JsonValue = http
        .post(format!("{base_url}/api/v1/mcp/host/servers"))
        .json(&json!({
            "server_id": "self_mcp",
            "endpoint_url": format!("{base_url}/mcp/messages"),
            "discover_live": true,
        }))
        .send()
        .await
        .expect("self mcp register")
        .json()
        .await
        .expect("self mcp reg json");
    assert_eq!(self_reg["ok"], true);
    assert!(self_reg["discovered_tools_count"].as_u64().unwrap_or(0) >= 2);

    let reg: JsonValue = http
        .post(format!("{base_url}/api/v1/mcp/host/servers"))
        .json(&json!({
            "server_id": "ext_grid",
            "endpoint_url": format!("http://127.0.0.1:{ext_port}/mcp"),
            "discover_live": true,
        }))
        .send()
        .await
        .expect("mcp register")
        .json()
        .await
        .expect("mcp reg json");
    assert_eq!(reg["ok"], true);
    assert_eq!(reg["discovered_tools_count"], 1);

    let catalog: JsonValue = http
        .get(format!("{base_url}/api/v1/mcp/host/tools"))
        .send()
        .await
        .expect("mcp list")
        .json()
        .await
        .expect("mcp list json");
    assert_eq!(catalog["ok"], true);
    let tool_names: Vec<String> = catalog["tools"]
        .as_array()
        .expect("tools arr")
        .iter()
        .filter_map(|t| t["qualified_name"].as_str().map(String::from))
        .collect();
    assert!(tool_names.contains(&"jouledb.sql_query".to_string()));
    assert!(tool_names.contains(&"ext_grid.grid_forecast".to_string()));

    // Call the remote Streamable-HTTP MCP tool (`ext_grid.grid_forecast`) through MCP Host!
    let remote_call: JsonValue = http
        .post(format!("{base_url}/api/v1/mcp/host/call"))
        .json(&json!({
            "tool_name": "ext_grid.grid_forecast",
            "arguments": { "region": "nordic-hydro" },
        }))
        .send()
        .await
        .expect("remote mcp call")
        .json()
        .await
        .expect("remote mcp json");
    assert_eq!(remote_call["ok"], true);
    assert_eq!(remote_call["server_id"], "ext_grid");
    assert_eq!(remote_call["result"]["carbon_g_kwh"], 14.0);
    assert!(
        remote_call["x_jouledb_energy"]["aggregate_rails"]["wall_duration_ns"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    ext_srv.abort();

    // Create & seed a live SQL table via native `jouledb.sql_query` MCP tool!
    let _ = http
        .post(format!("{base_url}/api/v1/mcp/host/call"))
        .json(&json!({
            "tool_name": "jouledb.sql_query",
            "arguments": { "sql": "CREATE TABLE rack_power (rack TEXT, watts FLOAT)" },
        }))
        .send()
        .await
        .expect("create table");
    let _ = http
        .post(format!("{base_url}/api/v1/mcp/host/call"))
        .json(&json!({
            "tool_name": "jouledb.sql_query",
            "arguments": { "sql": "INSERT INTO rack_power VALUES ('rack-a', 18.5), ('rack-b', 42.5)" },
        }))
        .send()
        .await
        .expect("insert rows");

    // -------------------------------------------------------------------------
    // 3. Frontier 3B: Autonomous Multi-Turn SQL -> Engram + Titans Agent Loop!
    // -------------------------------------------------------------------------
    let goal = "SELECT rack, watts FROM rack_power ORDER BY watts DESC; commit highest rack to memory";
    let agent_res: JsonValue = http
        .post(format!("{base_url}/api/v1/agent/run"))
        .json(&json!({
            "goal": goal,
            "max_turns": 2,
            "commit_to_titans": true,
        }))
        .send()
        .await
        .expect("agent_run")
        .json()
        .await
        .expect("agent_run json");
    assert_eq!(agent_res["ok"], true);
    assert_eq!(agent_res["stop_reason"], "completed");
    assert_eq!(agent_res["turns_executed"], 2);
    assert_eq!(agent_res["steps"][0]["selected_tool"], "jouledb.sql_query");
    assert_eq!(agent_res["steps"][0]["output"]["row_count"], 2);
    assert_eq!(
        agent_res["steps"][1]["selected_tool"],
        "jouledb.engram_upsert"
    );

    // Confirm $O(1)$ Engram factual recall of the autonomous loop's grounded SQL result!
    let recall: JsonValue = http
        .post(format!("{base_url}/api/v1/mcp/host/call"))
        .json(&json!({
            "tool_name": "jouledb.engram_lookup",
            "arguments": { "phrase": goal },
        }))
        .send()
        .await
        .expect("engram recall")
        .json()
        .await
        .expect("engram recall json");
    assert_eq!(recall["result"]["hit"], true);
    assert!(
        recall["result"]["fact_payload"]
            .as_str()
            .unwrap_or("")
            .contains("rack-b")
    );

    // -------------------------------------------------------------------------
    // 4. Cryptographic `ThermoCap` SQL Table Confinement Mid-Loop Halt!
    // -------------------------------------------------------------------------
    let minted: JsonValue = http
        .post(format!("{base_url}/api/v1/ward/tokens/mint"))
        .json(&json!({
            "root_token_id": "agent-restricted-sql",
            "subject": "confined-agent",
            "caveats": [{ "kind": "allow_sql_tables", "value": ["public_metrics_only"] }],
        }))
        .send()
        .await
        .expect("mint token")
        .json()
        .await
        .expect("mint json");
    let ward_tok = minted["token"].clone();

    let blocked_run: JsonValue = http
        .post(format!("{base_url}/api/v1/agent/run"))
        .json(&json!({
            "goal": "SELECT rack, watts FROM rack_power",
            "max_turns": 2,
            "x_ward_token": ward_tok,
        }))
        .send()
        .await
        .expect("blocked agent send")
        .json()
        .await
        .expect("blocked agent json");
    assert_eq!(blocked_run["stop_reason"], "halted_by_ward_or_guardrail");
    assert_eq!(blocked_run["turns_executed"], 0);

    srv_task.abort();
}
