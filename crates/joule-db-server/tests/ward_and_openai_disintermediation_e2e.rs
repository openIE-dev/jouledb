//! Wave 12 End-to-End Integration Suite:
//! `JouleWard` Capability Confinement + Cross-Fabric High-Resolution (`nJ` / `ns`) 6-Rail Energy
//! Accounting + Drop-In OpenAI (`/v1/*`) & Ollama (`/api/*`) AI Runtime Disintermediation.

use axum::{
    body::Body,
    http::{Method, Request, StatusCode},
};
use base64::Engine;
use joule_db_energy::{
    ThermalState,
    fabric::{FabricNodeClass, FabricNodeDescriptor, InterconnectKind, MemoryBusKind, SiliconLane},
};
use joule_db_server::{Server, ServerConfig};
use serde_json::{Value as JsonValue, json};
use tower::ServiceExt;

fn boot_test_server(dir: &std::path::Path) -> Server {
    let cfg = ServerConfig {
        db_path: dir.to_string_lossy().to_string(),
        auth_enabled: false,
        rate_limiting_enabled: false,
        ..ServerConfig::default()
    };
    Server::new(cfg).expect("boot test server")
}

async fn post_json(server: &Server, uri: &str, body: JsonValue) -> (StatusCode, JsonValue) {
    let app = server.router();
    let req = Request::builder()
        .method(Method::POST)
        .uri(uri)
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .expect("build post");
    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body");
    let parsed: JsonValue = serde_json::from_slice(&bytes).expect("parse json");
    (status, parsed)
}

async fn get_json(server: &Server, uri: &str) -> (StatusCode, JsonValue) {
    let app = server.router();
    let req = Request::builder()
        .method(Method::GET)
        .uri(uri)
        .body(Body::empty())
        .expect("build get");
    let resp = app.oneshot(req).await.expect("oneshot");
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), 16 * 1024 * 1024)
        .await
        .expect("read body");
    let parsed: JsonValue = serde_json::from_slice(&bytes).expect("parse json");
    (status, parsed)
}

#[tokio::test]
async fn test_ward_thermocap_attenuation_confinement_memoization_and_jcap_bundle_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(tmp.path());

    // 1. Mint root ThermoCap token
    let (status, minted) = post_json(
        &server,
        "/api/v1/ward/tokens/mint",
        json!({
            "root_token_id": "cap-e2e-root",
            "subject": "autonomous-builder",
            "caveats": [
                { "kind": "max_total_nanojoules", "value": 500_000_000_u64 }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let root_tok = minted["token"].clone();

    // 2. Offline-attenuate token: ReadOnlyFs(true), Airgap, HITL gate at 500_000_000 nJ
    let (status, att) = post_json(
        &server,
        "/api/v1/ward/tokens/attenuate",
        json!({
            "token": root_tok,
            "caveats": [
                { "kind": "read_only_fs", "value": true },
                { "kind": "allow_egress_hosts", "value": [] }
            ]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(att["attenuated_offline"], true);
    let child_tok = att["token"].clone();

    // 3. Compile multi-OS & WASI confinement profile
    let (status, compiled) = post_json(
        &server,
        "/api/v1/ward/confinement/compile",
        json!({ "token": child_tok }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let sbpl = compiled["confinement"]["macos_seatbelt_sbpl"]
        .as_str()
        .expect("sbpl");
    assert!(sbpl.contains("(deny network*)"));
    assert!(sbpl.contains("(deny file-write*)"));

    // 4. First confined execution -> runs process & populates Merkle CAS WardActionCache
    let exec_payload = json!({
        "token": child_tok,
        "command": ["/bin/echo", "jouleward-e2e-merkle"],
        "rootfs_merkle_hex": "0123456789abcdef",
        "memoize": true
    });
    let (status, run1) = post_json(&server, "/api/v1/ward/exec", exec_payload.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(run1["receipt"]["permitted"], true);
    assert_eq!(run1["receipt"]["memoized_cache_hit"], false);
    assert_eq!(
        run1["receipt"]["stdout"].as_str().map(str::trim),
        Some("jouleward-e2e-merkle")
    );
    let run1_nj = run1["receipt"]["energy_trace"]["aggregate_rails"]["cpu_nj"]
        .as_u64()
        .expect("cpu_nj");
    assert!(run1_nj > 85);

    // 5. Second identical execution -> 0 spawns, 85 nJ Merkle CAS replay!
    let (status, run2) = post_json(&server, "/api/v1/ward/exec", exec_payload).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(run2["receipt"]["memoized_cache_hit"], true);
    assert_eq!(
        run2["receipt"]["enforcement_mechanism"],
        "merkle-cas-zero-spawn-replay"
    );
    assert!(
        run2["receipt"]["energy_trace"]["saved_by_memoization_nj"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );

    // 6. Write attempt under ReadOnlyFs(true) -> 403 Forbidden!
    let (status, denied) = post_json(
        &server,
        "/api/v1/ward/exec",
        json!({
            "token": child_tok,
            "command": ["/usr/bin/touch", "illegal.txt"],
            "requires_fs_write": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(denied["receipt"]["permitted"], false);

    // 7. HITL gate escalation (`HitlGateAboveNanojoules(10)`) -> 428 Precondition Required, then 200 OK when approved!
    let (_, hitl_att) = post_json(
        &server,
        "/api/v1/ward/tokens/attenuate",
        json!({
            "token": child_tok,
            "caveats": [{ "kind": "hitl_gate_above_nanojoules", "value": 10_u64 }]
        }),
    )
    .await;
    let hitl_tok = hitl_att["token"].clone();
    let (status, hitl_pause) = post_json(
        &server,
        "/api/v1/ward/exec",
        json!({
            "token": hitl_tok,
            "command": ["/bin/echo", "needs-hitl"],
            "memoize": false
        }),
    )
    .await;
    assert_eq!(status, StatusCode::PRECONDITION_REQUIRED);
    assert_eq!(
        hitl_pause["receipt"]["decision"]["decision"],
        "requires_human_approval"
    );

    let (status, hitl_ok) = post_json(
        &server,
        "/api/v1/ward/exec",
        json!({
            "token": hitl_tok,
            "command": ["/bin/echo", "needs-hitl"],
            "memoize": false,
            "hitl_approved": true
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(hitl_ok["receipt"]["permitted"], true);

    // 8. Export & SHA-256 verify single-file `.jcap` bundle (`Apptainer` `.sif` disintermediation)
    let (status, exported) = post_json(
        &server,
        "/api/v1/ward/jcap/export",
        json!({
            "bundle_id": "appliance-e2e",
            "entrypoint": ["/bin/echo", "ready"],
            "files_utf8": { "/etc/app.json": "{\"ready\":true}" },
            "embedded_capability": child_tok
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let b64 = exported["jcap_base64"].as_str().expect("jcap_base64");

    let (status, verified) = post_json(
        &server,
        "/api/v1/ward/jcap/verify",
        json!({ "jcap_base64": b64 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(verified["capability_signature_valid"], true);
    assert_eq!(verified["bundle"]["bundle_id"], "appliance-e2e");

    // Corrupt 1 byte inside `.jcap` header -> 422 Unprocessable Entity!
    let mut raw = base64::engine::general_purpose::STANDARD
        .decode(b64)
        .expect("decode");
    raw[12] ^= 0xFF;
    let tampered_b64 = base64::engine::general_purpose::STANDARD.encode(&raw);
    let (status, _) = post_json(
        &server,
        "/api/v1/ward/jcap/verify",
        json!({ "jcap_base64": tampered_b64 }),
    )
    .await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn test_cross_fabric_heterogeneous_6rail_energy_accounting_e2e() {
    let dir_a = tempfile::tempdir().expect("dir_a");
    let dir_b = tempfile::tempdir().expect("dir_b");

    let node_a = boot_test_server(dir_a.path());
    let node_b = boot_test_server(dir_b.path());

    // Bind node_b on an ephemeral localhost TCP port to act as a remote `CloudCxl` hydro peer
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let addr = listener.local_addr().expect("addr");
    let router_b = node_b.router();
    let srv_handle = tokio::spawn(async move {
        let _ = axum::serve(listener, router_b).await;
    });

    // Register `hydro-peer-01` (`CloudCxl`, 15 gCO2/kWh) on `node_a`
    let peer_desc = FabricNodeDescriptor {
        node_id: "hydro-peer-01".into(),
        node_class: FabricNodeClass::CloudCxl,
        endpoint_url: Some(format!("http://{addr}")),
        lanes: vec![
            SiliconLane::CpuScalarSimd,
            SiliconLane::GpuShader,
            SiliconLane::PopcntBitwise,
        ],
        memory_bus: MemoryBusKind::CxlPooled,
        interconnect: InterconnectKind::DatacenterLan,
        tdp_watts: 120.0,
        current_power_watts: 30.0,
        grid_carbon_g_per_kwh: 15.0,
        battery_soc_pct: None,
        battery_charging: false,
        thermal_state: ThermalState::Nominal,
        utilization: 0.10,
    };
    let (status, _) = post_json(
        &node_a,
        "/api/v1/fabric/nodes",
        serde_json::to_value(peer_desc).expect("peer json"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // 1. Mint token forbidding wire hops (`NetworkWireRdma: 0`) -> peer hop MUST be blocked pre-flight!
    let (_, no_wire_mint) = post_json(
        &node_a,
        "/api/v1/ward/tokens/mint",
        json!({
            "root_token_id": "no-wire-cap",
            "subject": "local-only-worker",
            "caveats": [
                {
                    "kind": "max_rail_nanojoules",
                    "value": { "rail": "network_wire_rdma", "max_nj": 0_u64 }
                }
            ]
        }),
    )
    .await;
    let (status, blocked_hop) = post_json(
        &node_a,
        "/api/v1/ward/exec",
        json!({
            "token": no_wire_mint["token"],
            "command": ["/bin/echo", "should-not-cross-wire"],
            "target_node_id": "hydro-peer-01"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::FORBIDDEN);
    assert_eq!(blocked_hop["peer_forwarded"], false);

    // 2. Mint multi-fabric capability token -> execute across wire onto `hydro-peer-01` and verify 2-hop 6-rail receipt!
    let (_, fabric_mint) = post_json(
        &node_a,
        "/api/v1/ward/tokens/mint",
        json!({
            "root_token_id": "cross-fabric-cap",
            "subject": "distributed-orchestrator",
            "caveats": [
                { "kind": "max_total_nanojoules", "value": 900_000_000_u64 }
            ]
        }),
    )
    .await;
    let (status, hop_ok) = post_json(
        &node_a,
        "/api/v1/ward/exec",
        json!({
            "token": fabric_mint["token"],
            "command": ["/bin/echo", "executed-on-hydro-peer"],
            "target_node_id": "hydro-peer-01"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(hop_ok["peer_forwarded"], true);
    let hops = hop_ok["receipt"]["energy_trace"]["hops"]
        .as_array()
        .expect("hops");
    assert_eq!(
        hops.len(),
        2,
        "expected ingress wire hop + remote execution hop"
    );
    let agg_wire_nj = hop_ok["receipt"]["energy_trace"]["aggregate_rails"]["network_wire_nj"]
        .as_u64()
        .unwrap_or(0);
    assert!(
        agg_wire_nj >= 50,
        "expected positive network_wire_nj across fabric"
    );

    // 3. Also route OpenAI `/v1/chat/completions` across the compute fabric onto `hydro-peer-01`!
    let (status, chat_hop) = post_json(
        &node_a,
        "/v1/chat/completions",
        json!({
            "model": "jouledb-cascade-v1",
            "messages": [{ "role": "user", "content": "Compute carbon-optimal synthesis across fabric" }],
            "x_ward_token": fabric_mint["token"],
            "x_target_node_id": "hydro-peer-01"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(chat_hop["x_jouledb_energy"]["peer_forwarded"], true);
    assert!(
        chat_hop["x_jouledb_energy"]["aggregate_rails"]["network_wire_nj"]
            .as_u64()
            .unwrap_or(0)
            >= 60
    );

    // 4. Inspect `/api/v1/ward/ledger` on `node_a` -> proves multi-node, multi-rail nanojoule ledger!
    let (status, ledger) = get_json(&node_a, "/api/v1/ward/ledger").await;
    assert_eq!(status, StatusCode::OK);
    let cum_total_nj = ledger["ledger"]["cumulative_rails"]["cpu_nj"]
        .as_u64()
        .unwrap_or(0)
        + ledger["ledger"]["cumulative_rails"]["network_wire_nj"]
            .as_u64()
            .unwrap_or(0);
    assert!(cum_total_nj > 200);
    assert!(
        ledger["ledger"]["per_token_spent_nj"]["cross-fabric-cap"]
            .as_u64()
            .unwrap_or(0)
            > 200
    );

    srv_handle.abort();
}

#[tokio::test]
async fn test_openai_and_ollama_drop_in_disintermediation_with_4tier_cascade_e2e() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let server = boot_test_server(tmp.path());

    // 1. Factorize a live sub-1-bit `LittleBit-2` model layer (`512 x 256` -> ~0.56 BPW) & upsert an `Engram` conditional fact
    let weights: Vec<f32> = (0..(512 * 256))
        .map(|i| ((i as f32) * 0.13).sin() * 0.5)
        .collect();
    let (status, _) = post_json(
        &server,
        "/api/v1/synapse/littlebit/factorize",
        json!({
            "name": "minilm-test::encoder.layer.0.dense",
            "rows": 512,
            "cols": 256,
            "rank": 4,
            "weights": weights,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    let (status, _) = post_json(
        &server,
        "/api/v1/synapse/engram/upsert",
        json!({
            "phrase": "what is landauer limit at room temperature",
            "fact_payload": "k_B T ln(2) ≈ 2.87e-21 J/bit at 300 K"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);

    // 2. OpenAI `GET /v1/models` & Ollama `GET /api/tags`
    let (status, models) = get_json(&server, "/v1/models").await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(models["object"], "list");
    assert!(
        models["data"]
            .as_array()
            .expect("models")
            .iter()
            .any(|m| m["id"] == "minilm-test")
    );

    let (status, tags) = get_json(&server, "/api/tags").await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        tags["models"]
            .as_array()
            .expect("tags")
            .iter()
            .any(|m| m["model"] == "minilm-test")
    );

    // 3. OpenAI `POST /v1/embeddings` (batch of 2 inputs with 6-rail nanojoule receipt!)
    let (status, emb) = post_json(
        &server,
        "/v1/embeddings",
        json!({
            "model": "minilm-test",
            "input": ["thermodynamic database", "sub-one-bit living weights"]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(emb["data"].as_array().map(Vec::len), Some(2));
    assert_eq!(
        emb["data"][0]["embedding"].as_array().map(Vec::len),
        Some(384)
    );
    assert!(
        emb["x_jouledb_energy"]["bits_per_weight"]
            .as_f64()
            .unwrap_or(99.0)
            < 0.65
    );

    // 4. OpenAI `POST /v1/chat/completions`:
    //    - Call 1 -> Tier 4 `LittleBit-2` living forward
    //    - Call 2 (identical prompt) -> Tier 1 LSN `LangCache` hit in 95 nJ!
    //    - Call 3 (Engram phrase) -> Tier 2 O(1) `Engram` conditional memory hit in 220 nJ!
    let chat_req = json!({
        "model": "minilm-test",
        "messages": [{ "role": "user", "content": "Explain heterogeneous rail accounting across silicon" }]
    });
    let (status, c1) = post_json(&server, "/v1/chat/completions", chat_req.clone()).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        c1["x_jouledb_energy"]["cascade_tier"],
        "tier4_littlebit_living_synapse"
    );

    let (status, c2) = post_json(&server, "/v1/chat/completions", chat_req).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        c2["x_jouledb_energy"]["cascade_tier"],
        "tier1_lsn_langcache"
    );
    assert!(
        c2["x_jouledb_energy"]["saved_by_memoization_nj"]
            .as_u64()
            .unwrap_or(0)
            >= 900_000
    );

    let (status, c3) = post_json(
        &server,
        "/v1/chat/completions",
        json!({
            "model": "minilm-test",
            "messages": [{ "role": "user", "content": "what is landauer limit at room temperature" }]
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(
        c3["x_jouledb_energy"]["cascade_tier"],
        "tier2_engram_o1_memory"
    );
    assert_eq!(
        c3["choices"][0]["message"]["content"],
        "k_B T ln(2) ≈ 2.87e-21 J/bit at 300 K"
    );

    // 5. Ollama `POST /api/generate`
    let (status, ollama_gen) = post_json(
        &server,
        "/api/generate",
        json!({
            "model": "minilm-test",
            "prompt": "Synthesize embodied robot control loop"
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(ollama_gen["done"], true);
    assert!(
        ollama_gen["response"]
            .as_str()
            .unwrap_or("")
            .contains("JouleSynapse")
    );
}
