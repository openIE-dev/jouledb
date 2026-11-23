#!/usr/bin/env bash
# JouleDB Portable Fleet Platform Certifier (`fleet-certify.sh`)
#
# Run on ANY machine in the fleet (macOS Apple Silicon, Linux x86_64 RAPL,
# NVIDIA Jetson Orin/Thor, Raspberry Pi 5, AWS Graviton, CXL/CUDA Cloud, WSL2)
# to verify end-to-end correctness, hardware GPU/SIMD/Ternary lanes, and emit
# a machine-readable JSON certification record for the Heterogeneous Fabric & jouledb.com.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${JOULEDB_BIN:-$ROOT_DIR/target/release/jouledb}"
OUT_DIR="${JOULEDB_CERT_DIR:-$ROOT_DIR/deploy/jouledb/certifications}"
PORT="${JOULEDB_CERT_PORT:-18280}"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/jouledb-cert.XXXXXX")"
mkdir -p "$OUT_DIR"

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill "$SERVER_PID" 2>/dev/null || true
    fi
    rm -rf "$WORK_DIR"
}
trap cleanup EXIT

if [[ ! -x "$BIN" ]]; then
    echo "[certify] Building release jouledb binary..." >&2
    cargo build --release -p joule-db-server --bin jouledb
fi

HOST_TAG="${JOULEDB_CERT_SLUG:-$(hostname -s 2>/dev/null || uname -n)}"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"

JOULEDB_FABRIC_NODE_ID="cert-${HOST_TAG}" \
JOULEDB_LEDGER_ENABLED="1" \
JOULEDB_LEDGER_DIR="$WORK_DIR/ledger" \
JOULEDB_DISABLE_RATE_LIMIT="1" \
"$BIN" --port "$PORT" --data "$WORK_DIR/data" >"$WORK_DIR/server.log" 2>&1 &
SERVER_PID=$!

for _ in $(seq 1 50); do
    if curl -fsS "http://127.0.0.1:${PORT}/health/ready" >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done

# 1. SQL + Declarative Semantic Cascade
SQL_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/query" \
    -H "Content-Type: application/json" \
    -d '{"sql":"SELECT SEM_SIMILAR('\''heterogeneous silicon fabric'\'', '\''energy first silicon fabric'\'') AS sim"}')"

# 2. GPU / Parallel Shader Probe
GPU_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/fabric/execute" \
    -H "Content-Type: application/json" \
    -d '{"workload":{"kind":"parallel_columnar_or_dsp","ops_count":16777216,"working_set_bytes":1048576,"payload_bytes_if_remote":1048576,"latency_sla_us":50000,"max_energy_budget_uj":null,"causal_reflex":true,"carbon_weight":1.0},"batch_count":256,"vector_dim":512}')"

# 3. 1.58-Bit Ternary Probe
TERN_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/fabric/execute" \
    -H "Content-Type: application/json" \
    -d '{"workload":{"kind":"ternary_inference","ops_count":65536,"working_set_bytes":65536,"payload_bytes_if_remote":65536,"latency_sla_us":50000,"max_energy_budget_uj":null,"causal_reflex":true,"carbon_weight":1.0},"batch_count":128,"vector_dim":500}')"

# 4. Embodied Causal ASOF + Living SurpriseGate + HMI SigQL
SIGQL_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/hmi/sigql" \
    -H "Content-Type: application/json" \
    -d '{"signal_name":"eeg.alpha","sample_rate_hz":256.0,"samples":[0,0.24,0.47,0.68,0.84,0.95,1.0,0.98,0.90,0.77,0.59,0.38,0.14,-0.10,-0.34,-0.56],"query":"FROM eeg.alpha AGGREGATE { rms_val: rms } RETURNING confidence(0.95)"}')"

# 5. Post-Docker JouleCapsule (Build + Spawn + Fork in O(1) CoW without dockerd)
CAP_BUILD_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/capsules/images/build" \
    -H "Content-Type: application/json" \
    -d '{"tag":"cert-probe:v1","joulefile":"FROM scratch\nWORKDIR /app\nENV PROBE=1\nENERGY_BUDGET 200mJ\nCOPY hello.txt /app/hello.txt\n","context_files":{"hello.txt":"post-docker-ok"},"base64_encoded":false}')"
CAP_SPAWN_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/capsules/spawn" \
    -H "Content-Type: application/json" \
    -d '{"capsule_id":"cert-cap-a","image_ref":"cert-probe:v1"}')"
CAP_FORK_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/capsules/cert-cap-a/fork" \
    -H "Content-Type: application/json" \
    -d '{"new_capsule_id":"cert-cap-b","energy_budget_uj":100000}')"

# 6. 360-Degree SWOT TriModal RRF + HippoRAG PPR
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/hybrid/upsert" \
    -H "Content-Type: application/json" \
    -d '{"id":1,"text":"thermodynamic silicon fabric","embedding":[1,0,0,0],"role_fillers":[["k","v"]],"attributes":{},"neighbors":[2]}' >/dev/null
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/hybrid/upsert" \
    -H "Content-Type: application/json" \
    -d '{"id":2,"text":"merkle joule ledger","embedding":[0,1,0,0],"role_fillers":[],"attributes":{},"neighbors":[]}' >/dev/null
PPR_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/graphrag/retrieve" \
    -H "Content-Type: application/json" \
    -d '{"query_text":"thermodynamic silicon fabric","query_embedding":[1,0,0,0],"seed_k":1,"top_k":2}')"

# 7. Post-Silo JouleSpace (DeltaDB OpLog + Secret Scrub + LiveKit Atomic Barge-In)
SPACE_EDIT_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/space/edit" \
    -H "Content-Type: application/json" \
    -d '{"branch":"cert-space","actor":"agent","actor_kind":"ai_agent","path":"/app.rs","content":"let k = \"sk-certsecret1234567890\";","summary":"cert edit"}')"
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/space/rooms/join" \
    -H "Content-Type: application/json" \
    -d '{"room_id":"cert-room","linked_branch":"cert-space","participant":{"participant_id":"agent","kind":"ai_agent","activity":"idle","published_tracks":["audio_opus"],"cursor_anchor":null,"last_heartbeat_ms":1}}' >/dev/null
curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/space/rooms/turn" \
    -H "Content-Type: application/json" \
    -d '{"room_id":"cert-room","turn_id":"t1","agent_id":"agent","budgeted_uj":400000}' >/dev/null
BARGE_JSON="$(curl -fsS -X POST "http://127.0.0.1:${PORT}/api/v1/space/rooms/barge-in" \
    -H "Content-Type: application/json" \
    -d '{"room_id":"cert-room","interrupter_id":"human","reason":"vad"}')"

ENERGY_JSON="$(curl -fsS "http://127.0.0.1:${PORT}/api/v1/energy")"
TOPO_JSON="$(curl -fsS "http://127.0.0.1:${PORT}/api/v1/fabric/topology")"

CERT_FILE="$OUT_DIR/${OS_NAME}-${ARCH_NAME}-${HOST_TAG}.json"
python3 -c '
import json, sys, datetime
out_path, os_n, arch_n, host_n, sql_s, gpu_s, tern_s, sig_s, cap_b_s, cap_f_s, ppr_s, sp_e_s, bi_s, eng_s, topo_s = sys.argv[1:]
gpu = json.loads(gpu_s)
tern = json.loads(tern_s)
eng = json.loads(eng_s)
topo = json.loads(topo_s)
sig = json.loads(sig_s)
sql = json.loads(sql_s)
cap_b = json.loads(cap_b_s)
cap_f = json.loads(cap_f_s)
ppr = json.loads(ppr_s)
sp_e = json.loads(sp_e_s)
bi = json.loads(bi_s)
report = {
    "certified_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "platform_slug": host_n,
    "os": os_n,
    "arch": arch_n,
    "gpu_adapter": topo.get("gpu_adapter"),
    "node_descriptor": topo["nodes"][0] if topo.get("nodes") else None,
    "live_sensor_watts": eng.get("power_watts"),
    "thermal_state": eng.get("thermal_state"),
    "probes": {
        "sql_semantic_ok": bool(sql.get("rows")),
        "gpu_or_simd_lane": {
            "executed_lane": gpu.get("executed_lane"),
            "hardware_backend": gpu.get("hardware_backend"),
            "wall_time_us": gpu.get("wall_time_us"),
            "total_energy_uj": gpu["plan"]["selected"]["total_energy_uj"],
            "energy_saved_vs_worst_uj": gpu["plan"]["energy_saved_vs_worst_uj"],
        },
        "ternary_158bit_lane": {
            "executed_lane": tern.get("executed_lane"),
            "hardware_backend": tern.get("hardware_backend"),
            "wall_time_us": tern.get("wall_time_us"),
            "total_energy_uj": tern["plan"]["selected"]["total_energy_uj"],
        },
        "hmi_sigql_ok": sig.get("ok", False),
        "post_docker_joule_capsule_cow_ok": bool(cap_b.get("ok") and cap_f.get("ok")),
        "swot_trimodal_hipporag_ppr_ok": len(ppr.get("subgraph", [])) >= 2,
        "post_silo_joulespace_barge_in_ok": bool(sp_e.get("secrets_scrubbed") == 1 and bi.get("barge_in", {}).get("joules_reclaimed_uj", 0) > 0),
    },
}
with open(out_path, "w") as f:
    json.dump(report, f, indent=2)
print(json.dumps(report, indent=2))
' "$CERT_FILE" "$OS_NAME" "$ARCH_NAME" "$HOST_TAG" "$SQL_JSON" "$GPU_JSON" "$TERN_JSON" "$SIGQL_JSON" "$CAP_BUILD_JSON" "$CAP_FORK_JSON" "$PPR_JSON" "$SPACE_EDIT_JSON" "$BARGE_JSON" "$ENERGY_JSON" "$TOPO_JSON"
