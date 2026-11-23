#!/usr/bin/env bash
# Launch & wire a live 3-Node Heterogeneous Energy Fabric on localhost:
#   - Node 1 (`18080`): Apple Silicon UMA Coordinator (CPU + Metal GPU + Neural Engine)
#   - Node 2 (`18090`): Edge Robotic Arm Peer (5W TDP, 11% low battery SoC)
#   - Node 3 (`18100`): Hydroelectric Cloud CXL Peer (14 gCO2/kWh)
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BIN="${JOULEDB_BIN:-$ROOT_DIR/target/release/jouledb}"
STATE_DIR="${JOULEDB_FABRIC_STATE_DIR:-$ROOT_DIR/.jouledb-live-fabric}"

if [[ ! -x "$BIN" ]]; then
    echo "Building release jouledb binary..."
    cargo build --release -p joule-db-server --bin jouledb
fi

mkdir -p "$STATE_DIR/uma" "$STATE_DIR/edge" "$STATE_DIR/hydro" "$STATE_DIR/logs"

# Clean up any prior processes recorded in pidfile
if [[ -f "$STATE_DIR/fabric.pids" ]]; then
    while read -r pid; do
        kill "$pid" 2>/dev/null || true
    done < "$STATE_DIR/fabric.pids"
    rm -f "$STATE_DIR/fabric.pids"
    sleep 0.5
fi

JOULEDB_FABRIC_NODE_ID="uma-primary-01" \
JOULEDB_FABRIC_NODE_CLASS="apple_uma" \
JOULEDB_GRID_FACTOR="68.0" \
JOULEDB_LEDGER_ENABLED="1" \
JOULEDB_LEDGER_DIR="$STATE_DIR/uma/ledger" \
JOULEDB_DISABLE_RATE_LIMIT="1" \
"$BIN" --port 18080 --data "$STATE_DIR/uma/data" >"$STATE_DIR/logs/uma.log" 2>&1 &
echo $! >> "$STATE_DIR/fabric.pids"

JOULEDB_FABRIC_NODE_ID="edge-robot-01" \
JOULEDB_FABRIC_NODE_CLASS="edge_robotic" \
JOULEDB_BATTERY_SOC_PCT="11.0" \
JOULEDB_BATTERY_CHARGING="false" \
JOULEDB_GRID_FACTOR="140.0" \
JOULEDB_DISABLE_RATE_LIMIT="1" \
"$BIN" --port 18090 --data "$STATE_DIR/edge/data" >"$STATE_DIR/logs/edge.log" 2>&1 &
echo $! >> "$STATE_DIR/fabric.pids"

JOULEDB_FABRIC_NODE_ID="hydro-cloud-01" \
JOULEDB_FABRIC_NODE_CLASS="cloud_cxl" \
JOULEDB_GRID_FACTOR="14.0" \
JOULEDB_DISABLE_RATE_LIMIT="1" \
"$BIN" --port 18100 --data "$STATE_DIR/hydro/data" >"$STATE_DIR/logs/hydro.log" 2>&1 &
echo $! >> "$STATE_DIR/fabric.pids"

for port in 18080 18090 18100; do
    for _ in $(seq 1 50); do
        if curl -fsS "http://127.0.0.1:${port}/health/ready" >/dev/null 2>&1; then
            break
        fi
        sleep 0.1
    done
    curl -fsS "http://127.0.0.1:${port}/health/ready" >/dev/null
done

# Cross-register the 3 heterogeneous nodes into a live mesh
UMA_DESC='{
  "node_id":"uma-primary-01","node_class":"apple_uma","endpoint_url":"http://127.0.0.1:18080",
  "lanes":["cpu_scalar_simd","popcnt_bitwise","ternary158_bit","gpu_shader","npu_systolic","wasm_sandbox"],
  "memory_bus":"unified_zero_copy","interconnect":"datacenter_lan","tdp_watts":35.0,
  "current_power_watts":9.5,"grid_carbon_g_per_kwh":68.0,"battery_soc_pct":null,
  "battery_charging":false,"thermal_state":"Nominal","utilization":0.12
}'
EDGE_DESC='{
  "node_id":"edge-robot-01","node_class":"edge_robotic","endpoint_url":"http://127.0.0.1:18090",
  "lanes":["cpu_scalar_simd","popcnt_bitwise","ternary158_bit","npu_systolic"],
  "memory_bus":"local_dram","interconnect":"wireless_edge","tdp_watts":8.0,
  "current_power_watts":3.2,"grid_carbon_g_per_kwh":140.0,"battery_soc_pct":11.0,
  "battery_charging":false,"thermal_state":"Nominal","utilization":0.20
}'
HYDRO_DESC='{
  "node_id":"hydro-cloud-01","node_class":"cloud_cxl","endpoint_url":"http://127.0.0.1:18100",
  "lanes":["cpu_scalar_simd","popcnt_bitwise","ternary158_bit","gpu_shader","npu_systolic","lpu_deterministic_sram"],
  "memory_bus":"cxl_pooled","interconnect":"datacenter_lan","tdp_watts":350.0,
  "current_power_watts":110.0,"grid_carbon_g_per_kwh":14.0,"battery_soc_pct":null,
  "battery_charging":false,"thermal_state":"Nominal","utilization":0.15
}'

curl -fsS -X POST http://127.0.0.1:18080/api/v1/fabric/nodes -H "Content-Type: application/json" -d "$EDGE_DESC" >/dev/null
curl -fsS -X POST http://127.0.0.1:18080/api/v1/fabric/nodes -H "Content-Type: application/json" -d "$HYDRO_DESC" >/dev/null
curl -fsS -X POST http://127.0.0.1:18090/api/v1/fabric/nodes -H "Content-Type: application/json" -d "$UMA_DESC" >/dev/null
curl -fsS -X POST http://127.0.0.1:18090/api/v1/fabric/nodes -H "Content-Type: application/json" -d "$HYDRO_DESC" >/dev/null
curl -fsS -X POST http://127.0.0.1:18100/api/v1/fabric/nodes -H "Content-Type: application/json" -d "$UMA_DESC" >/dev/null
curl -fsS -X POST http://127.0.0.1:18100/api/v1/fabric/nodes -H "Content-Type: application/json" -d "$EDGE_DESC" >/dev/null

echo "LIVE_FABRIC_READY uma=http://127.0.0.1:18080 edge=http://127.0.0.1:18090 hydro=http://127.0.0.1:18100"
