//! Unified energy dashboard HTTP server.
//!
//! Runs alongside the daemon on port 7000 (configurable), aggregating energy
//! data from all managed database instances. Provides JSON API, Prometheus
//! metrics, and a simple HTML overview.
//!
//! Phase 4: queries each instance's energy sidecar for live power/energy data.

use crate::{InstanceInfo, InstanceState, RuntimeManager};
use std::sync::Arc;

/// Default dashboard port.
pub const DEFAULT_DASHBOARD_PORT: u16 = 7000;

/// Dashboard state shared across HTTP handlers.
pub struct DashboardState {
    pub manager: Arc<RuntimeManager>,
}

/// Deserialized response from an energy sidecar's `GET /energy` endpoint.
/// Mirrors the `EnergyResponse` struct in `energy_wrapper.rs` with
/// `#[serde(default)]` on every field for graceful partial parsing.
#[cfg(feature = "energy-sidecar")]
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct SidecarEnergyResponse {
    #[serde(default)]
    power_watts: f64,
    #[serde(default)]
    cumulative_joules: f64,
    #[serde(default)]
    cpu_utilization: f64,
    #[serde(default)]
    thermal_state: String,
    #[serde(default)]
    memory_pressure: f64,
    #[serde(default)]
    gpu_available: bool,
    #[serde(default)]
    gpu_utilization: f64,
    #[serde(default)]
    npu_available: bool,
    #[serde(default)]
    npu_utilization: f64,
    #[serde(default)]
    battery_percent: Option<f64>,
    #[serde(default)]
    battery_charging: bool,
}

/// Energy summary for a single instance.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InstanceEnergySummary {
    pub instance_id: String,
    pub name: String,
    pub engine: String,
    pub state: String,
    pub energy_port: Option<u16>,
    pub power_watts: f64,
    pub cumulative_joules: f64,
    pub cpu_utilization: f64,
    pub thermal_state: String,
    pub memory_pressure: f64,
    pub gpu_available: bool,
    pub gpu_utilization: f64,
    pub npu_available: bool,
    pub npu_utilization: f64,
    pub battery_percent: Option<f64>,
    pub battery_charging: bool,
    pub sidecar_reachable: bool,
    /// Number of accelerator devices allocated to this instance.
    pub accelerator_count: usize,
    /// Per-accelerator energy in joules (from AcceleratorManager).
    pub accelerator_energy_joules: f64,
    /// LLM inference: tokens per second (0.0 if not an LLM instance).
    #[serde(default)]
    pub llm_tokens_per_second: f64,
    /// LLM inference: joules per token (0.0 if not an LLM instance).
    #[serde(default)]
    pub llm_joules_per_token: f64,
    /// LLM inference: total tokens generated (0 if not an LLM instance).
    #[serde(default)]
    pub llm_total_tokens: u64,
    /// LLM inference: cost per million tokens in USD (0.0 if not applicable).
    #[serde(default)]
    pub llm_cost_per_million_tokens_usd: f64,
}

/// Aggregate energy data across all instances.
#[derive(Debug, Clone, serde::Serialize)]
pub struct EnergyAggregate {
    pub total_power_watts: f64,
    pub total_energy_joules: f64,
    pub instance_count: usize,
    pub instances: Vec<InstanceEnergySummary>,
}

/// Ledger summary for the dashboard. Mirrors the server's `/api/v1/ledger/stats`
/// response but with a subset of fields useful for the overview panel.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct LedgerDashboardSummary {
    #[serde(default)]
    pub total_receipts: u64,
    #[serde(default)]
    pub total_batches: u64,
    #[serde(default)]
    pub total_energy_joules: f64,
    #[serde(default)]
    pub total_kwh: f64,
    #[serde(default)]
    pub total_kg_co2e: f64,
}

/// Query a single energy sidecar. Returns `(response, reachable)`.
/// On any failure returns `(default, false)`.
#[cfg(feature = "energy-sidecar")]
async fn query_sidecar(client: &reqwest::Client, port: u16) -> (SidecarEnergyResponse, bool) {
    let url = format!("http://127.0.0.1:{}/energy", port);
    match client.get(&url).send().await {
        Ok(resp) if resp.status().is_success() => {
            match resp.json::<SidecarEnergyResponse>().await {
                Ok(energy) => (energy, true),
                Err(_) => (SidecarEnergyResponse::default(), false),
            }
        }
        _ => (SidecarEnergyResponse::default(), false),
    }
}

/// Query all running instances' sidecars and return total cumulative energy (joules).
///
/// Used by the daemon's status handler without needing a `DashboardState`.
/// Fans out with `tokio::task::JoinSet` for concurrent queries with 2s timeout.
#[cfg(feature = "energy-sidecar")]
pub async fn total_energy_from_sidecars(instances: &[InstanceInfo]) -> f64 {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap_or_default();

    let mut join_set = tokio::task::JoinSet::new();
    for inst in instances {
        if inst.state == InstanceState::Running {
            if let Some(port) = inst.energy_port {
                let client = client.clone();
                join_set.spawn(async move {
                    let (resp, _) = query_sidecar(&client, port).await;
                    resp.cumulative_joules
                });
            }
        }
    }

    let mut total = 0.0;
    while let Some(result) = join_set.join_next().await {
        if let Ok(joules) = result {
            total += joules;
        }
    }
    total
}

/// Stub when energy-sidecar feature is disabled.
#[cfg(not(feature = "energy-sidecar"))]
pub async fn total_energy_from_sidecars(_instances: &[InstanceInfo]) -> f64 {
    0.0
}

/// Build a zeroed `InstanceEnergySummary` from instance metadata.
fn zeroed_summary(inst: &InstanceInfo) -> InstanceEnergySummary {
    InstanceEnergySummary {
        instance_id: inst.id.to_string(),
        name: inst.name.clone(),
        engine: inst.engine.to_string(),
        state: inst.state.to_string(),
        energy_port: inst.energy_port,
        power_watts: 0.0,
        cumulative_joules: 0.0,
        cpu_utilization: 0.0,
        thermal_state: "unknown".to_string(),
        memory_pressure: 0.0,
        gpu_available: false,
        gpu_utilization: 0.0,
        npu_available: false,
        npu_utilization: 0.0,
        battery_percent: None,
        battery_charging: false,
        sidecar_reachable: false,
        accelerator_count: inst.accelerators.len(),
        accelerator_energy_joules: 0.0,
        llm_tokens_per_second: 0.0,
        llm_joules_per_token: 0.0,
        llm_total_tokens: 0,
        llm_cost_per_million_tokens_usd: 0.0,
    }
}

impl DashboardState {
    /// List all instances as JSON-serializable structs.
    pub fn list_instances(&self) -> Vec<InstanceInfo> {
        self.manager.list_instances()
    }

    /// Compute aggregate energy data by querying each instance's energy sidecar.
    ///
    /// Fans out HTTP queries concurrently with a 2-second timeout per sidecar.
    /// Instances without a sidecar or that are not running return zeroed data.
    #[cfg(feature = "energy-sidecar")]
    pub async fn energy_aggregate(&self) -> EnergyAggregate {
        let instances = self.manager.list_instances();
        let accel_mgr = self.manager.accelerator_manager();
        let llm_rt = self.manager.llm_runtime();
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap_or_default();

        let mut join_set = tokio::task::JoinSet::new();

        for inst in &instances {
            let inst_id = inst.id.to_string();
            let name = inst.name.clone();
            let engine = inst.engine.to_string();
            let state_str = inst.state.to_string();
            let energy_port = inst.energy_port;
            let is_running = inst.state == InstanceState::Running;
            let client = client.clone();
            let accel_count = accel_mgr.devices_for_instance(inst.id.as_str()).len();
            let accel_energy = accel_mgr.energy_for_instance(inst.id.as_str());

            // LLM telemetry (if this instance is running an LLM)
            let llm_telem = llm_rt.get_telemetry(inst.id.as_str());

            join_set.spawn(async move {
                let (llm_tps, llm_jpt, llm_tokens, llm_cost) = match &llm_telem {
                    Some(t) => (
                        t.tokens_per_second,
                        t.joules_per_token,
                        t.total_tokens,
                        t.cost_per_million_tokens_usd,
                    ),
                    None => (0.0, 0.0, 0, 0.0),
                };

                if is_running {
                    if let Some(port) = energy_port {
                        let (resp, reachable) = query_sidecar(&client, port).await;
                        return InstanceEnergySummary {
                            instance_id: inst_id,
                            name,
                            engine,
                            state: state_str,
                            energy_port,
                            power_watts: resp.power_watts,
                            cumulative_joules: resp.cumulative_joules,
                            cpu_utilization: resp.cpu_utilization,
                            thermal_state: resp.thermal_state,
                            memory_pressure: resp.memory_pressure,
                            gpu_available: resp.gpu_available,
                            gpu_utilization: resp.gpu_utilization,
                            npu_available: resp.npu_available,
                            npu_utilization: resp.npu_utilization,
                            battery_percent: resp.battery_percent,
                            battery_charging: resp.battery_charging,
                            sidecar_reachable: reachable,
                            accelerator_count: accel_count,
                            accelerator_energy_joules: accel_energy,
                            llm_tokens_per_second: llm_tps,
                            llm_joules_per_token: llm_jpt,
                            llm_total_tokens: llm_tokens,
                            llm_cost_per_million_tokens_usd: llm_cost,
                        };
                    }
                }
                InstanceEnergySummary {
                    instance_id: inst_id,
                    name,
                    engine,
                    state: state_str,
                    energy_port,
                    power_watts: 0.0,
                    cumulative_joules: 0.0,
                    cpu_utilization: 0.0,
                    thermal_state: "unknown".to_string(),
                    memory_pressure: 0.0,
                    gpu_available: false,
                    gpu_utilization: 0.0,
                    npu_available: false,
                    npu_utilization: 0.0,
                    battery_percent: None,
                    battery_charging: false,
                    sidecar_reachable: false,
                    accelerator_count: accel_count,
                    accelerator_energy_joules: accel_energy,
                    llm_tokens_per_second: llm_tps,
                    llm_joules_per_token: llm_jpt,
                    llm_total_tokens: llm_tokens,
                    llm_cost_per_million_tokens_usd: llm_cost,
                }
            });
        }

        let mut summaries = Vec::with_capacity(instances.len());
        while let Some(result) = join_set.join_next().await {
            if let Ok(summary) = result {
                summaries.push(summary);
            }
        }

        EnergyAggregate {
            total_power_watts: summaries.iter().map(|s| s.power_watts).sum(),
            total_energy_joules: summaries.iter().map(|s| s.cumulative_joules).sum(),
            instance_count: summaries.len(),
            instances: summaries,
        }
    }

    /// Compute aggregate energy data (stub when energy-sidecar is disabled).
    #[cfg(not(feature = "energy-sidecar"))]
    pub async fn energy_aggregate(&self) -> EnergyAggregate {
        let instances = self.manager.list_instances();
        let summaries: Vec<InstanceEnergySummary> =
            instances.iter().map(|inst| zeroed_summary(inst)).collect();

        EnergyAggregate {
            total_power_watts: 0.0,
            total_energy_joules: 0.0,
            instance_count: summaries.len(),
            instances: summaries,
        }
    }

    /// Health check all instances.
    pub async fn health_all(&self) -> Vec<(String, bool)> {
        let instances = self.manager.list_instances();
        let mut results = Vec::new();

        for instance in instances {
            let healthy = if instance.state == InstanceState::Running {
                self.manager
                    .health_check(instance.id.as_str())
                    .await
                    .unwrap_or(false)
            } else {
                false
            };
            results.push((instance.id.to_string(), healthy));
        }

        results
    }

    /// Fetch ledger summary from a running JouleDB server instance.
    ///
    /// Queries the server's `GET /api/v1/ledger/stats` endpoint.
    /// Returns zeroed summary if the server is unreachable or the feature is off.
    #[cfg(feature = "energy-sidecar")]
    pub async fn ledger_summary(&self, server_port: u16) -> LedgerDashboardSummary {
        let url = format!("http://127.0.0.1:{}/api/v1/ledger/stats", server_port);
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(2))
            .build()
            .unwrap_or_default();

        match client.get(&url).send().await {
            Ok(resp) if resp.status().is_success() => resp.json().await.unwrap_or_default(),
            _ => LedgerDashboardSummary::default(),
        }
    }

    /// Generate Prometheus-format metrics including per-instance energy data.
    pub async fn prometheus_metrics(&self) -> String {
        let agg = self.energy_aggregate().await;
        let instances = self.manager.list_instances();
        let mut out = String::new();

        out.push_str("# HELP jouledb_instances_total Number of managed instances\n");
        out.push_str("# TYPE jouledb_instances_total gauge\n");
        out.push_str(&format!("jouledb_instances_total {}\n\n", instances.len()));

        let running = instances
            .iter()
            .filter(|i| i.state == InstanceState::Running)
            .count();
        out.push_str("# HELP jouledb_instances_running Number of running instances\n");
        out.push_str("# TYPE jouledb_instances_running gauge\n");
        out.push_str(&format!("jouledb_instances_running {}\n\n", running));

        out.push_str("# HELP jouledb_instance_state Instance state (1=running, 0=not)\n");
        out.push_str("# TYPE jouledb_instance_state gauge\n");
        for inst in &instances {
            let is_running = if inst.state == InstanceState::Running {
                1
            } else {
                0
            };
            out.push_str(&format!(
                "jouledb_instance_state{{id=\"{}\",name=\"{}\",engine=\"{}\"}} {}\n",
                inst.id, inst.name, inst.engine, is_running
            ));
        }

        // Per-instance energy gauges
        out.push_str("\n# HELP jouledb_instance_power_watts Current power draw per instance\n");
        out.push_str("# TYPE jouledb_instance_power_watts gauge\n");
        for s in &agg.instances {
            out.push_str(&format!(
                "jouledb_instance_power_watts{{id=\"{}\",name=\"{}\",engine=\"{}\"}} {:.2}\n",
                s.instance_id, s.name, s.engine, s.power_watts
            ));
        }

        out.push_str("\n# HELP jouledb_instance_energy_joules Cumulative energy per instance\n");
        out.push_str("# TYPE jouledb_instance_energy_joules counter\n");
        for s in &agg.instances {
            out.push_str(&format!(
                "jouledb_instance_energy_joules{{id=\"{}\",name=\"{}\",engine=\"{}\"}} {:.2}\n",
                s.instance_id, s.name, s.engine, s.cumulative_joules
            ));
        }

        out.push_str("\n# HELP jouledb_instance_cpu_utilization CPU utilization per instance\n");
        out.push_str("# TYPE jouledb_instance_cpu_utilization gauge\n");
        for s in &agg.instances {
            out.push_str(&format!(
                "jouledb_instance_cpu_utilization{{id=\"{}\",name=\"{}\",engine=\"{}\"}} {:.4}\n",
                s.instance_id, s.name, s.engine, s.cpu_utilization
            ));
        }

        // Aggregates
        out.push_str("\n# HELP jouledb_total_power_watts Total power across all instances\n");
        out.push_str("# TYPE jouledb_total_power_watts gauge\n");
        out.push_str(&format!(
            "jouledb_total_power_watts {:.2}\n",
            agg.total_power_watts
        ));

        out.push_str("\n# HELP jouledb_total_energy_joules Total energy across all instances\n");
        out.push_str("# TYPE jouledb_total_energy_joules counter\n");
        out.push_str(&format!(
            "jouledb_total_energy_joules {:.2}\n",
            agg.total_energy_joules
        ));

        // Per-instance accelerator metrics
        out.push_str(
            "\n# HELP jouledb_instance_accelerator_count Accelerator devices per instance\n",
        );
        out.push_str("# TYPE jouledb_instance_accelerator_count gauge\n");
        for s in &agg.instances {
            out.push_str(&format!(
                "jouledb_instance_accelerator_count{{id=\"{}\",name=\"{}\"}} {}\n",
                s.instance_id, s.name, s.accelerator_count
            ));
        }

        out.push_str(
            "\n# HELP jouledb_instance_accelerator_energy_joules Accelerator energy per instance\n",
        );
        out.push_str("# TYPE jouledb_instance_accelerator_energy_joules counter\n");
        for s in &agg.instances {
            out.push_str(&format!(
                "jouledb_instance_accelerator_energy_joules{{id=\"{}\",name=\"{}\"}} {:.2}\n",
                s.instance_id, s.name, s.accelerator_energy_joules
            ));
        }

        // LLM inference metrics (only for instances with LLM telemetry)
        let has_llm = agg.instances.iter().any(|s| s.llm_total_tokens > 0);
        if has_llm {
            out.push_str(
                "\n# HELP jouledb_llm_tokens_per_second LLM inference throughput (tok/s)\n",
            );
            out.push_str("# TYPE jouledb_llm_tokens_per_second gauge\n");
            for s in &agg.instances {
                if s.llm_total_tokens > 0 {
                    out.push_str(&format!(
                        "jouledb_llm_tokens_per_second{{id=\"{}\",name=\"{}\"}} {:.2}\n",
                        s.instance_id, s.name, s.llm_tokens_per_second
                    ));
                }
            }

            out.push_str("\n# HELP jouledb_llm_joules_per_token Energy cost per token (J/tok)\n");
            out.push_str("# TYPE jouledb_llm_joules_per_token gauge\n");
            for s in &agg.instances {
                if s.llm_total_tokens > 0 {
                    out.push_str(&format!(
                        "jouledb_llm_joules_per_token{{id=\"{}\",name=\"{}\"}} {:.6}\n",
                        s.instance_id, s.name, s.llm_joules_per_token
                    ));
                }
            }

            out.push_str("\n# HELP jouledb_llm_total_tokens Total tokens generated\n");
            out.push_str("# TYPE jouledb_llm_total_tokens counter\n");
            for s in &agg.instances {
                if s.llm_total_tokens > 0 {
                    out.push_str(&format!(
                        "jouledb_llm_total_tokens{{id=\"{}\",name=\"{}\"}} {}\n",
                        s.instance_id, s.name, s.llm_total_tokens
                    ));
                }
            }

            out.push_str(
                "\n# HELP jouledb_llm_cost_per_million_tokens_usd Cost per 1M tokens (USD)\n",
            );
            out.push_str("# TYPE jouledb_llm_cost_per_million_tokens_usd gauge\n");
            for s in &agg.instances {
                if s.llm_total_tokens > 0 {
                    out.push_str(&format!(
                        "jouledb_llm_cost_per_million_tokens_usd{{id=\"{}\",name=\"{}\"}} {:.6}\n",
                        s.instance_id, s.name, s.llm_cost_per_million_tokens_usd
                    ));
                }
            }
        }

        // Accelerator device inventory
        let accel_devices = self.manager.accelerator_manager().list_devices();
        out.push_str(
            "\n# HELP jouledb_accelerator_devices_total Total detected accelerator devices\n",
        );
        out.push_str("# TYPE jouledb_accelerator_devices_total gauge\n");
        out.push_str(&format!(
            "jouledb_accelerator_devices_total {}\n",
            accel_devices.len()
        ));

        out
    }
}

/// Start the dashboard HTTP server using axum.
///
/// Returns a `JoinHandle` that can be aborted to stop the server.
#[cfg(feature = "energy-sidecar")]
pub async fn start_dashboard(
    state: Arc<DashboardState>,
    port: u16,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    start_dashboard_with_listener(state, listener)
}

/// Start the dashboard on a pre-bound listener. Avoids TOCTOU port races
/// in tests (bind port 0, then pass the listener here).
pub fn start_dashboard_with_listener(
    state: Arc<DashboardState>,
    listener: tokio::net::TcpListener,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    use axum::{Router, routing::get};

    let app = Router::new()
        .route("/", get(handle_index))
        .route("/api/instances", get(handle_instances))
        .route("/api/energy", get(handle_energy))
        .route("/api/accelerators", get(handle_accelerators))
        .route("/api/health", get(handle_health))
        .route("/api/ledger", get(handle_ledger))
        .route("/metrics", get(handle_metrics))
        .with_state(state);

    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            log::error!("Dashboard server error: {}", e);
        }
    });

    Ok(handle)
}

#[cfg(feature = "energy-sidecar")]
async fn handle_index(
    axum::extract::State(state): axum::extract::State<Arc<DashboardState>>,
) -> impl axum::response::IntoResponse {
    let energy = state.energy_aggregate().await;

    let html = format!(
        r#"<!DOCTYPE html>
<html><head><title>JouleDB Dashboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; }}
  th {{ background: #f5f5f5; }}
  .live {{ color: green; }}
  .dead {{ color: #aaa; }}
</style>
</head>
<body>
<h1>JouleDB Energy Dashboard</h1>
<p>Instances: {} | Total Power: {:.1}W | Total Energy: {:.1}J</p>
<table>
<tr><th>ID</th><th>Name</th><th>Engine</th><th>State</th>
<th>Power (W)</th><th>Energy (J)</th><th>CPU</th><th>GPU</th><th>NPU</th>
<th>Accel</th><th>Accel Energy (J)</th><th>tok/s</th><th>J/tok</th><th>Thermal</th><th>Sidecar</th></tr>
{}
</table>
<p><small><a href="/">reload</a> |
<a href="/api/energy">JSON</a> | <a href="/metrics">Prometheus</a></small></p>
</body></html>"#,
        energy.instance_count,
        energy.total_power_watts,
        energy.total_energy_joules,
        energy
            .instances
            .iter()
            .map(|s| format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td>\
                 <td>{:.1}</td><td>{:.1}</td><td>{:.0}%</td>\
                 <td>{}</td><td>{}</td>\
                 <td>{}</td><td>{:.1}</td>\
                 <td>{}</td><td>{}</td>\
                 <td>{}</td>\
                 <td class=\"{}\">{}</td></tr>",
                &s.instance_id[..8.min(s.instance_id.len())],
                s.name,
                s.engine,
                s.state,
                s.power_watts,
                s.cumulative_joules,
                s.cpu_utilization * 100.0,
                if s.gpu_available {
                    format!("{:.0}%", s.gpu_utilization * 100.0)
                } else {
                    "-".to_string()
                },
                if s.npu_available {
                    format!("{:.0}%", s.npu_utilization * 100.0)
                } else {
                    "-".to_string()
                },
                if s.accelerator_count > 0 {
                    s.accelerator_count.to_string()
                } else {
                    "-".to_string()
                },
                s.accelerator_energy_joules,
                if s.llm_total_tokens > 0 {
                    format!("{:.1}", s.llm_tokens_per_second)
                } else {
                    "-".to_string()
                },
                if s.llm_total_tokens > 0 {
                    format!("{:.4}", s.llm_joules_per_token)
                } else {
                    "-".to_string()
                },
                s.thermal_state,
                if s.sidecar_reachable { "live" } else { "dead" },
                if s.sidecar_reachable {
                    "live"
                } else {
                    "unreachable"
                },
            ))
            .collect::<Vec<_>>()
            .join("\n")
    );

    axum::response::Html(html)
}

#[cfg(feature = "energy-sidecar")]
async fn handle_instances(
    axum::extract::State(state): axum::extract::State<Arc<DashboardState>>,
) -> axum::Json<Vec<InstanceInfo>> {
    axum::Json(state.list_instances())
}

#[cfg(feature = "energy-sidecar")]
async fn handle_energy(
    axum::extract::State(state): axum::extract::State<Arc<DashboardState>>,
) -> axum::Json<EnergyAggregate> {
    axum::Json(state.energy_aggregate().await)
}

#[cfg(feature = "energy-sidecar")]
async fn handle_accelerators(
    axum::extract::State(state): axum::extract::State<Arc<DashboardState>>,
) -> axum::Json<serde_json::Value> {
    let devices = state.manager.accelerator_manager().list_devices();
    let available = devices.iter().filter(|d| d.allocated_to.is_none()).count();
    axum::Json(serde_json::json!({
        "total_devices": devices.len(),
        "available_devices": available,
        "devices": devices,
    }))
}

#[cfg(feature = "energy-sidecar")]
async fn handle_health(
    axum::extract::State(state): axum::extract::State<Arc<DashboardState>>,
) -> axum::Json<Vec<(String, bool)>> {
    let results = state.health_all().await;
    axum::Json(results)
}

#[cfg(feature = "energy-sidecar")]
async fn handle_ledger(
    axum::extract::State(state): axum::extract::State<Arc<DashboardState>>,
) -> axum::Json<LedgerDashboardSummary> {
    // Default port 3000 for JouleDB server. In a production setup this
    // would come from configuration, but for the dashboard integration
    // querying localhost is the standard pattern.
    let summary = state.ledger_summary(3000).await;
    axum::Json(summary)
}

#[cfg(feature = "energy-sidecar")]
async fn handle_metrics(
    axum::extract::State(state): axum::extract::State<Arc<DashboardState>>,
) -> String {
    state.prometheus_metrics().await
}

/// Stub for when the energy-sidecar feature is disabled.
#[cfg(not(feature = "energy-sidecar"))]
pub async fn start_dashboard(
    _state: Arc<DashboardState>,
    _port: u16,
) -> Result<tokio::task::JoinHandle<()>, std::io::Error> {
    Ok(tokio::spawn(async {}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RuntimeConfig;

    fn make_state(tmp: &tempfile::TempDir) -> Arc<DashboardState> {
        let manager = Arc::new(
            RuntimeManager::new(RuntimeConfig::default(), tmp.path().to_path_buf()).unwrap(),
        );
        Arc::new(DashboardState { manager })
    }

    #[tokio::test]
    async fn test_energy_aggregate_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let state = make_state(&tmp);
        let agg = state.energy_aggregate().await;
        assert_eq!(agg.instance_count, 0);
        assert_eq!(agg.total_power_watts, 0.0);
        assert!(agg.instances.is_empty());
    }

    #[tokio::test]
    async fn test_prometheus_metrics_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let state = make_state(&tmp);
        let metrics = state.prometheus_metrics().await;
        assert!(metrics.contains("jouledb_instances_total 0"));
        assert!(metrics.contains("jouledb_instances_running 0"));
        assert!(metrics.contains("jouledb_total_power_watts"));
        assert!(metrics.contains("jouledb_total_energy_joules"));
    }

    #[tokio::test]
    async fn test_health_all_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let state = make_state(&tmp);
        let results = state.health_all().await;
        assert!(results.is_empty());
    }

    #[cfg(feature = "energy-sidecar")]
    #[tokio::test]
    async fn test_dashboard_starts_and_responds() {
        let tmp = tempfile::tempdir().unwrap();
        let state = make_state(&tmp);

        // Bind ephemeral port and pass listener directly — no TOCTOU race
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = start_dashboard_with_listener(state, listener).unwrap();

        // Give server time to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Hit /api/instances
        let resp = reqwest::get(&format!("http://127.0.0.1:{}/api/instances", port))
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body: Vec<InstanceInfo> = resp.json().await.unwrap();
        assert!(body.is_empty());

        // Hit /metrics
        let resp = reqwest::get(&format!("http://127.0.0.1:{}/metrics", port))
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let text = resp.text().await.unwrap();
        assert!(text.contains("jouledb_instances_total"));
        assert!(text.contains("jouledb_total_power_watts"));

        handle.abort();
    }

    #[test]
    fn test_ledger_summary_default_is_zeroed() {
        let summary = LedgerDashboardSummary::default();
        assert_eq!(summary.total_receipts, 0);
        assert_eq!(summary.total_batches, 0);
        assert_eq!(summary.total_energy_joules, 0.0);
        assert_eq!(summary.total_kwh, 0.0);
        assert_eq!(summary.total_kg_co2e, 0.0);
    }

    #[cfg(feature = "energy-sidecar")]
    #[tokio::test]
    async fn test_ledger_route_returns_zeroed_when_no_server() {
        let tmp = tempfile::tempdir().unwrap();
        let state = make_state(&tmp);

        // No JouleDB server running → should return default zeros
        let summary = state.ledger_summary(19999).await;
        assert_eq!(summary.total_receipts, 0);
        assert_eq!(summary.total_batches, 0);
    }

    #[cfg(feature = "energy-sidecar")]
    #[tokio::test]
    async fn test_query_sidecar_unreachable() {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(1))
            .build()
            .unwrap();
        let (resp, reachable) = query_sidecar(&client, 59999).await;
        assert!(!reachable);
        assert_eq!(resp.power_watts, 0.0);
        assert_eq!(resp.cumulative_joules, 0.0);
    }

    #[tokio::test]
    async fn test_total_energy_from_sidecars_no_instances() {
        let total = total_energy_from_sidecars(&[]).await;
        assert_eq!(total, 0.0);
    }

    #[test]
    fn test_instance_energy_summary_serializes_new_fields() {
        let summary = InstanceEnergySummary {
            instance_id: "test-001".to_string(),
            name: "pg".to_string(),
            engine: "PostgreSQL".to_string(),
            state: "running".to_string(),
            energy_port: Some(15432),
            power_watts: 25.5,
            cumulative_joules: 1234.0,
            cpu_utilization: 0.45,
            thermal_state: "Nominal".to_string(),
            memory_pressure: 0.2,
            gpu_available: true,
            gpu_utilization: 0.1,
            npu_available: false,
            npu_utilization: 0.0,
            battery_percent: Some(85.0),
            battery_charging: true,
            sidecar_reachable: true,
            accelerator_count: 2,
            accelerator_energy_joules: 500.0,
            llm_tokens_per_second: 42.5,
            llm_joules_per_token: 0.032,
            llm_total_tokens: 10000,
            llm_cost_per_million_tokens_usd: 0.00107,
        };
        let json = serde_json::to_value(&summary).unwrap();
        assert_eq!(json["power_watts"], 25.5);
        assert_eq!(json["cpu_utilization"], 0.45);
        assert_eq!(json["thermal_state"], "Nominal");
        assert_eq!(json["gpu_available"], true);
        assert_eq!(json["sidecar_reachable"], true);
        assert_eq!(json["battery_percent"], 85.0);
        assert_eq!(json["accelerator_count"], 2);
        assert_eq!(json["accelerator_energy_joules"], 500.0);
        assert_eq!(json["llm_tokens_per_second"], 42.5);
        assert_eq!(json["llm_joules_per_token"], 0.032);
        assert_eq!(json["llm_total_tokens"], 10000);
    }

    #[cfg(feature = "energy-sidecar")]
    #[tokio::test]
    async fn test_dashboard_energy_route_enriched() {
        let tmp = tempfile::tempdir().unwrap();
        let state = make_state(&tmp);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();

        let handle = start_dashboard_with_listener(state, listener).unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let resp = reqwest::get(&format!("http://127.0.0.1:{}/api/energy", port))
            .await
            .unwrap();
        assert!(resp.status().is_success());
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body.get("total_power_watts").is_some());
        assert!(body.get("total_energy_joules").is_some());
        assert!(body.get("instances").unwrap().is_array());

        handle.abort();
    }
}
