//! Energy sidecar — lightweight HTTP server providing energy telemetry for any database.
//!
//! When the Joule runtime launches an external database (Postgres, MySQL, Redis, etc.),
//! it starts an energy sidecar alongside it. The sidecar monitors hardware energy
//! consumption (power_watts, thermal_state, CPU/GPU/NPU utilization) and exposes
//! three endpoints:
//!
//! - `GET /energy` — JSON snapshot of current energy state
//! - `GET /metrics` — Prometheus text format with `engine="postgres"` labels
//! - `GET /health` — checks if the wrapped database process is still alive
//!
//! JouleDB itself does NOT need the sidecar (it has built-in energy endpoints).
//! This module is feature-gated behind `energy-sidecar`.

use crate::DatabaseEngine;
use crate::accelerator::AcceleratorManager;
use joule_db_energy::{EnergyConfig, EnergyMonitor, EnergySnapshot, detect_platform};
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::{Json, Router, extract::State, response::IntoResponse, routing::get};
use serde::Serialize;

/// State shared between sidecar HTTP handlers.
#[derive(Clone)]
struct SidecarState {
    snapshot: Arc<RwLock<EnergySnapshot>>,
    engine_name: String,
    wrapped_pid: u32,
    port: u16,
    /// Optional accelerator manager for per-device energy tracking.
    accelerator_manager: Option<Arc<AcceleratorManager>>,
    /// Last energy accumulation timestamp for delta calculation.
    last_accumulate: Arc<RwLock<Option<Instant>>>,
}

/// JSON response for GET /energy.
#[derive(Serialize)]
struct EnergyResponse {
    engine: String,
    power_watts: f64,
    cumulative_joules: f64,
    cpu_utilization: f64,
    thermal_state: String,
    memory_pressure: f64,
    memory_used_bytes: u64,
    memory_total_bytes: u64,
    gpu_available: bool,
    gpu_utilization: f64,
    npu_available: bool,
    npu_utilization: f64,
    tpu_available: bool,
    tpu_utilization: f64,
    lpu_available: bool,
    lpu_utilization: f64,
    battery_percent: Option<f64>,
    battery_charging: bool,
    swap_used_bytes: u64,
}

/// Handle for a running energy sidecar. Drop or abort to stop.
pub struct EnergySidecar {
    pub port: u16,
    task_handle: tokio::task::JoinHandle<()>,
    _monitor_handle: std::thread::JoinHandle<()>,
}

impl EnergySidecar {
    /// Start an energy sidecar for an external database process.
    ///
    /// - `engine`: which database engine is being monitored
    /// - `wrapped_pid`: PID of the database process
    /// - `port`: port to bind the sidecar HTTP server on
    /// - `engine_port`: the database's listening port (for health display)
    pub async fn start(
        engine: DatabaseEngine,
        wrapped_pid: u32,
        port: u16,
        _engine_port: u16,
    ) -> Result<Self, std::io::Error> {
        Self::start_with_accelerators(engine, wrapped_pid, port, _engine_port, None).await
    }

    /// Start an energy sidecar with optional accelerator tracking.
    pub async fn start_with_accelerators(
        engine: DatabaseEngine,
        wrapped_pid: u32,
        port: u16,
        _engine_port: u16,
        accelerator_manager: Option<Arc<AcceleratorManager>>,
    ) -> Result<Self, std::io::Error> {
        let config = EnergyConfig::default();
        let monitor = EnergyMonitor::new(config);
        let (snapshot, monitor_handle) = monitor.start_background();

        let state = SidecarState {
            snapshot,
            engine_name: engine.display_name().to_string(),
            wrapped_pid,
            port,
            accelerator_manager,
            last_accumulate: Arc::new(RwLock::new(None)),
        };

        let app = Router::new()
            .route("/energy", get(energy_handler))
            .route("/energy/devices", get(devices_handler))
            .route("/metrics", get(metrics_handler))
            .route("/health", get(health_handler))
            .with_state(state);

        let listener = tokio::net::TcpListener::bind(format!("0.0.0.0:{}", port)).await?;

        log::info!(
            "Energy sidecar for {} started on port {} (monitoring PID {})",
            engine.display_name(),
            port,
            wrapped_pid
        );

        let task_handle = tokio::spawn(async move {
            if let Err(e) = axum::serve(listener, app).await {
                log::error!("Energy sidecar error: {}", e);
            }
        });

        Ok(Self {
            port,
            task_handle,
            _monitor_handle: monitor_handle,
        })
    }

    /// Stop the sidecar.
    pub fn stop(self) {
        self.task_handle.abort();
    }
}

async fn energy_handler(State(state): State<SidecarState>) -> impl IntoResponse {
    let snap = state.snapshot.read().unwrap().clone();

    // Accumulate accelerator energy if manager is present
    if let Some(ref mgr) = state.accelerator_manager {
        let now = Instant::now();
        let mut last = state.last_accumulate.write().unwrap();
        if let Some(prev) = *last {
            let elapsed = now.duration_since(prev).as_secs_f64();
            mgr.accumulate_energy(
                elapsed,
                snap.gpu_utilization,
                snap.npu_utilization,
                snap.tpu_utilization,
                snap.lpu_utilization,
            );
        }
        *last = Some(now);
    }

    Json(EnergyResponse {
        engine: state.engine_name.clone(),
        power_watts: snap.power_watts,
        cumulative_joules: snap.cumulative_joules,
        cpu_utilization: snap.cpu_utilization,
        thermal_state: snap.thermal_state.to_string(),
        memory_pressure: snap.memory_pressure,
        memory_used_bytes: snap.memory_used_bytes,
        memory_total_bytes: snap.memory_total_bytes,
        gpu_available: snap.gpu_available,
        gpu_utilization: snap.gpu_utilization,
        npu_available: snap.npu_available,
        npu_utilization: snap.npu_utilization,
        tpu_available: snap.tpu_available,
        tpu_utilization: snap.tpu_utilization,
        lpu_available: snap.lpu_available,
        lpu_utilization: snap.lpu_utilization,
        battery_percent: snap.battery_percent,
        battery_charging: snap.battery_charging,
        swap_used_bytes: snap.swap_used_bytes,
    })
}

/// Per-device accelerator energy endpoint.
async fn devices_handler(State(state): State<SidecarState>) -> impl IntoResponse {
    match &state.accelerator_manager {
        Some(mgr) => {
            let snap = state.snapshot.read().unwrap().clone();
            let devices = mgr.energy_snapshots(
                snap.gpu_utilization,
                snap.npu_utilization,
                snap.tpu_utilization,
                snap.lpu_utilization,
            );
            Json(serde_json::json!({
                "engine": state.engine_name,
                "device_count": devices.len(),
                "devices": devices,
            }))
        }
        None => Json(serde_json::json!({
            "engine": state.engine_name,
            "device_count": 0,
            "devices": [],
        })),
    }
}

async fn metrics_handler(State(state): State<SidecarState>) -> impl IntoResponse {
    let snap = state.snapshot.read().unwrap().clone();
    let platform = detect_platform();
    let engine = &state.engine_name;

    let mut out = String::with_capacity(2048);

    // Power
    out.push_str(&format!(
        "# HELP joule_power_watts Current power consumption in watts\n\
         # TYPE joule_power_watts gauge\n\
         joule_power_watts{{engine=\"{}\"}} {:.2}\n",
        engine, snap.power_watts
    ));

    // Cumulative energy
    out.push_str(&format!(
        "# HELP joule_energy_joules Total energy consumed since start\n\
         # TYPE joule_energy_joules counter\n\
         joule_energy_joules{{engine=\"{}\"}} {:.2}\n",
        engine, snap.cumulative_joules
    ));

    // CPU utilization
    out.push_str(&format!(
        "# HELP joule_cpu_utilization CPU utilization ratio (0.0-1.0)\n\
         # TYPE joule_cpu_utilization gauge\n\
         joule_cpu_utilization{{engine=\"{}\"}} {:.4}\n",
        engine, snap.cpu_utilization
    ));

    // Memory
    out.push_str(&format!(
        "# HELP joule_memory_used_bytes Memory used in bytes\n\
         # TYPE joule_memory_used_bytes gauge\n\
         joule_memory_used_bytes{{engine=\"{}\"}} {}\n",
        engine, snap.memory_used_bytes
    ));

    // GPU
    if snap.gpu_available {
        out.push_str(&format!(
            "# HELP joule_gpu_utilization GPU utilization ratio\n\
             # TYPE joule_gpu_utilization gauge\n\
             joule_gpu_utilization{{engine=\"{}\"}} {:.4}\n",
            engine, snap.gpu_utilization
        ));
    }

    // NPU
    if snap.npu_available {
        out.push_str(&format!(
            "# HELP joule_npu_utilization NPU/Neural Engine utilization ratio\n\
             # TYPE joule_npu_utilization gauge\n\
             joule_npu_utilization{{engine=\"{}\"}} {:.4}\n",
            engine, snap.npu_utilization
        ));
    }

    // Platform info
    out.push_str(&format!(
        "# HELP joule_platform_tdp_watts Estimated platform TDP\n\
         # TYPE joule_platform_tdp_watts gauge\n\
         joule_platform_tdp_watts{{cpu=\"{}\"}} {:.1}\n",
        platform.cpu_brand, platform.tdp_watts
    ));

    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/plain; version=0.0.4; charset=utf-8",
        )],
        out,
    )
}

async fn health_handler(State(state): State<SidecarState>) -> impl IntoResponse {
    let alive = crate::native::is_process_alive(state.wrapped_pid);

    let status = if alive { "healthy" } else { "unhealthy" };
    let code = if alive {
        axum::http::StatusCode::OK
    } else {
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    };

    (
        code,
        Json(serde_json::json!({
            "status": status,
            "engine": state.engine_name,
            "pid": state.wrapped_pid,
            "sidecar_port": state.port,
        })),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_response_serialization() {
        let resp = EnergyResponse {
            engine: "PostgreSQL".into(),
            power_watts: 42.5,
            cumulative_joules: 1000.0,
            cpu_utilization: 0.65,
            thermal_state: "Nominal".into(),
            memory_pressure: 0.3,
            memory_used_bytes: 8_000_000_000,
            memory_total_bytes: 16_000_000_000,
            gpu_available: true,
            gpu_utilization: 0.1,
            npu_available: true,
            npu_utilization: 0.0,
            tpu_available: false,
            tpu_utilization: 0.0,
            lpu_available: false,
            lpu_utilization: 0.0,
            battery_percent: Some(85.0),
            battery_charging: false,
            swap_used_bytes: 0,
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("\"engine\":\"PostgreSQL\""));
        assert!(json.contains("\"power_watts\":42.5"));
        assert!(json.contains("\"gpu_available\":true"));
    }

    #[tokio::test]
    async fn test_sidecar_binds_and_responds() {
        // Start sidecar on random available port
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);

        let sidecar = EnergySidecar::start(
            DatabaseEngine::Postgres,
            std::process::id(), // Use our own PID — guaranteed alive
            port,
            5432,
        )
        .await
        .unwrap();

        // Give the server a moment to start
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        // Test /energy endpoint
        let resp = reqwest::get(format!("http://127.0.0.1:{}/energy", port))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["engine"], "PostgreSQL");
        assert!(body["power_watts"].is_number());

        // Test /metrics endpoint
        let resp = reqwest::get(format!("http://127.0.0.1:{}/metrics", port))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let text = resp.text().await.unwrap();
        assert!(text.contains("joule_power_watts{engine=\"PostgreSQL\"}"));

        // Test /health endpoint
        let resp = reqwest::get(format!("http://127.0.0.1:{}/health", port))
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(body["status"], "healthy");

        sidecar.stop();
    }
}
