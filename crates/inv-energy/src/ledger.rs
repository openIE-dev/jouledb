//! Unified energy ledger — every operational layer deposits energy here.
//!
//! The ledger is the single source of truth for energy consumption across
//! the entire mesh node. It collects measurements from 15 operational
//! layers and produces:
//!
//! - Real-time snapshots for JWP frame headers (cumulative µWh)
//! - Per-layer breakdowns for billing and cost attribution
//! - Carbon accounting (energy × grid intensity → gCO2e)
//! - Audit-ready receipts with verification hashes
//!
//! # Usage
//!
//! ```ignore
//! let ledger = EnergyLedger::new();
//! ledger.record(OperationalLayer::TlsHandshake, 150); // 150 µJ
//! ledger.record(OperationalLayer::Authentication, 50);
//! ledger.record(OperationalLayer::CommandDispatch, 10);
//!
//! let snap = ledger.snapshot();
//! assert_eq!(snap.total_uj, 210);
//! assert_eq!(snap.cumulative_uwh(), 0); // 210 µJ < 3600 µJ/µWh
//! ```

use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};

// ── Operational Layers ─────────────────────────────────────────────

/// Every operational layer that consumes energy on a mesh node.
///
/// These map 1:1 to the infrastructure's execution path. Every µJ
/// consumed by the node is attributed to exactly one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperationalLayer {
    // ── Wire / Protocol ────────────────────────────────────────
    /// TLS handshake (ECDHE key exchange, cert validation).
    TlsHandshake,
    /// JWP frame encode/decode (header + CBOR serialization).
    JwpFraming,
    /// Authentication (challenge-response, HMAC, signature verify).
    Authentication,

    // ── Compute ────────────────────────────────────────────────
    /// WASM module execution (fuel-metered).
    WasmExecution,
    /// Container/VM lifecycle (start, stop, checkpoint).
    ContainerRuntime,
    /// Native binary process dispatch.
    ProcessDispatch,

    // ── AI / ML ────────────────────────────────────────────────
    /// Cortex AI kernel decisions (oracle cascade).
    CortexDecision,
    /// LLM inference (token generation).
    Inference,

    // ── Storage ────────────────────────────────────────────────
    /// Block/object storage read.
    StorageRead,
    /// Block/object storage write.
    StorageWrite,
    /// KV store operations.
    KvOperation,

    // ── Network ────────────────────────────────────────────────
    /// Mesh gossip protocol (peer state sync).
    MeshGossip,
    /// Heartbeat / keepalive.
    MeshHeartbeat,
    /// Node discovery (mDNS, bootstrap).
    MeshDiscovery,
    /// Bulk data transfer between nodes.
    DataTransfer,

    // ── Scheduling ─────────────────────────────────────────────
    /// Orchestrator evaluation cycle.
    SchedulerCycle,
    /// Migration: checkpoint + transfer + restore.
    Migration,

    // ── Consensus ──────────────────────────────────────────────
    /// Raft consensus round.
    ConsensusRound,
    /// CRDT merge / sync.
    CrdtSync,

    // ── Crypto ─────────────────────────────────────────────────
    /// Hashing (SHA-256, BLAKE3).
    HashComputation,
    /// Digital signature creation or verification.
    SignatureOperation,
    /// Symmetric encryption/decryption (AES, ChaCha20).
    Encryption,

    // ── Observability ──────────────────────────────────────────
    /// Telemetry collection and export.
    Telemetry,
    /// Distributed tracing span processing.
    Tracing,

    // ── Command dispatch ───────────────────────────────────────
    /// JWP command decode + routing + response encode.
    CommandDispatch,
}

impl OperationalLayer {
    /// All layer variants, for iteration.
    pub const ALL: &[OperationalLayer] = &[
        Self::TlsHandshake,
        Self::JwpFraming,
        Self::Authentication,
        Self::WasmExecution,
        Self::ContainerRuntime,
        Self::ProcessDispatch,
        Self::CortexDecision,
        Self::Inference,
        Self::StorageRead,
        Self::StorageWrite,
        Self::KvOperation,
        Self::MeshGossip,
        Self::MeshHeartbeat,
        Self::MeshDiscovery,
        Self::DataTransfer,
        Self::SchedulerCycle,
        Self::Migration,
        Self::ConsensusRound,
        Self::CrdtSync,
        Self::HashComputation,
        Self::SignatureOperation,
        Self::Encryption,
        Self::Telemetry,
        Self::Tracing,
        Self::CommandDispatch,
    ];

    /// Index into the fixed-size array.
    const fn index(self) -> usize {
        match self {
            Self::TlsHandshake => 0,
            Self::JwpFraming => 1,
            Self::Authentication => 2,
            Self::WasmExecution => 3,
            Self::ContainerRuntime => 4,
            Self::ProcessDispatch => 5,
            Self::CortexDecision => 6,
            Self::Inference => 7,
            Self::StorageRead => 8,
            Self::StorageWrite => 9,
            Self::KvOperation => 10,
            Self::MeshGossip => 11,
            Self::MeshHeartbeat => 12,
            Self::MeshDiscovery => 13,
            Self::DataTransfer => 14,
            Self::SchedulerCycle => 15,
            Self::Migration => 16,
            Self::ConsensusRound => 17,
            Self::CrdtSync => 18,
            Self::HashComputation => 19,
            Self::SignatureOperation => 20,
            Self::Encryption => 21,
            Self::Telemetry => 22,
            Self::Tracing => 23,
            Self::CommandDispatch => 24,
        }
    }

    /// Human-readable category for grouping in dashboards.
    pub fn category(&self) -> &'static str {
        match self {
            Self::TlsHandshake | Self::JwpFraming | Self::Authentication => "protocol",
            Self::WasmExecution | Self::ContainerRuntime | Self::ProcessDispatch => "compute",
            Self::CortexDecision | Self::Inference => "ai",
            Self::StorageRead | Self::StorageWrite | Self::KvOperation => "storage",
            Self::MeshGossip | Self::MeshHeartbeat | Self::MeshDiscovery | Self::DataTransfer => {
                "network"
            }
            Self::SchedulerCycle | Self::Migration => "scheduling",
            Self::ConsensusRound | Self::CrdtSync => "consensus",
            Self::HashComputation | Self::SignatureOperation | Self::Encryption => "crypto",
            Self::Telemetry | Self::Tracing => "observability",
            Self::CommandDispatch => "dispatch",
        }
    }
}

/// Number of operational layers (compile-time constant).
const LAYER_COUNT: usize = 25;

// ── Energy Ledger ──────────────────────────────────────────────────

/// Unified energy ledger for a single mesh node.
///
/// Thread-safe, lock-free. Every layer deposits energy via `record()`.
/// The ledger uses a fixed-size array of `AtomicU64` — zero allocations
/// on the hot path.
pub struct EnergyLedger {
    /// Per-layer cumulative energy in microjoules (µJ).
    layers: [AtomicU64; LAYER_COUNT],
    /// Per-layer operation count.
    ops: [AtomicU64; LAYER_COUNT],
    /// Total energy across all layers (µJ). Maintained separately
    /// for O(1) total reads without summing 25 atomics.
    total_uj: AtomicU64,
    /// Total operation count.
    total_ops: AtomicU64,
    /// Carbon intensity (gCO2e per kWh) — updated periodically.
    carbon_gco2_per_kwh: AtomicU64, // stored as f64 bits
}

impl EnergyLedger {
    /// Create a new ledger with all counters at zero.
    pub fn new() -> Self {
        Self {
            layers: std::array::from_fn(|_| AtomicU64::new(0)),
            ops: std::array::from_fn(|_| AtomicU64::new(0)),
            total_uj: AtomicU64::new(0),
            total_ops: AtomicU64::new(0),
            carbon_gco2_per_kwh: AtomicU64::new(0),
        }
    }

    /// Record energy consumed by an operational layer.
    ///
    /// This is the hot path — lock-free, zero-allocation.
    #[inline]
    pub fn record(&self, layer: OperationalLayer, microjoules: u64) {
        let idx = layer.index();
        self.layers[idx].fetch_add(microjoules, Ordering::Relaxed);
        self.ops[idx].fetch_add(1, Ordering::Relaxed);
        self.total_uj.fetch_add(microjoules, Ordering::Relaxed);
        self.total_ops.fetch_add(1, Ordering::Relaxed);
    }

    /// Record with explicit operation count (for batch operations).
    #[inline]
    pub fn record_batch(&self, layer: OperationalLayer, microjoules: u64, op_count: u64) {
        let idx = layer.index();
        self.layers[idx].fetch_add(microjoules, Ordering::Relaxed);
        self.ops[idx].fetch_add(op_count, Ordering::Relaxed);
        self.total_uj.fetch_add(microjoules, Ordering::Relaxed);
        self.total_ops.fetch_add(op_count, Ordering::Relaxed);
    }

    /// Total energy in microjoules across all layers.
    #[inline]
    pub fn total_uj(&self) -> u64 {
        self.total_uj.load(Ordering::Relaxed)
    }

    /// Total energy in microwatt-hours (for JWP frame headers).
    #[inline]
    pub fn total_uwh(&self) -> u64 {
        self.total_uj.load(Ordering::Relaxed) / 3600
    }

    /// Total energy in joules.
    #[inline]
    pub fn total_joules(&self) -> f64 {
        self.total_uj.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }

    /// Total operation count.
    #[inline]
    pub fn total_ops(&self) -> u64 {
        self.total_ops.load(Ordering::Relaxed)
    }

    /// Energy consumed by a single layer (µJ).
    #[inline]
    pub fn layer_uj(&self, layer: OperationalLayer) -> u64 {
        self.layers[layer.index()].load(Ordering::Relaxed)
    }

    /// Operation count for a single layer.
    #[inline]
    pub fn layer_ops(&self, layer: OperationalLayer) -> u64 {
        self.ops[layer.index()].load(Ordering::Relaxed)
    }

    /// Update the carbon intensity for carbon accounting.
    pub fn set_carbon_intensity(&self, gco2_per_kwh: f64) {
        let bits = gco2_per_kwh.to_bits();
        self.carbon_gco2_per_kwh.store(bits, Ordering::Relaxed);
    }

    /// Current carbon intensity.
    pub fn carbon_intensity_gco2_per_kwh(&self) -> f64 {
        let bits = self.carbon_gco2_per_kwh.load(Ordering::Relaxed);
        f64::from_bits(bits)
    }

    /// Total carbon emissions (gCO2e) based on total energy and current intensity.
    ///
    /// Formula: gCO2e = energy_kWh × gCO2/kWh
    /// where energy_kWh = energy_µJ / (1e6 × 3600 × 1000) = energy_µJ / 3.6e12
    pub fn total_carbon_gco2e(&self) -> f64 {
        let energy_uj = self.total_uj() as f64;
        let intensity = self.carbon_intensity_gco2_per_kwh();
        energy_uj * intensity / 3_600_000_000_000.0
    }

    /// Take a full snapshot of all layers.
    pub fn snapshot(&self) -> LedgerSnapshot {
        let mut layers = Vec::with_capacity(LAYER_COUNT);
        for &layer in OperationalLayer::ALL {
            let idx = layer.index();
            let energy_uj = self.layers[idx].load(Ordering::Relaxed);
            let ops = self.ops[idx].load(Ordering::Relaxed);
            if energy_uj > 0 || ops > 0 {
                layers.push(LayerSnapshot {
                    layer,
                    category: layer.category().to_string(),
                    energy_uj,
                    ops,
                });
            }
        }

        let total_uj = self.total_uj();
        LedgerSnapshot {
            total_uj,
            total_uwh: total_uj / 3600,
            total_joules: total_uj as f64 / 1_000_000.0,
            total_ops: self.total_ops(),
            carbon_gco2_per_kwh: self.carbon_intensity_gco2_per_kwh(),
            total_carbon_gco2e: self.total_carbon_gco2e(),
            layers,
            timestamp_ns: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64,
        }
    }

    /// Snapshot grouped by category (protocol, compute, ai, etc.).
    pub fn category_breakdown(&self) -> Vec<CategoryBreakdown> {
        let snapshot = self.snapshot();
        let total = snapshot.total_uj.max(1) as f64;

        let mut categories: std::collections::BTreeMap<String, (u64, u64)> =
            std::collections::BTreeMap::new();
        for ls in &snapshot.layers {
            let entry = categories.entry(ls.category.clone()).or_insert((0, 0));
            entry.0 += ls.energy_uj;
            entry.1 += ls.ops;
        }

        categories
            .into_iter()
            .map(|(cat, (energy_uj, ops))| CategoryBreakdown {
                category: cat.to_string(),
                energy_uj,
                ops,
                pct: (energy_uj as f64 / total) * 100.0,
            })
            .collect()
    }

    /// Reset all counters to zero (for testing or epoch boundaries).
    pub fn reset(&self) {
        for i in 0..LAYER_COUNT {
            self.layers[i].store(0, Ordering::Relaxed);
            self.ops[i].store(0, Ordering::Relaxed);
        }
        self.total_uj.store(0, Ordering::Relaxed);
        self.total_ops.store(0, Ordering::Relaxed);
    }
}

impl Default for EnergyLedger {
    fn default() -> Self {
        Self::new()
    }
}

// ── Snapshot Types ─────────────────────────────────────────────────

/// Point-in-time snapshot of the entire energy ledger.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerSnapshot {
    /// Total energy in microjoules.
    pub total_uj: u64,
    /// Total energy in microwatt-hours (for JWP headers).
    pub total_uwh: u64,
    /// Total energy in joules (for billing/display).
    pub total_joules: f64,
    /// Total operation count across all layers.
    pub total_ops: u64,
    /// Current carbon intensity (gCO2e/kWh).
    pub carbon_gco2_per_kwh: f64,
    /// Total carbon emissions (gCO2e).
    pub total_carbon_gco2e: f64,
    /// Per-layer breakdown (only layers with activity).
    pub layers: Vec<LayerSnapshot>,
    /// Timestamp when snapshot was taken (Unix nanos).
    pub timestamp_ns: u64,
}

impl LedgerSnapshot {
    /// Cumulative µWh for JWP frame headers.
    pub fn cumulative_uwh(&self) -> u64 {
        self.total_uwh
    }

    /// Format as a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "{:.3} J | {} µWh | {:.6} gCO₂e | {} ops across {} layers",
            self.total_joules,
            self.total_uwh,
            self.total_carbon_gco2e,
            self.total_ops,
            self.layers.len()
        )
    }
}

/// Snapshot of a single operational layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSnapshot {
    /// Which layer.
    pub layer: OperationalLayer,
    /// Category grouping.
    pub category: String,
    /// Cumulative energy in µJ.
    pub energy_uj: u64,
    /// Operation count.
    pub ops: u64,
}

impl LayerSnapshot {
    /// Average energy per operation in µJ.
    pub fn avg_uj_per_op(&self) -> f64 {
        if self.ops == 0 {
            0.0
        } else {
            self.energy_uj as f64 / self.ops as f64
        }
    }
}

/// Category-level aggregation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoryBreakdown {
    /// Category name (protocol, compute, ai, etc.).
    pub category: String,
    /// Total energy in µJ.
    pub energy_uj: u64,
    /// Total operations.
    pub ops: u64,
    /// Percentage of total energy.
    pub pct: f64,
}

// ── Energy Cost Constants ──────────────────────────────────────────
// Empirical estimates for operations that can't be hardware-metered
// directly. These are calibrated against RAPL/Apple Silicon measurements.

/// Estimated energy costs per operation (microjoules).
///
/// These are conservative upper-bound estimates used when hardware
/// metering isn't available. Calibrated against:
/// - Intel RAPL on Xeon (inv-seed-ash)
/// - Apple Silicon powermetrics (dev Mac)
pub mod costs {
    // Protocol layer
    /// TLS 1.3 ECDHE-P256 handshake: ~300 µJ (measured via RAPL).
    pub const TLS_HANDSHAKE_UJ: u64 = 300;
    /// JWP frame encode/decode (21-byte header + CBOR): ~5 µJ.
    pub const JWP_FRAME_UJ: u64 = 5;
    /// HMAC-SHA256 auth challenge-response: ~15 µJ.
    pub const AUTH_HMAC_UJ: u64 = 15;
    /// ECDSA P-256 signature verify: ~50 µJ.
    pub const AUTH_ECDSA_VERIFY_UJ: u64 = 50;

    // Compute layer
    /// WASM fuel unit → µJ conversion factor.
    /// 1 fuel unit ≈ one simple instruction ≈ 0.1 µJ on modern silicon.
    pub const WASM_FUEL_TO_UJ: f64 = 0.1;

    // Storage layer
    /// 4 KiB block read from NVMe: ~2 µJ.
    pub const BLOCK_READ_4K_UJ: u64 = 2;
    /// 4 KiB block write to NVMe: ~8 µJ.
    pub const BLOCK_WRITE_4K_UJ: u64 = 8;
    /// KV store get/put (in-memory DashMap): ~1 µJ.
    pub const KV_OP_UJ: u64 = 1;

    // Network layer
    /// Gossip pull round (serialize + send + receive): ~20 µJ.
    pub const GOSSIP_ROUND_UJ: u64 = 20;
    /// Heartbeat (tiny frame): ~3 µJ.
    pub const HEARTBEAT_UJ: u64 = 3;
    /// Per-byte network transfer: ~0.5 nJ/byte = 0.0005 µJ/byte.
    /// For bulk: use `transfer_uj(bytes)`.
    pub const TRANSFER_NJ_PER_BYTE: f64 = 0.5;

    // Scheduling
    /// One orchestrator evaluation cycle: ~100 µJ.
    pub const SCHEDULER_CYCLE_UJ: u64 = 100;
    /// Migration checkpoint + transfer: ~10,000 µJ (depends on size).
    pub const MIGRATION_BASE_UJ: u64 = 10_000;

    // Consensus
    /// Single Raft round (log append + vote): ~30 µJ.
    pub const RAFT_ROUND_UJ: u64 = 30;
    /// CRDT merge: ~10 µJ.
    pub const CRDT_MERGE_UJ: u64 = 10;

    // Crypto
    /// SHA-256 hash of 1 KiB: ~3 µJ.
    pub const SHA256_1K_UJ: u64 = 3;
    /// BLAKE3 hash of 1 KiB: ~1 µJ (faster than SHA-256).
    pub const BLAKE3_1K_UJ: u64 = 1;
    /// AES-256-GCM encrypt 1 KiB: ~2 µJ (with AES-NI).
    pub const AES_ENCRYPT_1K_UJ: u64 = 2;

    // Observability
    /// Emit one telemetry event: ~5 µJ.
    pub const TELEMETRY_EVENT_UJ: u64 = 5;
    /// Process one tracing span: ~3 µJ.
    pub const TRACING_SPAN_UJ: u64 = 3;

    // Command dispatch
    /// Route + decode + encode one JWP command: ~10 µJ.
    pub const COMMAND_DISPATCH_UJ: u64 = 10;

    /// Energy cost of bulk data transfer.
    pub fn transfer_uj(bytes: u64) -> u64 {
        ((bytes as f64) * TRANSFER_NJ_PER_BYTE / 1000.0) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_read() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::TlsHandshake, 300);
        ledger.record(OperationalLayer::Authentication, 15);
        ledger.record(OperationalLayer::JwpFraming, 5);

        assert_eq!(ledger.total_uj(), 320);
        assert_eq!(ledger.total_ops(), 3);
        assert_eq!(ledger.layer_uj(OperationalLayer::TlsHandshake), 300);
        assert_eq!(ledger.layer_ops(OperationalLayer::TlsHandshake), 1);
    }

    #[test]
    fn uwh_conversion() {
        let ledger = EnergyLedger::new();
        // 7200 µJ = 2 µWh
        ledger.record(OperationalLayer::WasmExecution, 7200);
        assert_eq!(ledger.total_uwh(), 2);
    }

    #[test]
    fn joules_conversion() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::Inference, 1_000_000); // 1 J
        assert!((ledger.total_joules() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn carbon_accounting() {
        let ledger = EnergyLedger::new();
        // 1 kWh = 3.6e12 µJ, at 400 gCO2/kWh
        ledger.set_carbon_intensity(400.0);
        ledger.record(OperationalLayer::WasmExecution, 3_600_000_000_000); // 1 kWh
        assert!((ledger.total_carbon_gco2e() - 400.0).abs() < 0.01);
    }

    #[test]
    fn carbon_small_workload() {
        let ledger = EnergyLedger::new();
        ledger.set_carbon_intensity(233.0); // global average
        // 1 J = 1e6 µJ
        ledger.record(OperationalLayer::CommandDispatch, 1_000_000);
        // 1 J = 1/3.6e6 kWh ≈ 2.78e-7 kWh
        // carbon = 2.78e-7 × 233 ≈ 6.47e-5 gCO2e
        let carbon = ledger.total_carbon_gco2e();
        assert!(carbon > 6e-5 && carbon < 7e-5);
    }

    #[test]
    fn snapshot_only_active_layers() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::TlsHandshake, 100);
        ledger.record(OperationalLayer::MeshHeartbeat, 3);

        let snap = ledger.snapshot();
        assert_eq!(snap.layers.len(), 2); // only 2 active
        assert_eq!(snap.total_uj, 103);
        assert_eq!(snap.total_ops, 2);
    }

    #[test]
    fn category_breakdown() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::TlsHandshake, 300);
        ledger.record(OperationalLayer::JwpFraming, 5);
        ledger.record(OperationalLayer::Authentication, 15);
        ledger.record(OperationalLayer::WasmExecution, 1000);

        let cats = ledger.category_breakdown();
        let protocol = cats.iter().find(|c| c.category == "protocol").unwrap();
        assert_eq!(protocol.energy_uj, 320); // 300 + 5 + 15
        assert_eq!(protocol.ops, 3);

        let compute = cats.iter().find(|c| c.category == "compute").unwrap();
        assert_eq!(compute.energy_uj, 1000);
    }

    #[test]
    fn batch_recording() {
        let ledger = EnergyLedger::new();
        ledger.record_batch(OperationalLayer::DataTransfer, 5000, 100);

        assert_eq!(ledger.layer_uj(OperationalLayer::DataTransfer), 5000);
        assert_eq!(ledger.layer_ops(OperationalLayer::DataTransfer), 100);
        assert_eq!(ledger.total_ops(), 100);
    }

    #[test]
    fn reset_clears_all() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::Inference, 999);
        assert_eq!(ledger.total_uj(), 999);

        ledger.reset();
        assert_eq!(ledger.total_uj(), 0);
        assert_eq!(ledger.total_ops(), 0);
        assert_eq!(ledger.layer_uj(OperationalLayer::Inference), 0);
    }

    #[test]
    fn all_layers_enumerated() {
        assert_eq!(OperationalLayer::ALL.len(), LAYER_COUNT);
        // Verify indices are unique and in range
        for (i, layer) in OperationalLayer::ALL.iter().enumerate() {
            assert_eq!(layer.index(), i);
        }
    }

    #[test]
    fn layer_categories() {
        assert_eq!(OperationalLayer::TlsHandshake.category(), "protocol");
        assert_eq!(OperationalLayer::WasmExecution.category(), "compute");
        assert_eq!(OperationalLayer::CortexDecision.category(), "ai");
        assert_eq!(OperationalLayer::StorageRead.category(), "storage");
        assert_eq!(OperationalLayer::MeshGossip.category(), "network");
        assert_eq!(OperationalLayer::SchedulerCycle.category(), "scheduling");
        assert_eq!(OperationalLayer::ConsensusRound.category(), "consensus");
        assert_eq!(OperationalLayer::HashComputation.category(), "crypto");
        assert_eq!(OperationalLayer::Telemetry.category(), "observability");
        assert_eq!(OperationalLayer::CommandDispatch.category(), "dispatch");
    }

    #[test]
    fn snapshot_summary_format() {
        let ledger = EnergyLedger::new();
        ledger.set_carbon_intensity(400.0);
        ledger.record(OperationalLayer::TlsHandshake, 1_000_000);
        let snap = ledger.snapshot();
        let summary = snap.summary();
        assert!(summary.contains("1.000 J"));
        assert!(summary.contains("gCO₂e"));
    }

    #[test]
    fn avg_uj_per_op() {
        let ledger = EnergyLedger::new();
        ledger.record(OperationalLayer::CommandDispatch, 100);
        ledger.record(OperationalLayer::CommandDispatch, 200);

        let snap = ledger.snapshot();
        let cmd = snap
            .layers
            .iter()
            .find(|l| l.layer == OperationalLayer::CommandDispatch)
            .unwrap();
        assert_eq!(cmd.ops, 2);
        assert!((cmd.avg_uj_per_op() - 150.0).abs() < 1e-10);
    }

    #[test]
    fn concurrent_recording() {
        use std::sync::Arc;
        let ledger = Arc::new(EnergyLedger::new());
        let mut handles = vec![];

        for _ in 0..10 {
            let l = ledger.clone();
            handles.push(std::thread::spawn(move || {
                for _ in 0..1000 {
                    l.record(OperationalLayer::JwpFraming, 5);
                }
            }));
        }

        for h in handles {
            h.join().unwrap();
        }

        assert_eq!(ledger.total_uj(), 50_000); // 10 threads × 1000 × 5
        assert_eq!(ledger.total_ops(), 10_000);
        assert_eq!(ledger.layer_uj(OperationalLayer::JwpFraming), 50_000);
    }

    #[test]
    fn cost_constants_reasonable() {
        // TLS handshake should be more expensive than a heartbeat
        assert!(costs::TLS_HANDSHAKE_UJ > costs::HEARTBEAT_UJ);
        // ECDSA verify more expensive than HMAC
        assert!(costs::AUTH_ECDSA_VERIFY_UJ > costs::AUTH_HMAC_UJ);
        // Write more expensive than read
        assert!(costs::BLOCK_WRITE_4K_UJ > costs::BLOCK_READ_4K_UJ);
        // BLAKE3 faster than SHA-256
        assert!(costs::BLAKE3_1K_UJ < costs::SHA256_1K_UJ);
    }

    #[test]
    fn transfer_cost_calculation() {
        // 1 MB = 1_000_000 bytes at 0.5 nJ/byte = 500 µJ
        let cost = costs::transfer_uj(1_000_000);
        assert_eq!(cost, 500);
    }
}
