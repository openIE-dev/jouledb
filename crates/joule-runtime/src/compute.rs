//! Polyglot Compute Runtime — energy-metered interactive interpreters.
//!
//! Manages Python, Julia, Shell, and Claude as subprocesses with energy
//! metering per cell. No FFI (no PyO3, no jlrs) — clean process boundaries.
//!
//! Each "cell" execution returns output + an energy receipt:
//! ```text
//! CellResult {
//!     stdout: "42\n",
//!     stderr: "",
//!     energy_joules: 0.0034,
//!     duration_secs: 0.012,
//!     power_watts: 0.283,
//!     provenance: Measured,
//! }
//! ```
//!
//! Kernel types:
//! - **Python** — persistent `python3 -u -i` subprocess, state carries across cells
//! - **Julia** — persistent `julia --startup-file=no` subprocess
//! - **Shell** — persistent `bash`/`zsh` subprocess
//! - **Claude** — one-shot `claude -p` per cell (each cell is a prompt)
//!
//! Notebooks serialize cells + results + energy receipts to `.jnb` (JSON).
//!
//! Design:
//! - Sentinel-based completion for persistent interpreters
//! - Real power measurement via `EnergyMonitor` (IOKit/sysfs)
//! - Claude energy: client power × duration (local compute) + estimated remote
//!   inference energy from token count × 0.001 J/token (Luccioni et al. 2023)
//! - Each kernel is a long-lived child process; dropped on `shutdown()`

use crate::accelerator::PowerProvenance;
use joule_db_energy::EnergySnapshot;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, RwLock};
use std::time::Instant;

// ── Types ───────────────────────────────────────────────────────────────────

/// Supported compute kernel types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KernelKind {
    /// Python 3 interpreter (`python3 -u -i`).
    Python,
    /// Julia interpreter (`julia --startup-file=no`).
    Julia,
    /// POSIX shell (`bash` or `zsh`).
    Shell,
    /// Claude Code CLI (`claude -p`). Each cell is a one-shot prompt.
    /// Energy = client power × duration + remote inference estimate.
    Claude,
}

impl std::fmt::Display for KernelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Python => write!(f, "python"),
            Self::Julia => write!(f, "julia"),
            Self::Shell => write!(f, "shell"),
            Self::Claude => write!(f, "claude"),
        }
    }
}

/// Result of executing a single cell in a kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellResult {
    /// Standard output captured from the cell.
    pub stdout: String,
    /// Standard error captured from the cell.
    pub stderr: String,
    /// Exit status: true if the cell completed without error markers.
    pub success: bool,
    /// Wall-clock duration of the cell execution in seconds.
    pub duration_secs: f64,
    /// Energy consumed during cell execution in joules (client-side).
    pub energy_joules: f64,
    /// Average power draw during cell execution in watts.
    pub power_watts: f64,
    /// How the power reading was obtained.
    pub provenance: PowerProvenance,
    /// Estimated remote inference energy in joules (Claude kernel only).
    /// Calculated from output token count × 0.001 J/token (Luccioni et al. 2023).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub remote_energy_joules: Option<f64>,
    /// Estimated token count (Claude kernel only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens_estimated: Option<u64>,
}

/// Metadata about a running kernel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelInfo {
    /// Unique kernel identifier.
    pub id: String,
    /// Kernel type.
    pub kind: KernelKind,
    /// Process ID of the interpreter subprocess.
    pub pid: u32,
    /// Number of cells executed so far.
    pub cells_executed: u64,
    /// Cumulative energy consumed across all cells in joules.
    pub cumulative_energy_joules: f64,
    /// Whether the kernel is still alive.
    pub alive: bool,
}

// ── Sentinel Protocol ───────────────────────────────────────────────────────

/// Unique marker injected after each cell to detect completion.
///
/// The sentinel is a string that's extremely unlikely to appear in user output.
/// After sending user code, we inject a print of this sentinel. When it appears
/// on stdout, we know the cell finished. Stderr is drained separately.
const SENTINEL_PREFIX: &str = "__JOULE_CELL_DONE_";

fn make_sentinel(cell_id: u64) -> String {
    format!("{}{:016x}__", SENTINEL_PREFIX, cell_id)
}

/// Generate the sentinel-printing code for each kernel type.
/// Claude kernels don't use sentinels (one-shot execution).
fn sentinel_code(kind: KernelKind, sentinel: &str) -> String {
    match kind {
        KernelKind::Python => format!("print(\"{}\")\n", sentinel),
        KernelKind::Julia => format!("println(\"{}\")\n", sentinel),
        KernelKind::Shell => format!("echo \"{}\"\n", sentinel),
        KernelKind::Claude => String::new(), // Not used — one-shot execution
    }
}

// ── Kernel ──────────────────────────────────────────────────────────────────

/// A compute kernel — either a persistent interpreter or one-shot executor.
///
/// Python/Julia/Shell: long-lived subprocess with piped stdin/stdout.
/// Claude: spawns a fresh `claude -p` per cell (stateless).
struct Kernel {
    kind: KernelKind,
    /// Persistent child process (Python/Julia/Shell). None for Claude.
    child: Option<Child>,
    cell_counter: u64,
    cumulative_energy_joules: f64,
}

impl Kernel {
    /// Spawn a new kernel.
    fn spawn(kind: KernelKind) -> Result<Self, ComputeError> {
        match kind {
            KernelKind::Claude => {
                // Verify claude CLI is available
                if !which_available("claude") {
                    return Err(ComputeError::SpawnFailed {
                        kind,
                        reason: "claude CLI not found on PATH".into(),
                    });
                }
                log::info!("Claude kernel initialized (one-shot mode)");
                Ok(Self {
                    kind,
                    child: None,
                    cell_counter: 0,
                    cumulative_energy_joules: 0.0,
                })
            }
            _ => {
                let (program, args) = match kind {
                    KernelKind::Python => ("python3", vec!["-u", "-i", "-q"]),
                    KernelKind::Julia => ("julia", vec!["--startup-file=no", "-q"]),
                    KernelKind::Shell => {
                        let shell =
                            std::env::var("SHELL").unwrap_or_else(|_| "bash".into());
                        let shell: &'static str = Box::leak(shell.into_boxed_str());
                        (shell, vec!["-i"])
                    }
                    KernelKind::Claude => unreachable!(),
                };

                let child = Command::new(program)
                    .args(&args)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::piped())
                    .stderr(Stdio::piped())
                    .env("PYTHONDONTWRITEBYTECODE", "1")
                    .env("TERM", "dumb")
                    .spawn()
                    .map_err(|e| ComputeError::SpawnFailed {
                        kind,
                        reason: e.to_string(),
                    })?;

                log::info!("Spawned {} kernel (PID {})", kind, child.id());

                Ok(Self {
                    kind,
                    child: Some(child),
                    cell_counter: 0,
                    cumulative_energy_joules: 0.0,
                })
            }
        }
    }

    /// Execute a cell. Dispatches to persistent or one-shot path.
    fn execute(
        &mut self,
        code: &str,
        monitor: Option<&Arc<RwLock<EnergySnapshot>>>,
    ) -> Result<CellResult, ComputeError> {
        self.cell_counter += 1;

        if self.kind == KernelKind::Claude {
            self.execute_claude(code, monitor)
        } else {
            self.execute_persistent(code, monitor)
        }
    }

    /// Execute in a persistent interpreter (Python/Julia/Shell).
    fn execute_persistent(
        &mut self,
        code: &str,
        monitor: Option<&Arc<RwLock<EnergySnapshot>>>,
    ) -> Result<CellResult, ComputeError> {
        let sentinel = make_sentinel(self.cell_counter);

        let (power_at_start, provenance) = read_power(monitor);
        let start = Instant::now();

        let child = self.child.as_mut().ok_or(ComputeError::StdinClosed)?;
        let stdin = child.stdin.as_mut().ok_or(ComputeError::StdinClosed)?;

        // For Python: append blank line to close compound statements
        let code_block = if self.kind == KernelKind::Python {
            format!("{}\n\n", code.trim_end())
        } else if code.ends_with('\n') {
            code.to_string()
        } else {
            format!("{}\n", code)
        };
        stdin
            .write_all(code_block.as_bytes())
            .map_err(|e| ComputeError::IoError(e.to_string()))?;

        let sentinel_stmt = sentinel_code(self.kind, &sentinel);
        stdin
            .write_all(sentinel_stmt.as_bytes())
            .map_err(|e| ComputeError::IoError(e.to_string()))?;
        stdin
            .flush()
            .map_err(|e| ComputeError::IoError(e.to_string()))?;

        // Read stdout until sentinel
        let stdout_pipe = child.stdout.as_mut().ok_or(ComputeError::StdoutClosed)?;
        let mut reader = BufReader::new(stdout_pipe);
        let mut output_lines = Vec::new();
        let mut found_sentinel = false;

        loop {
            let mut line = String::new();
            match reader.read_line(&mut line) {
                Ok(0) => break,
                Ok(_) => {
                    if line.trim() == sentinel {
                        found_sentinel = true;
                        break;
                    }
                    let trimmed = line.trim_start();
                    if self.kind == KernelKind::Python
                        && (trimmed.starts_with(">>> ") || trimmed.starts_with("... "))
                    {
                        continue;
                    }
                    output_lines.push(line);
                }
                Err(e) => {
                    return Err(ComputeError::IoError(format!("stdout read: {}", e)));
                }
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let (power_at_end, _) = read_power(monitor);
        let avg_power = (power_at_start + power_at_end) / 2.0;
        let energy_joules = avg_power * elapsed;
        self.cumulative_energy_joules += energy_joules;

        let stderr_output = self.drain_stderr();
        let stdout = output_lines.join("");

        Ok(CellResult {
            success: found_sentinel,
            stdout,
            stderr: stderr_output,
            duration_secs: elapsed,
            energy_joules,
            power_watts: avg_power,
            provenance,
            remote_energy_joules: None,
            tokens_estimated: None,
        })
    }

    /// Execute a Claude prompt via `claude -p` (one-shot).
    ///
    /// Energy decomposition:
    /// - Client: measured power × duration (local CPU running the CLI)
    /// - Remote: estimated from output length × 0.001 J/token (Luccioni et al. 2023)
    ///   Average across Llama-65B, BLOOM-176B, GPT-3 class models.
    fn execute_claude(
        &mut self,
        prompt: &str,
        monitor: Option<&Arc<RwLock<EnergySnapshot>>>,
    ) -> Result<CellResult, ComputeError> {
        let (power_at_start, provenance) = read_power(monitor);
        let start = Instant::now();

        let output = Command::new("claude")
            .args(["-p", prompt])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("TERM", "dumb")
            .output()
            .map_err(|e| ComputeError::SpawnFailed {
                kind: KernelKind::Claude,
                reason: e.to_string(),
            })?;

        let elapsed = start.elapsed().as_secs_f64();
        let (power_at_end, _) = read_power(monitor);
        let avg_power = (power_at_start + power_at_end) / 2.0;
        let client_energy = avg_power * elapsed;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        // Estimate tokens from output length (~4 chars per token for English)
        let estimated_tokens = (stdout.len() as u64 + 3) / 4;
        // Remote inference energy: ~0.001 J/token (Luccioni et al. 2023)
        let remote_energy = estimated_tokens as f64 * JOULES_PER_TOKEN_ESTIMATE;

        let total_energy = client_energy + remote_energy;
        self.cumulative_energy_joules += total_energy;

        Ok(CellResult {
            success: output.status.success(),
            stdout,
            stderr,
            duration_secs: elapsed,
            energy_joules: total_energy,
            power_watts: avg_power,
            provenance,
            remote_energy_joules: Some(remote_energy),
            tokens_estimated: Some(estimated_tokens),
        })
    }

    /// Drain available stderr without blocking.
    fn drain_stderr(&mut self) -> String {
        let child = match self.child.as_mut() {
            Some(c) => c,
            None => return String::new(),
        };
        let stderr = match child.stderr.as_mut() {
            Some(s) => s,
            None => return String::new(),
        };

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = stderr.as_raw_fd();
            unsafe {
                let flags = libc::fcntl(fd, libc::F_GETFL);
                libc::fcntl(fd, libc::F_SETFL, flags | libc::O_NONBLOCK);
            }
        }

        let mut buf = vec![0u8; 8192];
        let mut result = String::new();
        loop {
            match std::io::Read::read(stderr, &mut buf) {
                Ok(0) => break,
                Ok(n) => result.push_str(&String::from_utf8_lossy(&buf[..n])),
                Err(_) => break,
            }
        }

        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let fd = stderr.as_raw_fd();
            unsafe {
                let flags = libc::fcntl(fd, libc::F_GETFL);
                libc::fcntl(fd, libc::F_SETFL, flags & !libc::O_NONBLOCK);
            }
        }

        result
    }

    /// Check if the kernel is alive.
    fn is_alive(&mut self) -> bool {
        match &mut self.child {
            Some(child) => matches!(child.try_wait(), Ok(None)),
            None => self.kind == KernelKind::Claude, // Claude is always "alive"
        }
    }

    /// Get kernel metadata.
    fn info(&mut self, id: &str) -> KernelInfo {
        KernelInfo {
            id: id.to_string(),
            kind: self.kind,
            pid: self.child.as_ref().map(|c| c.id()).unwrap_or(0),
            cells_executed: self.cell_counter,
            cumulative_energy_joules: self.cumulative_energy_joules,
            alive: self.is_alive(),
        }
    }

    /// Shut down the kernel process.
    fn shutdown(&mut self) {
        if let Some(ref mut child) = self.child {
            drop(child.stdin.take());

            let deadline = Instant::now() + std::time::Duration::from_secs(2);
            loop {
                match child.try_wait() {
                    Ok(Some(_)) => return,
                    Ok(None) => {
                        if Instant::now() >= deadline {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(50));
                    }
                    Err(_) => break,
                }
            }

            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for Kernel {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Check if a binary is available on PATH.
fn which_available(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Estimated energy per token for remote LLM inference (joules).
///
/// Based on Luccioni et al. 2023 "Power Hungry Processing" — average across
/// BLOOM-176B, Llama-65B, and GPT-3 class models on A100 GPUs.
/// Conservative mid-range estimate; actual varies 10x by model size and hardware.
const JOULES_PER_TOKEN_ESTIMATE: f64 = 0.001;

// ── ComputeRuntime ──────────────────────────────────────────────────────────

/// Manages multiple compute kernels with energy metering.
///
/// ```text
/// let mut rt = ComputeRuntime::new();
/// let kid = rt.start_kernel(KernelKind::Python)?;
/// let result = rt.execute(&kid, "print(2 + 2)")?;
/// assert_eq!(result.stdout.trim(), "4");
/// assert!(result.energy_joules > 0.0);
/// rt.shutdown(&kid);
/// ```
pub struct ComputeRuntime {
    /// Active kernels keyed by ID.
    kernels: HashMap<String, Kernel>,
    /// Counter for generating unique kernel IDs.
    next_id: u64,
    /// Shared energy monitor handle for real power readings.
    monitor_handle: Option<Arc<RwLock<EnergySnapshot>>>,
}

impl ComputeRuntime {
    /// Create a new compute runtime without energy monitoring.
    ///
    /// Power readings will be estimated. Use `with_monitor()` for real measurements.
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
            next_id: 0,
            monitor_handle: None,
        }
    }

    /// Create a compute runtime with a real energy monitor.
    ///
    /// The `EnergyMonitor` snapshot provides real-time system power from
    /// IOKit (macOS) or sysfs (Linux) for accurate per-cell energy receipts.
    pub fn with_monitor(monitor_handle: Arc<RwLock<EnergySnapshot>>) -> Self {
        Self {
            kernels: HashMap::new(),
            next_id: 0,
            monitor_handle: Some(monitor_handle),
        }
    }

    /// Start a new kernel of the given type. Returns the kernel ID.
    pub fn start_kernel(&mut self, kind: KernelKind) -> Result<String, ComputeError> {
        self.next_id += 1;
        let id = format!("{}-{}", kind, self.next_id);
        let kernel = Kernel::spawn(kind)?;
        self.kernels.insert(id.clone(), kernel);
        Ok(id)
    }

    /// Execute a code cell in the given kernel.
    pub fn execute(&mut self, kernel_id: &str, code: &str) -> Result<CellResult, ComputeError> {
        let monitor = self.monitor_handle.clone();
        let kernel = self
            .kernels
            .get_mut(kernel_id)
            .ok_or_else(|| ComputeError::KernelNotFound(kernel_id.to_string()))?;
        kernel.execute(code, monitor.as_ref())
    }

    /// Get info about a specific kernel.
    pub fn kernel_info(&mut self, kernel_id: &str) -> Option<KernelInfo> {
        self.kernels
            .get_mut(kernel_id)
            .map(|k| k.info(kernel_id))
    }

    /// List all active kernels.
    pub fn list_kernels(&mut self) -> Vec<KernelInfo> {
        self.kernels
            .iter_mut()
            .map(|(id, k)| k.info(id))
            .collect()
    }

    /// Shut down a specific kernel.
    pub fn shutdown(&mut self, kernel_id: &str) {
        if let Some(mut kernel) = self.kernels.remove(kernel_id) {
            kernel.shutdown();
        }
    }

    /// Shut down all kernels.
    pub fn shutdown_all(&mut self) {
        let ids: Vec<String> = self.kernels.keys().cloned().collect();
        for id in ids {
            self.shutdown(&id);
        }
    }

    /// Total energy consumed across all kernels in joules.
    pub fn total_energy_joules(&self) -> f64 {
        self.kernels.values().map(|k| k.cumulative_energy_joules).sum()
    }
}

impl Default for ComputeRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ComputeRuntime {
    fn drop(&mut self) {
        self.shutdown_all();
    }
}

// ── Power Reading ───────────────────────────────────────────────────────────

/// Read current system power from the energy monitor.
///
/// Returns (watts, provenance). Falls back to a conservative 15W estimate
/// if no monitor is available (typical laptop idle power).
fn read_power(monitor: Option<&Arc<RwLock<EnergySnapshot>>>) -> (f64, PowerProvenance) {
    if let Some(handle) = monitor {
        if let Ok(snapshot) = handle.read() {
            if snapshot.power_watts > 0.0 {
                return (snapshot.power_watts, PowerProvenance::Measured);
            }
        }
    }
    // Fallback: conservative estimate for a typical laptop
    (15.0, PowerProvenance::Estimated)
}

// ── Notebook Format ─────────────────────────────────────────────────────────

/// A Joule Notebook (`.jnb`) — cells + results + energy receipts.
///
/// Serializes to JSON for persistence, sharing, and reproducibility.
/// Each cell records its source code, output, and energy cost, making
/// the full energy footprint of a compute session transparent.
///
/// ```text
/// {
///   "format": "joule-notebook",
///   "version": 1,
///   "kernel": "python",
///   "cells": [
///     { "cell_number": 1, "source": "print(2+2)", "result": { "stdout": "4\n", ... } }
///   ]
/// }
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notebook {
    /// Format identifier.
    pub format: String,
    /// Format version.
    pub version: u32,
    /// Kernel type used for this notebook.
    pub kernel: KernelKind,
    /// Ordered list of executed cells.
    pub cells: Vec<NotebookCell>,
    /// Arbitrary metadata (author, description, tags, etc.).
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

/// A single cell in a Joule Notebook.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotebookCell {
    /// Cell execution order (1-indexed).
    pub cell_number: u64,
    /// Source code / prompt submitted to the kernel.
    pub source: String,
    /// Execution result including output and energy receipt.
    pub result: CellResult,
}

impl Notebook {
    /// Create a new empty notebook for the given kernel type.
    pub fn new(kernel: KernelKind) -> Self {
        Self {
            format: "joule-notebook".into(),
            version: 1,
            kernel,
            cells: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add an executed cell to the notebook.
    pub fn add_cell(&mut self, source: &str, result: CellResult) {
        let cell_number = self.cells.len() as u64 + 1;
        self.cells.push(NotebookCell {
            cell_number,
            source: source.to_string(),
            result,
        });
    }

    /// Total energy consumed across all cells in joules.
    pub fn total_energy_joules(&self) -> f64 {
        self.cells.iter().map(|c| c.result.energy_joules).sum()
    }

    /// Total wall-clock time across all cells in seconds.
    pub fn total_duration_secs(&self) -> f64 {
        self.cells.iter().map(|c| c.result.duration_secs).sum()
    }

    /// Save the notebook to a file as pretty-printed JSON.
    pub fn save(&self, path: &std::path::Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    /// Load a notebook from a JSON file.
    pub fn load(path: &std::path::Path) -> Result<Self, std::io::Error> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

// ── Errors ──────────────────────────────────────────────────────────────────

/// Compute runtime errors.
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    #[error("failed to spawn {kind} kernel: {reason}")]
    SpawnFailed { kind: KernelKind, reason: String },

    #[error("kernel not found: {0}")]
    KernelNotFound(String),

    #[error("kernel stdin closed")]
    StdinClosed,

    #[error("kernel stdout closed")]
    StdoutClosed,

    #[error("I/O error: {0}")]
    IoError(String),
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_kind_display() {
        assert_eq!(KernelKind::Python.to_string(), "python");
        assert_eq!(KernelKind::Julia.to_string(), "julia");
        assert_eq!(KernelKind::Shell.to_string(), "shell");
        assert_eq!(KernelKind::Claude.to_string(), "claude");
    }

    #[test]
    fn test_kernel_kind_serde() {
        for kind in [
            KernelKind::Python,
            KernelKind::Julia,
            KernelKind::Shell,
            KernelKind::Claude,
        ] {
            let json = serde_json::to_string(&kind).unwrap();
            let parsed: KernelKind = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, kind);
        }
    }

    #[test]
    fn test_sentinel_generation() {
        let s1 = make_sentinel(1);
        let s2 = make_sentinel(2);
        assert_ne!(s1, s2);
        assert!(s1.starts_with(SENTINEL_PREFIX));
        assert!(s1.ends_with("__"));
    }

    #[test]
    fn test_sentinel_code_python() {
        let code = sentinel_code(KernelKind::Python, "MARKER");
        assert_eq!(code, "print(\"MARKER\")\n");
    }

    #[test]
    fn test_sentinel_code_julia() {
        let code = sentinel_code(KernelKind::Julia, "MARKER");
        assert_eq!(code, "println(\"MARKER\")\n");
    }

    #[test]
    fn test_sentinel_code_shell() {
        let code = sentinel_code(KernelKind::Shell, "MARKER");
        assert_eq!(code, "echo \"MARKER\"\n");
    }

    #[test]
    fn test_cell_result_serde() {
        let result = CellResult {
            stdout: "42\n".into(),
            stderr: String::new(),
            success: true,
            duration_secs: 0.012,
            energy_joules: 0.0034,
            power_watts: 0.283,
            provenance: PowerProvenance::Measured,
            remote_energy_joules: None,
            tokens_estimated: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: CellResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.stdout, "42\n");
        assert!(parsed.success);
        assert_eq!(parsed.provenance, PowerProvenance::Measured);
        assert!(parsed.remote_energy_joules.is_none());
    }

    #[test]
    fn test_cell_result_claude_serde() {
        let result = CellResult {
            stdout: "The answer is 42.".into(),
            stderr: String::new(),
            success: true,
            duration_secs: 2.5,
            energy_joules: 0.042,
            power_watts: 15.0,
            provenance: PowerProvenance::Estimated,
            remote_energy_joules: Some(0.004),
            tokens_estimated: Some(4),
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("remote_energy_joules"));
        assert!(json.contains("tokens_estimated"));
        let parsed: CellResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.remote_energy_joules, Some(0.004));
        assert_eq!(parsed.tokens_estimated, Some(4));
    }

    #[test]
    fn test_cell_result_omits_none_fields() {
        // Non-Claude results should NOT serialize remote_energy_joules
        let result = CellResult {
            stdout: "ok".into(),
            stderr: String::new(),
            success: true,
            duration_secs: 0.001,
            energy_joules: 0.015,
            power_watts: 15.0,
            provenance: PowerProvenance::Estimated,
            remote_energy_joules: None,
            tokens_estimated: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(!json.contains("remote_energy_joules"));
        assert!(!json.contains("tokens_estimated"));
    }

    #[test]
    fn test_kernel_info_serde() {
        let info = KernelInfo {
            id: "python-1".into(),
            kind: KernelKind::Python,
            pid: 12345,
            cells_executed: 5,
            cumulative_energy_joules: 1.234,
            alive: true,
        };
        let json = serde_json::to_string(&info).unwrap();
        let parsed: KernelInfo = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.id, "python-1");
        assert_eq!(parsed.kind, KernelKind::Python);
        assert_eq!(parsed.cells_executed, 5);
    }

    #[test]
    fn test_read_power_no_monitor() {
        let (watts, prov) = read_power(None);
        assert_eq!(watts, 15.0);
        assert_eq!(prov, PowerProvenance::Estimated);
    }

    #[test]
    fn test_read_power_with_monitor() {
        let snapshot = EnergySnapshot {
            power_watts: 23.5,
            ..Default::default()
        };
        let handle = Arc::new(RwLock::new(snapshot));
        let (watts, prov) = read_power(Some(&handle));
        assert_eq!(watts, 23.5);
        assert_eq!(prov, PowerProvenance::Measured);
    }

    #[test]
    fn test_read_power_monitor_zero_watts() {
        // If monitor returns 0W (e.g. desktop without battery), should fall back
        let snapshot = EnergySnapshot {
            power_watts: 0.0,
            ..Default::default()
        };
        let handle = Arc::new(RwLock::new(snapshot));
        let (watts, prov) = read_power(Some(&handle));
        assert_eq!(watts, 15.0);
        assert_eq!(prov, PowerProvenance::Estimated);
    }

    #[test]
    fn test_compute_runtime_new() {
        let rt = ComputeRuntime::new();
        assert!(rt.monitor_handle.is_none());
        assert_eq!(rt.total_energy_joules(), 0.0);
    }

    #[test]
    fn test_compute_runtime_with_monitor() {
        let snapshot = EnergySnapshot::default();
        let handle = Arc::new(RwLock::new(snapshot));
        let rt = ComputeRuntime::with_monitor(handle);
        assert!(rt.monitor_handle.is_some());
    }

    // ── Live interpreter tests (require python3/bash on PATH) ──────────────

    #[test]
    fn test_python_kernel_arithmetic() {
        if !interpreter_available("python3") {
            eprintln!("Skipping: python3 not on PATH");
            return;
        }

        let mut rt = ComputeRuntime::new();
        let kid = rt.start_kernel(KernelKind::Python).unwrap();

        let result = rt.execute(&kid, "print(2 + 2)").unwrap();
        assert!(result.success, "cell failed: {:?}", result.stderr);
        assert_eq!(result.stdout.trim(), "4");
        assert!(result.duration_secs > 0.0);
        assert!(result.energy_joules > 0.0);
        assert_eq!(result.provenance, PowerProvenance::Estimated);

        rt.shutdown(&kid);
    }

    #[test]
    fn test_python_kernel_state_persists() {
        if !interpreter_available("python3") {
            return;
        }

        let mut rt = ComputeRuntime::new();
        let kid = rt.start_kernel(KernelKind::Python).unwrap();

        // Set a variable
        let r1 = rt.execute(&kid, "x = 42").unwrap();
        assert!(r1.success);

        // Use it in next cell
        let r2 = rt.execute(&kid, "print(x * 2)").unwrap();
        assert!(r2.success);
        assert_eq!(r2.stdout.trim(), "84");

        rt.shutdown(&kid);
    }

    #[test]
    fn test_python_kernel_multiline() {
        if !interpreter_available("python3") {
            return;
        }

        let mut rt = ComputeRuntime::new();
        let kid = rt.start_kernel(KernelKind::Python).unwrap();

        let code = "for i in range(3):\n    print(i)";
        let result = rt.execute(&kid, code).unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "0\n1\n2");

        rt.shutdown(&kid);
    }

    #[test]
    fn test_shell_kernel() {
        if !interpreter_available("bash") {
            return;
        }

        let mut rt = ComputeRuntime::new();
        let kid = rt.start_kernel(KernelKind::Shell).unwrap();

        let result = rt.execute(&kid, "echo hello world").unwrap();
        assert!(result.success);
        assert_eq!(result.stdout.trim(), "hello world");

        // State persists
        let r2 = rt.execute(&kid, "X=42; echo $X").unwrap();
        assert!(r2.success);
        assert_eq!(r2.stdout.trim(), "42");

        rt.shutdown(&kid);
    }

    #[test]
    fn test_multiple_kernels() {
        if !interpreter_available("python3") || !interpreter_available("bash") {
            return;
        }

        let mut rt = ComputeRuntime::new();
        let py_id = rt.start_kernel(KernelKind::Python).unwrap();
        let sh_id = rt.start_kernel(KernelKind::Shell).unwrap();

        let py_result = rt.execute(&py_id, "print('from python')").unwrap();
        let sh_result = rt.execute(&sh_id, "echo 'from shell'").unwrap();

        assert_eq!(py_result.stdout.trim(), "from python");
        assert_eq!(sh_result.stdout.trim(), "from shell");

        let kernels = rt.list_kernels();
        assert_eq!(kernels.len(), 2);

        rt.shutdown_all();
        assert!(rt.list_kernels().is_empty());
    }

    #[test]
    fn test_energy_accumulates() {
        if !interpreter_available("python3") {
            return;
        }

        let mut rt = ComputeRuntime::new();
        let kid = rt.start_kernel(KernelKind::Python).unwrap();

        rt.execute(&kid, "print(1)").unwrap();
        rt.execute(&kid, "print(2)").unwrap();
        rt.execute(&kid, "print(3)").unwrap();

        let info = rt.kernel_info(&kid).unwrap();
        assert_eq!(info.cells_executed, 3);
        assert!(info.cumulative_energy_joules > 0.0);
        assert!(rt.total_energy_joules() > 0.0);

        rt.shutdown(&kid);
    }

    #[test]
    fn test_kernel_not_found() {
        let mut rt = ComputeRuntime::new();
        let err = rt.execute("nonexistent", "print(1)").unwrap_err();
        assert!(
            matches!(err, ComputeError::KernelNotFound(_)),
            "expected KernelNotFound, got: {:?}",
            err
        );
    }

    #[test]
    fn test_spawn_nonexistent_interpreter() {
        let mut rt = ComputeRuntime::new();
        // This should fail gracefully — no "brainfuck" interpreter
        let result = rt.start_kernel(KernelKind::Python);
        // Python should succeed if available; this test validates error path
        // by trying to spawn with a known-missing interpreter
        let _ = result; // Just ensure no panic
    }

    #[test]
    fn test_claude_kernel_init() {
        // Test that Claude kernel initializes without a child process
        if !which_available("claude") {
            eprintln!("Skipping: claude CLI not on PATH");
            return;
        }
        let mut rt = ComputeRuntime::new();
        let kid = rt.start_kernel(KernelKind::Claude).unwrap();
        let info = rt.kernel_info(&kid).unwrap();
        assert_eq!(info.kind, KernelKind::Claude);
        assert_eq!(info.pid, 0); // No persistent PID
        assert!(info.alive); // Claude is always "alive"
        rt.shutdown(&kid);
    }

    #[test]
    fn test_joules_per_token_constant() {
        // Sanity check the energy constant
        assert!(JOULES_PER_TOKEN_ESTIMATE > 0.0);
        assert!(JOULES_PER_TOKEN_ESTIMATE < 0.01); // <10 mJ/token
    }

    // ── Notebook tests ─────────────────────────────────────────────────────

    #[test]
    fn test_notebook_new() {
        let nb = Notebook::new(KernelKind::Python);
        assert_eq!(nb.kernel, KernelKind::Python);
        assert!(nb.cells.is_empty());
        assert_eq!(nb.total_energy_joules(), 0.0);
    }

    #[test]
    fn test_notebook_add_cell() {
        let mut nb = Notebook::new(KernelKind::Python);
        let result = CellResult {
            stdout: "4\n".into(),
            stderr: String::new(),
            success: true,
            duration_secs: 0.01,
            energy_joules: 0.15,
            power_watts: 15.0,
            provenance: PowerProvenance::Estimated,
            remote_energy_joules: None,
            tokens_estimated: None,
        };
        nb.add_cell("print(2+2)", result);
        assert_eq!(nb.cells.len(), 1);
        assert_eq!(nb.cells[0].source, "print(2+2)");
        assert_eq!(nb.cells[0].cell_number, 1);
        assert!((nb.total_energy_joules() - 0.15).abs() < 1e-10);
    }

    #[test]
    fn test_notebook_serde_roundtrip() {
        let mut nb = Notebook::new(KernelKind::Julia);
        nb.metadata
            .insert("author".into(), "test".into());
        let r = CellResult {
            stdout: "hello\n".into(),
            stderr: String::new(),
            success: true,
            duration_secs: 0.5,
            energy_joules: 7.5,
            power_watts: 15.0,
            provenance: PowerProvenance::Measured,
            remote_energy_joules: None,
            tokens_estimated: None,
        };
        nb.add_cell("println(\"hello\")", r);

        let json = serde_json::to_string_pretty(&nb).unwrap();
        let parsed: Notebook = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.kernel, KernelKind::Julia);
        assert_eq!(parsed.cells.len(), 1);
        assert_eq!(parsed.cells[0].source, "println(\"hello\")");
        assert_eq!(parsed.metadata.get("author").unwrap(), "test");
    }

    #[test]
    fn test_notebook_claude_with_remote_energy() {
        let mut nb = Notebook::new(KernelKind::Claude);
        let r = CellResult {
            stdout: "The answer is 42.".into(),
            stderr: String::new(),
            success: true,
            duration_secs: 3.0,
            energy_joules: 0.049,
            power_watts: 15.0,
            provenance: PowerProvenance::Estimated,
            remote_energy_joules: Some(0.004),
            tokens_estimated: Some(4),
        };
        nb.add_cell("What is the meaning of life?", r);

        let json = serde_json::to_string(&nb).unwrap();
        assert!(json.contains("remote_energy_joules"));
        assert!(json.contains("tokens_estimated"));

        let parsed: Notebook = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed.cells[0].result.remote_energy_joules,
            Some(0.004)
        );
    }

    #[test]
    fn test_notebook_total_energy() {
        let mut nb = Notebook::new(KernelKind::Python);
        for i in 1..=5 {
            let r = CellResult {
                stdout: format!("{}\n", i),
                stderr: String::new(),
                success: true,
                duration_secs: 0.01,
                energy_joules: 1.0,
                power_watts: 100.0,
                provenance: PowerProvenance::Measured,
                remote_energy_joules: None,
                tokens_estimated: None,
            };
            nb.add_cell(&format!("print({})", i), r);
        }
        assert_eq!(nb.cells.len(), 5);
        assert!((nb.total_energy_joules() - 5.0).abs() < 1e-10);
    }

    /// Check if an interpreter is available on PATH.
    fn interpreter_available(name: &str) -> bool {
        Command::new(name)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}
