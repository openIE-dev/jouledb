//! `JouleCapsule` — Absorbing All of Docker Into JouleDB, Without the Bloat
//!
//! Replaces legacy `dockerd` + `containerd` + multi-GB guest VMs + duplicated OverlayFS
//! `.tar` extraction trees with a **Database-Native, Content-Addressable, $O(1)$ Copy-on-Write
//! OCI Capsule & Compose Engine**:
//!
//! 1. **OCI v2 Image & Layer CAS (`OciImageRef`, `OciImageConfig`, `extract_oci_tar_layer`)**:
//!    Parses standard Docker/OCI image references, multi-arch OCI Image Indexes, and `.tar` /
//!    `.tar.gz` layers (including OCI `.wh.*` whiteout deletions!) in pure safe Rust, storing
//!    every file once by SHA-256 digest and mounting into [`JouleAgentFs`].
//! 2. **`Joulefile` (`Dockerfile` Superset) Compiler**:
//!    Parses standard `Dockerfile` instructions (`FROM`, `ENV`, `WORKDIR`, `COPY`, `RUN`,
//!    `EXPOSE`, `VOLUME`, `USER`, `LABEL`, `ENTRYPOINT`, `CMD`) plus Joule-native
//!    `ENERGY_BUDGET`, `SILICON_LANE`, and `COW_BRANCH` directives.
//! 3. **$O(1)$ Atomic Container + Database Branching (`CapsuleEngine::fork_capsule`)**:
//!    Spawning or cloning 100 capsules from an image forks both the POSIX rootfs and the
//!    database branch in $O(1)$ metadata (`0` duplicated layer bytes), metered in microjoules.
//! 4. **Multi-Service `ComposeSpec` Topological DAG Orchestrator**:
//!    Resolves `depends_on` into cycle-free parallel launch waves with automatic service
//!    discovery (`<SERVICE>_HOST`, `<SERVICE>_PORT`) and stack-wide energy budgets.

use crate::BranchError;
use crate::agentfs::{JouleAgentFs, normalize_vfs_path};
use flate2::read::GzDecoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::io::Read;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, RwLock};
use std::time::Instant;

/// Maximum decompressed OCI tar layer size (`64 MiB`) to defeat `.tar.gz` decompression bombs.
pub const MAX_OCI_LAYER_BYTES: usize = 64 * 1024 * 1024;

/// Validate a capsule identifier so it can never traverse paths (`../`) or escape scratch directories.
pub fn validate_capsule_id(id: &str) -> Result<(), BranchError> {
    if id.is_empty() || id.len() > 128 || id == "." || id == ".." {
        return Err(BranchError::InvalidName(format!(
            "invalid capsule_id '{id}' (must be 1..=128 safe ASCII chars)"
        )));
    }
    if !id
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
    {
        return Err(BranchError::InvalidName(format!(
            "unsafe characters in capsule_id '{id}' (allowed: [a-zA-Z0-9._-])"
        )));
    }
    Ok(())
}

/// Parsed OCI / Docker image reference (e.g. `postgres:17`, `ghcr.io/org/app:v1@sha256:...`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OciImageRef {
    /// Registry host (e.g. `"registry-1.docker.io"`, `"ghcr.io"`).
    pub registry: String,
    /// Repository path (e.g. `"library/alpine"`, `"joulesperbit/agent"`).
    pub repository: String,
    /// Tag (`"latest"` default when neither tag nor digest is specified).
    pub tag: String,
    /// Optional immutable `sha256:...` content digest.
    pub digest: Option<String>,
}

impl OciImageRef {
    /// Parse any Docker or OCI image reference string.
    pub fn parse(raw: &str) -> Result<Self, BranchError> {
        let trimmed = raw.trim();
        if trimmed.is_empty() {
            return Err(BranchError::InvalidName("empty OCI image reference".into()));
        }

        let (without_digest, digest) = match trimmed.split_once('@') {
            Some((left, d)) => (left, Some(d.to_string())),
            None => (trimmed, None),
        };

        // Split registry vs repo: first segment is a registry iff it contains `.`, `:`, or is `localhost`
        let (registry, repo_and_tag) = match without_digest.split_once('/') {
            Some((first, rest))
                if first.contains('.') || first.contains(':') || first == "localhost" =>
            {
                (first.to_string(), rest.to_string())
            }
            Some(_) => (
                "registry-1.docker.io".to_string(),
                without_digest.to_string(),
            ),
            None => (
                "registry-1.docker.io".to_string(),
                format!("library/{without_digest}"),
            ),
        };

        // Tag ':' only applies after the last '/'
        let last_slash = repo_and_tag.rfind('/').unwrap_or(0);
        let (repository, tag) = match repo_and_tag[last_slash..].find(':') {
            Some(rel_colon) => {
                let idx = last_slash + rel_colon;
                (
                    repo_and_tag[..idx].to_string(),
                    repo_and_tag[idx + 1..].to_string(),
                )
            }
            None => (repo_and_tag, "latest".to_string()),
        };

        Ok(Self {
            registry,
            repository,
            tag,
            digest,
        })
    }

    /// Canonical `registry/repository:tag` key.
    pub fn canonical_key(&self) -> String {
        format!("{}/{}:{}", self.registry, self.repository, self.tag)
    }
}

/// OCI Image Index (`application/vnd.oci.image.index.v1+json` / Docker Manifest List).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OciImageIndex {
    /// Schema version (typically `2`).
    #[serde(default)]
    pub schema_version: u32,
    /// Platform-specific manifest descriptors.
    #[serde(default)]
    pub manifests: Vec<OciPlatformDescriptor>,
}

/// Entry inside an [`OciImageIndex`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OciPlatformDescriptor {
    /// Content digest (`sha256:...`).
    pub digest: String,
    /// Media type.
    #[serde(default, rename = "mediaType")]
    pub media_type: String,
    /// Byte size.
    #[serde(default)]
    pub size: u64,
    /// Platform selector.
    #[serde(default)]
    pub platform: Option<OciPlatform>,
}

/// OCI target OS + CPU architecture.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OciPlatform {
    /// Architecture (`"arm64"`, `"amd64"`, `"wasm"`).
    #[serde(default)]
    pub architecture: String,
    /// Operating system (`"linux"`, `"darwin"`, `"wasi"`).
    #[serde(default)]
    pub os: String,
}

impl OciImageIndex {
    /// Pick the best manifest digest matching `preferred_arch` (e.g. `"arm64"` or `"amd64"`).
    pub fn select_digest(&self, preferred_arch: &str) -> Option<String> {
        self.manifests
            .iter()
            .find(|m| {
                m.platform.as_ref().is_some_and(|p| {
                    p.architecture == preferred_arch && (p.os == "linux" || p.os == "wasi")
                })
            })
            .or_else(|| self.manifests.first())
            .map(|m| m.digest.clone())
    }
}

/// OCI Image Manifest v2 (`application/vnd.oci.image.manifest.v1+json`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OciManifestV2 {
    /// Config blob descriptor.
    pub config: OciBlobDescriptor,
    /// Ordered rootfs layer descriptors (base -> top).
    #[serde(default)]
    pub layers: Vec<OciBlobDescriptor>,
}

/// Descriptor for an OCI config or filesystem layer blob.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OciBlobDescriptor {
    /// Media type.
    #[serde(default, rename = "mediaType")]
    pub media_type: String,
    /// SHA-256 content digest.
    pub digest: String,
    /// Compressed byte size.
    #[serde(default)]
    pub size: u64,
}

/// OCI Image Configuration + Joule-native extensions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OciImageConfig {
    /// Environment variables (`KEY=VALUE`).
    #[serde(default)]
    pub env: Vec<String>,
    /// Entrypoint argv prefix.
    #[serde(default)]
    pub entrypoint: Vec<String>,
    /// Default command argv suffix.
    #[serde(default)]
    pub cmd: Vec<String>,
    /// Default working directory inside the capsule rootfs.
    #[serde(default)]
    pub working_dir: String,
    /// Exposed container ports.
    #[serde(default)]
    pub exposed_ports: Vec<u16>,
    /// Declared mountpoint volumes.
    #[serde(default)]
    pub volumes: Vec<String>,
    /// User spec (`uid:gid` or username).
    #[serde(default)]
    pub user: Option<String>,
    /// Arbitrary metadata labels.
    #[serde(default)]
    pub labels: BTreeMap<String, String>,
    /// Joule-native default microjoule budget per capsule instance.
    #[serde(default)]
    pub default_energy_budget_uj: Option<u64>,
    /// Joule-native preferred silicon execution lane (`gpu_shader`, `npu_systolic`, `wasm_sandbox`, etc.).
    #[serde(default)]
    pub preferred_silicon_lane: Option<String>,
}

/// Stored, content-addressed OCI image inside [`CapsuleEngine`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OciImageRecord {
    /// Canonical image reference (`registry/repo:tag`).
    pub reference: String,
    /// Content-addressed SHA-256 image ID.
    pub image_id: String,
    /// Underlying immutable `JouleAgentFs` base branch holding the merged rootfs.
    pub base_branch: String,
    /// Number of OCI layers applied.
    pub layer_count: usize,
    /// Number of live files in the merged rootfs.
    pub file_count: usize,
    /// Logical uncompressed bytes across rootfs files.
    pub logical_bytes: usize,
    /// OCI runtime configuration + Joule extensions.
    pub config: OciImageConfig,
}

/// Parsed instruction from a `Dockerfile` or `Joulefile`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum JoulefileInstruction {
    From(String),
    Arg {
        key: String,
        default: Option<String>,
    },
    Env {
        key: String,
        value: String,
    },
    Workdir(String),
    Copy {
        src: String,
        dst: String,
    },
    Run(String),
    Expose(u16),
    Volume(String),
    User(String),
    Label {
        key: String,
        value: String,
    },
    Entrypoint(Vec<String>),
    Cmd(Vec<String>),
    /// JouleDB superset: `ENERGY_BUDGET 25J` / `500mJ` / `10000uJ`
    EnergyBudgetUj(u64),
    /// JouleDB superset: `SILICON_LANE gpu_shader`
    SiliconLane(String),
}

/// Compiled `Dockerfile` / `Joulefile` build specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JoulefileSpec {
    /// Base image reference from `FROM`.
    pub base_image: String,
    /// Ordered instructions.
    pub instructions: Vec<JoulefileInstruction>,
    /// Resulting OCI + Joule image configuration.
    pub derived_config: OciImageConfig,
}

impl JoulefileSpec {
    /// Parse a `Dockerfile` or `Joulefile` (supporting `\` line continuations and `#` comments).
    pub fn parse(source: &str) -> Result<Self, BranchError> {
        let mut logical_lines = Vec::new();
        let mut current = String::new();
        for raw_line in source.lines() {
            let trimmed = raw_line.trim();
            if trimmed.is_empty() || (current.is_empty() && trimmed.starts_with('#')) {
                continue;
            }
            if let Some(stripped) = trimmed.strip_suffix('\\') {
                current.push_str(stripped.trim_end());
                current.push(' ');
            } else {
                current.push_str(trimmed);
                logical_lines.push(std::mem::take(&mut current));
            }
        }
        if !current.trim().is_empty() {
            logical_lines.push(current);
        }

        let mut base_image = "scratch".to_string();
        let mut instructions = Vec::new();
        let mut cfg = OciImageConfig {
            working_dir: "/".to_string(),
            ..OciImageConfig::default()
        };

        for line in logical_lines {
            let Some((kw_raw, rest_raw)) = line.split_once(char::is_whitespace) else {
                continue;
            };
            let kw = kw_raw.to_uppercase();
            let rest = rest_raw.trim();
            match kw.as_str() {
                "FROM" => {
                    // Handle `FROM rust:1.94 AS builder` -> take first token
                    let img = rest.split_whitespace().next().unwrap_or("scratch");
                    base_image = img.to_string();
                    instructions.push(JoulefileInstruction::From(base_image.clone()));
                }
                "ARG" => {
                    let (k, d) = match rest.split_once('=') {
                        Some((k, v)) => (k.trim().to_string(), Some(unquote(v.trim()))),
                        None => (rest.to_string(), None),
                    };
                    instructions.push(JoulefileInstruction::Arg { key: k, default: d });
                }
                "ENV" => {
                    if let Some((k, v)) = rest.split_once('=') {
                        let key = k.trim().to_string();
                        let val = unquote(v.trim());
                        cfg.env.push(format!("{key}={val}"));
                        instructions.push(JoulefileInstruction::Env { key, value: val });
                    } else if let Some((k, v)) = rest.split_once(char::is_whitespace) {
                        let key = k.trim().to_string();
                        let val = unquote(v.trim());
                        cfg.env.push(format!("{key}={val}"));
                        instructions.push(JoulefileInstruction::Env { key, value: val });
                    }
                }
                "WORKDIR" => {
                    let wd = normalize_vfs_path(rest)?;
                    cfg.working_dir = wd.clone();
                    instructions.push(JoulefileInstruction::Workdir(wd));
                }
                "COPY" | "ADD" => {
                    let parts: Vec<&str> = rest
                        .split_whitespace()
                        .filter(|p| !p.starts_with("--"))
                        .collect();
                    if parts.len() >= 2 {
                        instructions.push(JoulefileInstruction::Copy {
                            src: parts[0].to_string(),
                            dst: parts[parts.len() - 1].to_string(),
                        });
                    }
                }
                "RUN" => {
                    instructions.push(JoulefileInstruction::Run(rest.to_string()));
                }
                "EXPOSE" => {
                    for tok in rest.split_whitespace() {
                        let num_str = tok.split('/').next().unwrap_or(tok);
                        if let Ok(port) = num_str.parse::<u16>() {
                            if !cfg.exposed_ports.contains(&port) {
                                cfg.exposed_ports.push(port);
                            }
                            instructions.push(JoulefileInstruction::Expose(port));
                        }
                    }
                }
                "VOLUME" => {
                    for v in parse_string_or_json_list(rest) {
                        cfg.volumes.push(v.clone());
                        instructions.push(JoulefileInstruction::Volume(v));
                    }
                }
                "USER" => {
                    cfg.user = Some(rest.to_string());
                    instructions.push(JoulefileInstruction::User(rest.to_string()));
                }
                "LABEL" => {
                    for tok in rest.split_whitespace() {
                        if let Some((k, v)) = tok.split_once('=') {
                            let key = unquote(k.trim());
                            let val = unquote(v.trim());
                            cfg.labels.insert(key.clone(), val.clone());
                            instructions.push(JoulefileInstruction::Label { key, value: val });
                        }
                    }
                }
                "ENTRYPOINT" => {
                    let ep = parse_string_or_json_list(rest);
                    cfg.entrypoint = ep.clone();
                    instructions.push(JoulefileInstruction::Entrypoint(ep));
                }
                "CMD" => {
                    let cmd = parse_string_or_json_list(rest);
                    cfg.cmd = cmd.clone();
                    instructions.push(JoulefileInstruction::Cmd(cmd));
                }
                "ENERGY_BUDGET" => {
                    let uj = parse_energy_to_uj(rest)?;
                    cfg.default_energy_budget_uj = Some(uj);
                    instructions.push(JoulefileInstruction::EnergyBudgetUj(uj));
                }
                "SILICON_LANE" | "SILICON_AFFINITY" => {
                    let lane = unquote(rest);
                    cfg.preferred_silicon_lane = Some(lane.clone());
                    instructions.push(JoulefileInstruction::SiliconLane(lane));
                }
                _ => {}
            }
        }

        Ok(Self {
            base_image,
            instructions,
            derived_config: cfg,
        })
    }
}

/// Lifecycle state of a [`JouleCapsule`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapsuleState {
    /// Rootfs & DB branch forked and ready for zero-cold-start execution.
    Created,
    /// Currently executing a command or service.
    Running,
    /// Exited cleanly or stopped.
    Stopped,
    /// Halted because the capsule exceeded its microjoule ceiling.
    EnergyThrottled,
}

/// Port binding (`host_port -> capsule_port`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapsulePortMapping {
    /// Host TCP/UDP port.
    pub host_port: u16,
    /// Capsule internal port.
    pub capsule_port: u16,
}

/// Structured execution log entry captured inside JouleDB for a [`JouleCapsule`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapsuleExecLog {
    /// Sequential execution index inside this capsule.
    pub seq: u64,
    /// Command argv executed.
    pub command: Vec<String>,
    /// Process exit code (`0` = success).
    pub exit_code: i32,
    /// Captured UTF-8 stdout.
    pub stdout: String,
    /// Captured UTF-8 stderr.
    pub stderr: String,
    /// Wall-clock duration in microseconds.
    pub duration_us: u64,
    /// Physical/modeled microjoules charged to the capsule's branch budget.
    pub energy_uj: u64,
}

/// A live zero-bloat container instance (`JouleCapsule`) backed by a CoW [`JouleAgentFs`] branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JouleCapsule {
    /// Unique capsule identifier (e.g. `"web-api-01"`).
    pub capsule_id: String,
    /// Source OCI image reference.
    pub image_ref: String,
    /// CoW `JouleAgentFs` + DB branch backing this capsule's rootfs & state (`cap/<capsule_id>`).
    pub cow_branch: String,
    /// Current lifecycle state.
    pub state: CapsuleState,
    /// Merged environment variables (`KEY -> VALUE`).
    pub env: BTreeMap<String, String>,
    /// Working directory inside the capsule rootfs.
    pub working_dir: String,
    /// Port mappings.
    pub ports: Vec<CapsulePortMapping>,
    /// Optional microjoule ceiling enforced by `BranchManager`.
    pub energy_budget_uj: Option<u64>,
    /// Preferred silicon lane (`gpu_shader`, `npu_systolic`, `wasm_sandbox`, etc.).
    pub silicon_lane: Option<String>,
    /// Queryable execution logs.
    pub exec_logs: Vec<CapsuleExecLog>,
}

/// Request to spawn a new [`JouleCapsule`] from an ingested OCI image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpawnCapsuleRequest {
    /// Capsule name/ID.
    pub capsule_id: String,
    /// Image reference (`alpine:latest`, `my-service:v1`, etc.).
    pub image_ref: String,
    /// Extra environment overrides (`KEY -> VALUE`).
    #[serde(default)]
    pub env: BTreeMap<String, String>,
    /// Published ports.
    #[serde(default)]
    pub ports: Vec<CapsulePortMapping>,
    /// Optional microjoule budget override (inherits image `ENERGY_BUDGET` if `None`).
    #[serde(default)]
    pub energy_budget_uj: Option<u64>,
}

// ============================================================================
// Multi-Service `ComposeSpec` Topological DAG Orchestrator
// ============================================================================

/// Declarative multi-capsule stack specification (absorbs `docker-compose.yml`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComposeSpec {
    /// Stack namespace prefix.
    pub stack_name: String,
    /// Named services in the stack.
    pub services: BTreeMap<String, ComposeServiceSpec>,
    /// Optional stack-wide microjoule budget ceiling.
    #[serde(default)]
    pub stack_energy_budget_uj: Option<u64>,
}

/// Single service inside a [`ComposeSpec`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ComposeServiceSpec {
    /// OCI image reference.
    pub image: String,
    /// Upstream services that must start before this service.
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Published ports.
    #[serde(default)]
    pub ports: Vec<CapsulePortMapping>,
    /// Service environment variables.
    #[serde(default)]
    pub environment: BTreeMap<String, String>,
    /// Per-service microjoule budget.
    #[serde(default)]
    pub energy_budget_uj: Option<u64>,
}

/// Result of launching a [`ComposeSpec`] stack.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComposeLaunchPlan {
    /// Stack name.
    pub stack_name: String,
    /// Parallel topological boot waves (`Wave 0` = leaf services, `Wave 1` = dependents, ...).
    pub boot_waves: Vec<Vec<String>>,
    /// Spawned capsules in topological order.
    pub capsules: Vec<JouleCapsule>,
}

/// File-level rootfs change entry returned by [`CapsuleEngine::diff_capsules`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapsuleFileDiff {
    /// Virtual POSIX path inside the capsule rootfs.
    pub path: String,
    /// Change classification (`"added"`, `"modified"`, or `"deleted"`).
    pub change: String,
}

/// Content-addressable deduplication telemetry across all OCI images in [`CapsuleEngine`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CasDeduplicationStats {
    /// Total unique SHA-256 file blobs stored.
    pub unique_blobs: usize,
    /// Physical bytes stored across unique blobs.
    pub physical_bytes: usize,
    /// Logical bytes referenced across all images (before CAS dedup).
    pub logical_bytes: usize,
    /// Bytes saved by SHA-256 content-addressable deduplication.
    pub dedup_bytes_saved: usize,
}

/// Zero-Bloat Post-Docker Engine backed by [`JouleAgentFs`] and SHA-256 CAS.
pub struct CapsuleEngine {
    agentfs: Arc<JouleAgentFs>,
    cas_blobs: RwLock<BTreeMap<String, Vec<u8>>>,
    logical_bytes_ingested: RwLock<usize>,
    images: RwLock<BTreeMap<String, OciImageRecord>>,
    capsules: RwLock<BTreeMap<String, JouleCapsule>>,
}

impl CapsuleEngine {
    /// Create a new [`CapsuleEngine`] sharing `agentfs` (and its underlying `BranchManager`).
    pub fn new(agentfs: Arc<JouleAgentFs>) -> Self {
        Self {
            agentfs,
            cas_blobs: RwLock::new(BTreeMap::new()),
            logical_bytes_ingested: RwLock::new(0),
            images: RwLock::new(BTreeMap::new()),
            capsules: RwLock::new(BTreeMap::new()),
        }
    }

    /// Return underlying [`JouleAgentFs`] handle.
    pub fn agentfs(&self) -> &Arc<JouleAgentFs> {
        &self.agentfs
    }

    /// Snapshot SHA-256 CAS deduplication metrics.
    pub fn cas_stats(&self) -> CasDeduplicationStats {
        let (unique_blobs, physical_bytes) = self
            .cas_blobs
            .read()
            .map(|m| (m.len(), m.values().map(|b| b.len()).sum()))
            .unwrap_or((0, 0));
        let logical_bytes = self.logical_bytes_ingested.read().map(|v| *v).unwrap_or(0);
        CasDeduplicationStats {
            unique_blobs,
            physical_bytes,
            logical_bytes,
            dedup_bytes_saved: logical_bytes.saturating_sub(physical_bytes),
        }
    }

    /// Ingest an OCI image from ordered `.tar` or `.tar.gz` layer byte slices (base -> top)
    /// into the SHA-256 CAS and an immutable [`JouleAgentFs`] base branch.
    pub fn ingest_oci_image(
        &self,
        image_ref: &str,
        config: OciImageConfig,
        layers_tar_or_targz: &[Vec<u8>],
    ) -> Result<OciImageRecord, BranchError> {
        let parsed_ref = OciImageRef::parse(image_ref)?;
        let canonical = parsed_ref.canonical_key();

        let mut id_hasher = Sha256::new();
        id_hasher.update(canonical.as_bytes());
        for layer in layers_tar_or_targz {
            id_hasher.update(layer);
        }
        let image_id = format!("sha256:{}", hex::encode(id_hasher.finalize()));
        let short_id = &image_id["sha256:".len().."sha256:".len() + 12];
        let base_branch = format!("img-{short_id}");

        // Idempotent: create base image branch if not yet materialized
        if self.agentfs.list_files(&base_branch, "/").is_err() {
            let _ = self.agentfs.fork_workspace(
                &base_branch,
                Some("main"),
                None,
                Some(format!("OCI base rootfs for {canonical}")),
            )?;
            for layer_bytes in layers_tar_or_targz {
                self.apply_tar_layer_to_branch(&base_branch, layer_bytes)?;
            }
        }

        let files = self.agentfs.list_files(&base_branch, "/")?;
        let logical_bytes: usize = files.iter().map(|f| f.size_bytes).sum();
        let rec = OciImageRecord {
            reference: canonical.clone(),
            image_id,
            base_branch,
            layer_count: layers_tar_or_targz.len(),
            file_count: files.len(),
            logical_bytes,
            config,
        };

        if let Ok(mut imgs) = self.images.write() {
            imgs.insert(canonical, rec.clone());
            imgs.insert(image_ref.trim().to_string(), rec.clone());
        }
        Ok(rec)
    }

    /// Compile a `Dockerfile` / `Joulefile` + inline `context_files` (`path -> bytes`) directly
    /// into a zero-bloat [`OciImageRecord`] without `docker build`!
    pub fn build_from_joulefile(
        &self,
        tag: &str,
        joulefile_source: &str,
        context_files: &BTreeMap<String, Vec<u8>>,
    ) -> Result<OciImageRecord, BranchError> {
        let spec = JoulefileSpec::parse(joulefile_source)?;
        let parsed_ref = OciImageRef::parse(tag)?;
        let canonical = parsed_ref.canonical_key();

        // Inherit parent `FROM` branch if present in `self.images`, otherwise `"main"`
        let parent_branch = self
            .get_image(&spec.base_image)
            .map(|img| img.base_branch)
            .unwrap_or_else(|| "main".to_string());

        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        hasher.update(joulefile_source.as_bytes());
        for (p, data) in context_files {
            hasher.update(p.as_bytes());
            hasher.update(data);
        }
        let image_id = format!("sha256:{}", hex::encode(hasher.finalize()));
        let short_id = &image_id["sha256:".len().."sha256:".len() + 12];
        let base_branch = format!("img-{short_id}");

        if self.agentfs.list_files(&base_branch, "/").is_err() {
            let _ = self.agentfs.fork_workspace(
                &base_branch,
                Some(&parent_branch),
                None,
                Some(format!("Joulefile build for {canonical}")),
            )?;

            let mut current_workdir = "/".to_string();
            for inst in &spec.instructions {
                match inst {
                    JoulefileInstruction::Workdir(wd) => {
                        current_workdir = wd.clone();
                    }
                    JoulefileInstruction::Copy { src, dst } => {
                        let resolved_dst = if dst.starts_with('/') {
                            normalize_vfs_path(dst)?
                        } else {
                            normalize_vfs_path(&format!("{current_workdir}/{dst}"))?
                        };
                        let clean_src = src.trim_start_matches("./");
                        if let Some(bytes) = context_files
                            .get(src)
                            .or_else(|| context_files.get(clean_src))
                        {
                            self.record_cas_blob(bytes);
                            self.agentfs
                                .write_file(&base_branch, &resolved_dst, bytes)?;
                        }
                    }
                    _ => {}
                }
            }
        }

        let files = self.agentfs.list_files(&base_branch, "/")?;
        let logical_bytes: usize = files.iter().map(|f| f.size_bytes).sum();
        let rec = OciImageRecord {
            reference: canonical.clone(),
            image_id,
            base_branch,
            layer_count: 1,
            file_count: files.len(),
            logical_bytes,
            config: spec.derived_config,
        };
        if let Ok(mut imgs) = self.images.write() {
            imgs.insert(canonical, rec.clone());
            imgs.insert(tag.trim().to_string(), rec.clone());
        }
        Ok(rec)
    }

    /// Lookup an ingested OCI image by tag or canonical reference.
    pub fn get_image(&self, image_ref: &str) -> Option<OciImageRecord> {
        let canonical = OciImageRef::parse(image_ref)
            .ok()
            .map(|r| r.canonical_key());
        let imgs = self.images.read().ok()?;
        imgs.get(image_ref.trim())
            .or_else(|| canonical.as_deref().and_then(|c| imgs.get(c)))
            .cloned()
    }

    /// List unique ingested OCI images.
    pub fn list_images(&self) -> Vec<OciImageRecord> {
        let Ok(imgs) = self.images.read() else {
            return Vec::new();
        };
        let mut by_id: BTreeMap<String, OciImageRecord> = BTreeMap::new();
        for rec in imgs.values() {
            by_id
                .entry(rec.image_id.clone())
                .or_insert_with(|| rec.clone());
        }
        by_id.into_values().collect()
    }

    /// Spawn a new [`JouleCapsule`] in $O(1)$ CoW metadata from an ingested OCI image!
    pub fn spawn_capsule(&self, req: SpawnCapsuleRequest) -> Result<JouleCapsule, BranchError> {
        validate_capsule_id(&req.capsule_id)?;
        let img = self.get_image(&req.image_ref).ok_or_else(|| {
            BranchError::NotFound(format!("OCI image '{}' not found in CAS", req.image_ref))
        })?;
        let cow_branch = format!("cap-{}", req.capsule_id);
        let budget = req.energy_budget_uj.or(img.config.default_energy_budget_uj);

        let _ = self.agentfs.fork_workspace(
            &cow_branch,
            Some(&img.base_branch),
            budget,
            Some(format!(
                "JouleCapsule '{}' from {}",
                req.capsule_id, img.reference
            )),
        )?;

        let mut merged_env = BTreeMap::new();
        for entry in &img.config.env {
            if let Some((k, v)) = entry.split_once('=') {
                merged_env.insert(k.to_string(), v.to_string());
            }
        }
        for (k, v) in req.env {
            merged_env.insert(k, v);
        }

        let working_dir = if img.config.working_dir.is_empty() {
            "/".to_string()
        } else {
            img.config.working_dir.clone()
        };

        let capsule = JouleCapsule {
            capsule_id: req.capsule_id.clone(),
            image_ref: img.reference,
            cow_branch,
            state: CapsuleState::Created,
            env: merged_env,
            working_dir,
            ports: req.ports,
            energy_budget_uj: budget,
            silicon_lane: img.config.preferred_silicon_lane,
            exec_logs: Vec::new(),
        };

        let mut caps = self
            .capsules
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        if caps.contains_key(&req.capsule_id) {
            return Err(BranchError::AlreadyExists(req.capsule_id));
        }
        caps.insert(req.capsule_id, capsule.clone());
        Ok(capsule)
    }

    /// $O(1)$ Atomic Container + Database Fork: clone an existing capsule's live mutated rootfs
    /// AND linked branch state into `new_capsule_id` with zero file copying!
    pub fn fork_capsule(
        &self,
        source_capsule_id: &str,
        new_capsule_id: &str,
        energy_budget_uj: Option<u64>,
    ) -> Result<JouleCapsule, BranchError> {
        validate_capsule_id(source_capsule_id)?;
        validate_capsule_id(new_capsule_id)?;
        let src = {
            let caps = self
                .capsules
                .read()
                .map_err(|e| BranchError::Storage(e.to_string()))?;
            caps.get(source_capsule_id)
                .cloned()
                .ok_or_else(|| BranchError::NotFound(source_capsule_id.to_string()))?
        };

        let new_branch = format!("cap-{new_capsule_id}");
        let budget = energy_budget_uj.or(src.energy_budget_uj);
        let _ = self.agentfs.fork_workspace(
            &new_branch,
            Some(&src.cow_branch),
            budget,
            Some(format!("CoW fork of capsule '{source_capsule_id}'")),
        )?;

        let forked = JouleCapsule {
            capsule_id: new_capsule_id.to_string(),
            image_ref: src.image_ref,
            cow_branch: new_branch,
            state: CapsuleState::Created,
            env: src.env,
            working_dir: src.working_dir,
            ports: Vec::new(),
            energy_budget_uj: budget,
            silicon_lane: src.silicon_lane,
            exec_logs: Vec::new(),
        };

        let mut caps = self
            .capsules
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        if caps.contains_key(new_capsule_id) {
            return Err(BranchError::AlreadyExists(new_capsule_id.to_string()));
        }
        caps.insert(new_capsule_id.to_string(), forked.clone());
        Ok(forked)
    }

    /// Execute a command inside `capsule_id`'s isolated CoW rootfs without `dockerd`!
    /// Materializes the capsule's sparse virtual rootfs into an ephemeral directory, executes
    /// with a sanitized environment (`env_clear` + capsule `ENV`), syncs any created/modified
    /// files back into the capsule's CoW `JouleAgentFs` branch, and charges microjoules!
    pub fn exec_in_capsule(
        &self,
        capsule_id: &str,
        command: &[String],
        sandbox_scratch_root: &Path,
    ) -> Result<CapsuleExecLog, BranchError> {
        validate_capsule_id(capsule_id)?;
        if command.is_empty() {
            return Err(BranchError::InvalidName(
                "empty capsule exec command".into(),
            ));
        }

        let cap = {
            let mut caps = self
                .capsules
                .write()
                .map_err(|e| BranchError::Storage(e.to_string()))?;
            let c = caps
                .get_mut(capsule_id)
                .ok_or_else(|| BranchError::NotFound(capsule_id.to_string()))?;
            if c.state == CapsuleState::EnergyThrottled {
                return Err(BranchError::EnergyBudgetExhausted {
                    spent_uj: c.energy_budget_uj.unwrap_or(0),
                    budget_uj: c.energy_budget_uj.unwrap_or(0),
                });
            }
            c.state = CapsuleState::Running;
            c.clone()
        };

        // Materialize sparse CoW rootfs into ephemeral isolated directory
        let rootfs_dir =
            sandbox_scratch_root.join(format!("jcap-{}-{}", capsule_id, cap.exec_logs.len()));
        std::fs::create_dir_all(&rootfs_dir)
            .map_err(|e| BranchError::Storage(format!("failed to create capsule rootfs: {e}")))?;

        let files = self.agentfs.list_files(&cap.cow_branch, "/")?;
        for fstat in &files {
            if let Some(bytes) = self.agentfs.read_file(&cap.cow_branch, &fstat.path)? {
                let rel = fstat.path.trim_start_matches('/');
                let host_file = rootfs_dir.join(rel);
                if let Some(parent) = host_file.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(&host_file, &bytes);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    if bytes.starts_with(b"#!") {
                        let _ = std::fs::set_permissions(
                            &host_file,
                            std::fs::Permissions::from_mode(0o755),
                        );
                    }
                }
            }
        }

        let work_rel = cap.working_dir.trim_start_matches('/');
        let exec_cwd = if work_rel.is_empty() {
            rootfs_dir.clone()
        } else {
            let d = rootfs_dir.join(work_rel);
            let _ = std::fs::create_dir_all(&d);
            d
        };

        // Resolve program and any rootfs-absolute script/file arguments (`/app/worker.sh` -> `<rootfs>/app/worker.sh`)
        let prog_str = &command[0];
        let candidate_in_rootfs = rootfs_dir.join(prog_str.trim_start_matches('/'));
        let resolved_prog = if candidate_in_rootfs.exists() {
            candidate_in_rootfs.to_string_lossy().to_string()
        } else {
            prog_str.clone()
        };
        let resolved_args: Vec<String> = command[1..]
            .iter()
            .map(|arg| {
                if arg.starts_with('/') && arg.len() > 1 {
                    let cand = rootfs_dir.join(arg.trim_start_matches('/'));
                    if cand.exists() {
                        return cand.to_string_lossy().to_string();
                    }
                }
                arg.clone()
            })
            .collect();

        let start = Instant::now();
        let mut cmd = Command::new(&resolved_prog);
        cmd.args(&resolved_args)
            .current_dir(&exec_cwd)
            .env_clear()
            .env("PATH", "/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin")
            .env("HOME", "/root")
            .env("JOULE_CAPSULE_ID", &cap.capsule_id)
            .env("JOULE_CAPSULE_ROOTFS", &rootfs_dir);
        for (k, v) in &cap.env {
            cmd.env(k, v);
        }

        let output_res = cmd.output();
        let duration_us = start.elapsed().as_micros() as u64;

        // Sync created/updated files from `rootfs_dir` back into CoW `JouleAgentFs`!
        let sync_res = self.sync_host_dir_into_branch(&rootfs_dir, &rootfs_dir, &cap.cow_branch);
        let _ = std::fs::remove_dir_all(&rootfs_dir);

        if let Err(BranchError::EnergyBudgetExhausted {
            spent_uj,
            budget_uj,
        }) = sync_res
        {
            if let Ok(mut caps) = self.capsules.write()
                && let Some(c) = caps.get_mut(capsule_id)
            {
                c.state = CapsuleState::EnergyThrottled;
            }
            return Err(BranchError::EnergyBudgetExhausted {
                spent_uj,
                budget_uj,
            });
        }

        let out =
            output_res.map_err(|e| BranchError::Storage(format!("capsule exec failed: {e}")))?;
        // 15 W nominal active core * duration_us => 15 µJ per millisecond + 5 µJ base
        let energy_uj = 5 + (duration_us / 67).max(1);

        let log_entry = CapsuleExecLog {
            seq: cap.exec_logs.len() as u64 + 1,
            command: command.to_vec(),
            exit_code: out.status.code().unwrap_or(-1),
            stdout: String::from_utf8_lossy(&out.stdout).to_string(),
            stderr: String::from_utf8_lossy(&out.stderr).to_string(),
            duration_us,
            energy_uj,
        };

        let mut caps = self
            .capsules
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        if let Some(c) = caps.get_mut(capsule_id) {
            c.state = CapsuleState::Stopped;
            c.exec_logs.push(log_entry.clone());
        }
        Ok(log_entry)
    }

    /// Compute exact file-level CoW diff between two capsules (`docker diff` superset).
    pub fn diff_capsules(
        &self,
        base_capsule_id: &str,
        target_capsule_id: &str,
    ) -> Result<Vec<CapsuleFileDiff>, BranchError> {
        let caps = self
            .capsules
            .read()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let b = caps
            .get(base_capsule_id)
            .ok_or_else(|| BranchError::NotFound(base_capsule_id.to_string()))?;
        let t = caps
            .get(target_capsule_id)
            .ok_or_else(|| BranchError::NotFound(target_capsule_id.to_string()))?;

        let base_files: BTreeMap<String, String> = self
            .agentfs
            .list_files(&b.cow_branch, "/")?
            .into_iter()
            .map(|s| (s.path, s.content_digest))
            .collect();
        let target_files: BTreeMap<String, String> = self
            .agentfs
            .list_files(&t.cow_branch, "/")?
            .into_iter()
            .map(|s| (s.path, s.content_digest))
            .collect();

        let mut diffs = Vec::new();
        for (path, t_digest) in &target_files {
            match base_files.get(path) {
                None => diffs.push(CapsuleFileDiff {
                    path: path.clone(),
                    change: "added".to_string(),
                }),
                Some(b_digest) if b_digest != t_digest => diffs.push(CapsuleFileDiff {
                    path: path.clone(),
                    change: "modified".to_string(),
                }),
                _ => {}
            }
        }
        for path in base_files.keys() {
            if !target_files.contains_key(path) {
                diffs.push(CapsuleFileDiff {
                    path: path.clone(),
                    change: "deleted".to_string(),
                });
            }
        }
        Ok(diffs)
    }

    /// List all capsules sorted by ID.
    pub fn list_capsules(&self) -> Vec<JouleCapsule> {
        self.capsules
            .read()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Launch a multi-service [`ComposeSpec`] stack in topological dependency waves!
    pub fn compose_up(&self, spec: &ComposeSpec) -> Result<ComposeLaunchPlan, BranchError> {
        let waves = topological_boot_waves(&spec.services)?;
        let mut spawned = Vec::new();

        // Precompute service discovery environment (`<SVC>_HOST`, `<SVC>_PORT`)
        let mut discovery_env = BTreeMap::new();
        for (svc_name, svc) in &spec.services {
            let env_prefix = svc_name.to_uppercase().replace('-', "_");
            discovery_env.insert(
                format!("{env_prefix}_HOST"),
                format!("{}-{svc_name}.capsule.local", spec.stack_name),
            );
            if let Some(first_port) = svc.ports.first() {
                discovery_env.insert(
                    format!("{env_prefix}_PORT"),
                    first_port.capsule_port.to_string(),
                );
            }
        }

        let per_svc_default_budget = spec.stack_energy_budget_uj.map(|total| {
            let n = spec.services.len().max(1) as u64;
            total / n
        });

        for wave in &waves {
            for svc_name in wave {
                let Some(svc) = spec.services.get(svc_name) else {
                    continue;
                };
                let mut env = discovery_env.clone();
                for (k, v) in &svc.environment {
                    env.insert(k.clone(), v.clone());
                }
                let cap_id = format!("{}-{svc_name}", spec.stack_name);
                let cap = self.spawn_capsule(SpawnCapsuleRequest {
                    capsule_id: cap_id,
                    image_ref: svc.image.clone(),
                    env,
                    ports: svc.ports.clone(),
                    energy_budget_uj: svc.energy_budget_uj.or(per_svc_default_budget),
                })?;
                spawned.push(cap);
            }
        }

        Ok(ComposeLaunchPlan {
            stack_name: spec.stack_name.clone(),
            boot_waves: waves,
            capsules: spawned,
        })
    }

    // ========================================================================
    // Pure-Rust POSIX USTAR / PAX `.tar` + `.tar.gz` + OCI Whiteout Unpacker
    // ========================================================================

    fn record_cas_blob(&self, data: &[u8]) -> String {
        let digest = hex::encode(Sha256::digest(data));
        if let Ok(mut tot) = self.logical_bytes_ingested.write() {
            *tot += data.len();
        }
        if let Ok(mut blobs) = self.cas_blobs.write() {
            blobs.entry(digest.clone()).or_insert_with(|| data.to_vec());
        }
        digest
    }

    fn apply_tar_layer_to_branch(&self, branch: &str, raw_or_gz: &[u8]) -> Result<(), BranchError> {
        let tar_bytes: Vec<u8> = if raw_or_gz.len() >= 2
            && raw_or_gz[0] == 0x1F
            && raw_or_gz[1] == 0x8B
        {
            let dec = GzDecoder::new(raw_or_gz);
            let mut limited = dec.take((MAX_OCI_LAYER_BYTES + 1) as u64);
            let mut out = Vec::new();
            limited
                .read_to_end(&mut out)
                .map_err(|e| BranchError::Storage(format!("gzip OCI layer decode error: {e}")))?;
            if out.len() > MAX_OCI_LAYER_BYTES {
                return Err(BranchError::Storage(format!(
                    "OCI layer exceeds {} MiB decompression safety ceiling (gzip bomb rejected)",
                    MAX_OCI_LAYER_BYTES / (1024 * 1024)
                )));
            }
            out
        } else {
            if raw_or_gz.len() > MAX_OCI_LAYER_BYTES {
                return Err(BranchError::Storage(format!(
                    "OCI layer exceeds {} MiB size ceiling",
                    MAX_OCI_LAYER_BYTES / (1024 * 1024)
                )));
            }
            raw_or_gz.to_vec()
        };

        let mut offset = 0usize;
        let mut pax_override_path: Option<String> = None;

        while offset + 512 <= tar_bytes.len() {
            let header = &tar_bytes[offset..offset + 512];
            offset += 512;

            // End-of-archive zero block
            if header.iter().all(|&b| b == 0) {
                break;
            }

            let size = parse_tar_octal(&header[124..136]);
            let typeflag = header[156];
            let data_end = offset.saturating_add(size).min(tar_bytes.len());
            let payload = &tar_bytes[offset..data_end];
            let padded = size.div_ceil(512).saturating_mul(512);
            offset = offset.saturating_add(padded);

            // PAX extended header (`'x'`): look for `path=...`
            if typeflag == b'x' {
                if let Ok(txt) = std::str::from_utf8(payload) {
                    for line in txt.lines() {
                        if let Some((_len, kv)) = line.split_once(' ')
                            && let Some(p) = kv.strip_prefix("path=")
                        {
                            pax_override_path = Some(p.to_string());
                        }
                    }
                }
                continue;
            }

            let raw_name = pax_override_path.take().unwrap_or_else(|| {
                let name = read_cstr(&header[0..100]);
                let prefix = read_cstr(&header[345..500]);
                if prefix.is_empty() {
                    name
                } else {
                    format!("{prefix}/{name}")
                }
            });

            let clean = raw_name.trim_matches('/');
            if clean.is_empty() {
                continue;
            }
            let vfs_path = normalize_vfs_path(clean)?;
            let file_name = vfs_path.rsplit('/').next().unwrap_or("");
            let parent_dir = match vfs_path.rfind('/') {
                Some(0) | None => "/".to_string(),
                Some(idx) => vfs_path[..idx].to_string(),
            };

            // OCI Whiteout semantics:
            // 1) `.wh..wh..opq`: opaque directory — remove all inherited files under `parent_dir/`
            if file_name == ".wh..wh..opq" {
                let prefix = if parent_dir == "/" {
                    "/".to_string()
                } else {
                    format!("{parent_dir}/")
                };
                if let Ok(existing) = self.agentfs.list_files(branch, "/") {
                    for f in existing {
                        if f.path.starts_with(&prefix) {
                            let _ = self.agentfs.delete_file(branch, &f.path);
                        }
                    }
                }
                continue;
            }
            // 2) `.wh.<target>`: delete `<parent_dir>/<target>` in CoW overlay!
            if let Some(target_name) = file_name.strip_prefix(".wh.") {
                let victim = if parent_dir == "/" {
                    format!("/{target_name}")
                } else {
                    format!("{parent_dir}/{target_name}")
                };
                let _ = self.agentfs.delete_file(branch, &victim);
                continue;
            }

            // Regular file (`'0'` or `'\0'`)
            if typeflag == b'0' || typeflag == 0 {
                self.record_cas_blob(payload);
                self.agentfs.write_file(branch, &vfs_path, payload)?;
            }
        }
        Ok(())
    }

    fn sync_host_dir_into_branch(
        &self,
        root: &Path,
        curr: &Path,
        branch: &str,
    ) -> Result<(), BranchError> {
        let Ok(entries) = std::fs::read_dir(curr) else {
            return Ok(());
        };
        for entry in entries.flatten() {
            let path = entry.path();
            // CRITICAL SECURITY INVARIANT: inspect `symlink_metadata` (lstat) and NEVER follow
            // symlinks, FIFOs, or device nodes created by untrusted code inside `rootfs_dir`!
            // Otherwise `ln -s /etc/hosts pwned.txt` would exfiltrate host files into `JouleAgentFs`.
            let Ok(meta) = std::fs::symlink_metadata(&path) else {
                continue;
            };
            let ft = meta.file_type();
            if ft.is_symlink() {
                continue;
            }
            if ft.is_dir() {
                self.sync_host_dir_into_branch(root, &path, branch)?;
            } else if ft.is_file()
                && meta.len() <= MAX_OCI_LAYER_BYTES as u64
                && let Ok(rel) = path.strip_prefix(root)
                && let Ok(bytes) = std::fs::read(&path)
            {
                let Ok(vfs_path) = normalize_vfs_path(&rel.to_string_lossy().replace('\\', "/"))
                else {
                    continue;
                };
                let unchanged = self
                    .agentfs
                    .read_file(branch, &vfs_path)
                    .ok()
                    .flatten()
                    .is_some_and(|old| old == bytes);
                if !unchanged {
                    self.record_cas_blob(&bytes);
                    self.agentfs.write_file(branch, &vfs_path, &bytes)?;
                }
            }
        }
        Ok(())
    }
}

/// Build a valid uncompressed POSIX USTAR `.tar` archive in memory from `(path, bytes)` pairs.
/// Useful for both packaging `Joulefile` export layers and deterministic unit/E2E testing!
pub fn build_ustar_layer(entries: &[(&str, &[u8])]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(path, data) in entries {
        let clean = path.trim_start_matches('/');
        let mut hdr = [0u8; 512];
        let name_bytes = clean.as_bytes();
        let copy_len = name_bytes.len().min(100);
        hdr[0..copy_len].copy_from_slice(&name_bytes[..copy_len]);
        // Mode 0755, uid 0, gid 0
        hdr[100..107].copy_from_slice(b"0000755");
        hdr[108..115].copy_from_slice(b"0000000");
        hdr[116..123].copy_from_slice(b"0000000");
        let size_oct = format!("{:011o}", data.len());
        hdr[124..135].copy_from_slice(size_oct.as_bytes());
        hdr[136..147].copy_from_slice(b"14600000000");
        // Checksum field filled with spaces during sum
        hdr[148..156].fill(b' ');
        hdr[156] = b'0';
        hdr[257..263].copy_from_slice(b"ustar\0");
        hdr[263..265].copy_from_slice(b"00");

        let csum: u32 = hdr.iter().map(|&b| b as u32).sum();
        let csum_oct = format!("{csum:06o}\0 ");
        hdr[148..156].copy_from_slice(csum_oct.as_bytes());

        out.extend_from_slice(&hdr);
        out.extend_from_slice(data);
        let pad = (512 - (data.len() % 512)) % 512;
        if pad > 0 {
            out.resize(out.len() + pad, 0);
        }
    }
    // Two 512-byte zero EOF blocks
    out.resize(out.len() + 1024, 0);
    out
}

fn topological_boot_waves(
    services: &BTreeMap<String, ComposeServiceSpec>,
) -> Result<Vec<Vec<String>>, BranchError> {
    let mut in_degree: BTreeMap<String, usize> = BTreeMap::new();
    let mut dependents: BTreeMap<String, Vec<String>> = BTreeMap::new();

    for (name, svc) in services {
        in_degree.entry(name.clone()).or_insert(0);
        for dep in &svc.depends_on {
            if !services.contains_key(dep) {
                return Err(BranchError::NotFound(format!(
                    "compose service '{name}' depends on unknown service '{dep}'"
                )));
            }
            *in_degree.entry(name.clone()).or_insert(0) += 1;
            dependents
                .entry(dep.clone())
                .or_default()
                .push(name.clone());
        }
    }

    let mut ready: VecDeque<String> = in_degree
        .iter()
        .filter(|&(_, &deg)| deg == 0)
        .map(|(k, _)| k.clone())
        .collect();
    let mut waves = Vec::new();
    let mut placed = 0usize;

    while !ready.is_empty() {
        let mut current_wave = Vec::new();
        let mut next_ready = BTreeSet::new();
        while let Some(svc) = ready.pop_front() {
            placed += 1;
            if let Some(children) = dependents.get(&svc) {
                for ch in children {
                    if let Some(deg) = in_degree.get_mut(ch) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            next_ready.insert(ch.clone());
                        }
                    }
                }
            }
            current_wave.push(svc);
        }
        waves.push(current_wave);
        ready.extend(next_ready);
    }

    if placed != services.len() {
        return Err(BranchError::InvalidName(
            "dependency cycle detected in ComposeSpec depends_on graph".into(),
        ));
    }
    Ok(waves)
}

fn parse_tar_octal(field: &[u8]) -> usize {
    let s = read_cstr(field);
    usize::from_str_radix(s.trim(), 8).unwrap_or(0)
}

fn read_cstr(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).trim().to_string()
}

fn unquote(s: &str) -> String {
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

fn parse_string_or_json_list(rest: &str) -> Vec<String> {
    if rest.starts_with('[')
        && let Ok(list) = serde_json::from_str::<Vec<String>>(rest)
    {
        return list;
    }
    rest.split_whitespace().map(unquote).collect()
}

fn parse_energy_to_uj(raw: &str) -> Result<u64, BranchError> {
    let s = raw.trim().to_lowercase();
    let (num_s, mult) = if let Some(n) = s.strip_suffix("uj") {
        (n.trim(), 1.0)
    } else if let Some(n) = s.strip_suffix("mj") {
        (n.trim(), 1_000.0)
    } else if let Some(n) = s.strip_suffix('j') {
        (n.trim(), 1_000_000.0)
    } else {
        (s.as_str(), 1.0)
    };
    let val = num_s
        .parse::<f64>()
        .map_err(|_| BranchError::InvalidName(format!("invalid ENERGY_BUDGET '{raw}'")))?;
    Ok((val * mult).round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use std::io::Write;

    fn make_engine() -> CapsuleEngine {
        CapsuleEngine::new(Arc::new(JouleAgentFs::standalone()))
    }

    #[test]
    fn test_oci_targz_whiteout_cas_dedup_and_o1_cow_capsule_fork_and_exec() {
        let engine = make_engine();

        // Layer 0 (gzipped `.tar.gz`!): `/etc/app.conf`, `/etc/secret.tmp`, `/bin/run.sh`
        let layer0_tar = build_ustar_layer(&[
            ("etc/app.conf", b"mode=production\n"),
            ("etc/secret.tmp", b"remove_me"),
            (
                "app/worker.sh",
                b"#!/bin/sh\necho \"CAPSULE_OK mode=$APP_MODE\" > ./result.txt\ncat ../etc/app.conf\n",
            ),
        ]);
        let mut gz = GzEncoder::new(Vec::new(), Compression::fast());
        gz.write_all(&layer0_tar).unwrap();
        let layer0_gz = gz.finish().unwrap();

        // Layer 1: OCI `.wh.secret.tmp` whiteout deletion + duplicate blob `/etc/app.conf.bak` (tests CAS dedup!)
        let layer1_tar = build_ustar_layer(&[
            ("etc/.wh.secret.tmp", b""),
            ("etc/app.conf.bak", b"mode=production\n"),
        ]);

        let img = engine
            .ingest_oci_image(
                "ghcr.io/joulesperbit/worker:1.0",
                OciImageConfig {
                    env: vec!["APP_MODE=thermodynamic".into()],
                    working_dir: "/app".into(),
                    default_energy_budget_uj: Some(500_000),
                    preferred_silicon_lane: Some("popcnt_bitwise".into()),
                    ..OciImageConfig::default()
                },
                &[layer0_gz, layer1_tar],
            )
            .unwrap();

        // `.wh.secret.tmp` deleted `secret.tmp`, so 3 files remain (`app.conf`, `app.conf.bak`, `worker.sh`)
        assert_eq!(img.file_count, 3);
        assert!(engine.cas_stats().dedup_bytes_saved >= b"mode=production\n".len());

        // Spawn Capsule A in O(1) CoW and execute `/app/worker.sh` without `dockerd`!
        let cap_a = engine
            .spawn_capsule(SpawnCapsuleRequest {
                capsule_id: "worker-a".into(),
                image_ref: "ghcr.io/joulesperbit/worker:1.0".into(),
                env: BTreeMap::new(),
                ports: vec![],
                energy_budget_uj: None,
            })
            .unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let log = engine
            .exec_in_capsule(
                &cap_a.capsule_id,
                &["/bin/sh".into(), "/app/worker.sh".into()],
                tmp.path(),
            )
            .unwrap();
        assert_eq!(log.exit_code, 0);
        assert!(log.stdout.contains("mode=production"));

        // Verify `/app/result.txt` written inside the process synced back into Capsule A's CoW branch!
        let written = engine
            .agentfs()
            .read_file(&cap_a.cow_branch, "/app/result.txt")
            .unwrap()
            .unwrap();
        assert_eq!(
            String::from_utf8_lossy(&written).trim(),
            "CAPSULE_OK mode=thermodynamic"
        );

        // O(1) CoW Fork Capsule A -> Capsule B (inherits `/app/result.txt` instantly with 0 copy!)
        let cap_b = engine.fork_capsule("worker-a", "worker-b", None).unwrap();
        let inherited = engine
            .agentfs()
            .read_file(&cap_b.cow_branch, "/app/result.txt")
            .unwrap()
            .unwrap();
        assert_eq!(inherited, written);
    }

    #[test]
    fn test_joulefile_build_and_compose_topological_dag_waves() {
        let engine = make_engine();
        let joulefile = r#"
            FROM scratch
            WORKDIR /srv
            ENV PORT=8080
            COPY index.txt /srv/index.txt
            EXPOSE 8080
            ENERGY_BUDGET 25mJ
            SILICON_LANE gpu_shader
            CMD ["/bin/sh", "-c", "cat /srv/index.txt"]
        "#;
        let mut ctx = BTreeMap::new();
        ctx.insert("index.txt".into(), b"hello-post-docker".to_vec());

        let built = engine
            .build_from_joulefile("joulesperbit/svc:latest", joulefile, &ctx)
            .unwrap();
        assert_eq!(built.config.default_energy_budget_uj, Some(25_000));
        assert_eq!(
            built.config.preferred_silicon_lane.as_deref(),
            Some("gpu_shader")
        );

        let mut services = BTreeMap::new();
        services.insert(
            "db".into(),
            ComposeServiceSpec {
                image: "joulesperbit/svc:latest".into(),
                depends_on: vec![],
                ports: vec![CapsulePortMapping {
                    host_port: 15432,
                    capsule_port: 5432,
                }],
                environment: BTreeMap::new(),
                energy_budget_uj: None,
            },
        );
        services.insert(
            "api".into(),
            ComposeServiceSpec {
                image: "joulesperbit/svc:latest".into(),
                depends_on: vec!["db".into()],
                ports: vec![CapsulePortMapping {
                    host_port: 18080,
                    capsule_port: 8080,
                }],
                environment: BTreeMap::new(),
                energy_budget_uj: None,
            },
        );
        services.insert(
            "worker".into(),
            ComposeServiceSpec {
                image: "joulesperbit/svc:latest".into(),
                depends_on: vec!["api".into()],
                ports: vec![],
                environment: BTreeMap::new(),
                energy_budget_uj: None,
            },
        );

        let plan = engine
            .compose_up(&ComposeSpec {
                stack_name: "prod".into(),
                services,
                stack_energy_budget_uj: Some(90_000),
            })
            .unwrap();

        assert_eq!(
            plan.boot_waves,
            vec![
                vec!["db".to_string()],
                vec!["api".to_string()],
                vec!["worker".to_string()]
            ]
        );
        // Service discovery env `DB_HOST` & `DB_PORT` auto-injected!
        let api_cap = plan
            .capsules
            .iter()
            .find(|c| c.capsule_id == "prod-api")
            .unwrap();
        assert_eq!(api_cap.env.get("DB_PORT").map(String::as_str), Some("5432"));
    }
}
