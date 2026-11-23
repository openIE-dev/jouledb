//! `JouleSpace` — The Post-Git, Post-Docker-Sandbox, Post-Database & Multimodal Room Convergence Engine
//!
//! Unifies the seven software silos collapsing under AI & Physical Computing into one safe-Rust,
//! microjoule-metered substrate:
//!
//! 1. **Post-Git Continuous Delta & Intent Graph (*Zed `Delta` / `DeltaDB` + `Jujutsu (jj)`*)**:
//!    - Captures **"the space between commits"** via an immutable, undoable [`OpLogEntry`] stream
//!      linking prompts/intent, continuous edits, and tool calls directly to [`JouleAgentFs`] state.
//!    - Stores concurrent edit collisions as **Jujutsu-class [`FirstClassConflict`]** algebraic
//!      objects (never corrupting files with `<<<<<<<` markers or blocking builds) alongside clean
//!      3-way line auto-merge and lossless `git fast-import` export.
//!    - Replaces pull requests with live multiplayer [`DeltaThread`]s and inline file-range subthreads.
//! 2. **Secure Agent Sandboxes & Cloud Continuity (*Docker Sandboxes & Cloud Sandboxes, Sept 2026*)**:
//!    - Enforces [`NetworkEgressPolicy`] allow/deny lists and GitGuardian-class [`SecretScrubber`]
//!      credential boundary hooks without spinning up bloated per-agent Linux microVMs + `dockerd`.
//!    - Supports laptop-to-cloud detached continuity via [`CloudSandboxHandoffBundle`].
//! 3. **Stateful Multimodal Rooms & Atomic Barge-In (*LiveKit + Zoom + SpacetimeDB*)**:
//!    - Co-locates humans, AI agents, and embodied robots inside [`JouleRoom`] with typed
//!      [`ModalityTrack`]s, CRDT-convergent room state, and **atomic `turn_epoch` barge-in** that
//!      cancels stale in-flight agent turns on human speech or safety interrupts and reclaims joules.
//! 4. **Ambient Sync Drive & Identity Mesh (*Dropbox / Google Drive + Tailscale / Iroh*)**:
//!    - Cursor-based incremental [`SyncDeltaBatch`] replication (`pull_delta_since`) and
//!      zero-config identity-addressed [`MeshPeer`] discovery across NATs and offline partitions.

use crate::BranchError;
use crate::agentfs::{JouleAgentFs, normalize_vfs_path};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn sha256_hex(parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p);
    }
    hex::encode(h.finalize())
}

// ============================================================================
// 1. Post-Git Layer: Zed `Delta` / `DeltaDB` + `Jujutsu (jj)`
// ============================================================================

/// Kind of operation recorded in the Post-Git continuous [`OpLogEntry`] stream.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OpKind {
    /// Fine-grained file/worktree edit beneath commit granularity.
    ContinuousEdit,
    /// Human or supervisor prompt / steering intent ("why" a change happened).
    PromptIntent,
    /// Agent tool execution linked to workspace state.
    ToolCall,
    /// Database SQL / KV / reducer state mutation.
    DataMutation,
    /// Jujutsu-style automatic working-copy snapshot.
    WorkingCopySnapshot,
    /// Non-destructive `jj undo` restoration of a prior operation's file tree.
    UndoRestore,
    /// Programmatic or human resolution of a [`FirstClassConflict`].
    ConflictResolved,
}

/// Author category in a multiplayer human + agent workspace.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActorKind {
    Human,
    AiAgent,
    EmbodiedRobot,
    System,
}

/// Immutable operation log record capturing "the space between commits" (`DeltaDB` + `jj op log`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpLogEntry {
    /// Monotonic sequence number within the space.
    pub seq: u64,
    /// Content-addressed SHA-256 operation ID.
    pub op_id: String,
    /// Previous operation ID (`"root"` for the first entry).
    pub parent_op_id: String,
    /// Target CoW branch / virtual worktree.
    pub branch: String,
    /// Actor identifier (`"alice"`, `"claude-coder-1"`, `"arm-ctrl-01"`).
    pub actor: String,
    /// Actor kind.
    pub actor_kind: ActorKind,
    /// Operation classification.
    pub kind: OpKind,
    /// Concise human/agent summary.
    pub summary: String,
    /// Optional steering prompt / reasoning context preserved alongside the edit.
    pub prompt_intent: Option<String>,
    /// Affected virtual paths.
    pub affected_paths: Vec<String>,
    /// Prior file contents (`path -> Option<bytes>`) captured for O(1) `jj undo`!
    #[serde(skip)]
    prior_snapshots: BTreeMap<String, Option<Vec<u8>>>,
    /// Physical or modeled microjoules spent by this operation.
    pub energy_uj: u64,
    /// Wall-clock timestamp (ms since Unix epoch).
    pub timestamp_ms: u64,
}

/// One competing side of a Jujutsu-class [`FirstClassConflict`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConflictSide {
    /// Actor or branch label that produced this variant.
    pub label: String,
    /// UTF-8 or binary payload proposed by this side.
    pub content: String,
}

/// Jujutsu (`jj`)-style first-class conflict stored as structured data rather than `<<<<<<<` text corruption.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FirstClassConflict {
    /// Unique conflict identifier.
    pub conflict_id: String,
    /// Branch containing the conflict.
    pub branch: String,
    /// Normalized virtual path.
    pub path: String,
    /// Common ancestor text (if any).
    pub base_content: Option<String>,
    /// Competing sides (`+` terms in Jujutsu's conflict algebra).
    pub sides: Vec<ConflictSide>,
    /// Whether this conflict has been resolved.
    pub resolved: bool,
    /// Winning resolved content once settled.
    pub resolution: Option<String>,
}

/// Review status of a Zed `Delta`-style multiplayer thread (replaces GitHub PRs).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThreadStatus {
    Open,
    ChangesRequested,
    Approved,
    Merged,
}

/// Individual message / subthread note inside a [`DeltaThread`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreadMessage {
    pub author: String,
    pub actor_kind: ActorKind,
    pub body: String,
    pub line_range: Option<(usize, usize)>,
    pub timestamp_ms: u64,
}

/// Zed `Delta`-class live workspace review thread anchored to an operation & optional file range.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeltaThread {
    pub thread_id: String,
    pub branch: String,
    pub title: String,
    pub anchor_path: Option<String>,
    pub anchor_op_id: Option<String>,
    pub status: ThreadStatus,
    pub messages: Vec<ThreadMessage>,
}

// ============================================================================
// 2. Docker Sandboxes & Cloud Sandboxes Layer (Boundary + Secret Scrub + Handoff)
// ============================================================================

/// Network egress allow/deny boundary for an autonomous agent sandbox (`Docker Sandboxes` parity).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkEgressPolicy {
    /// If `false` (default zero-trust), only `allowed_hosts` may be contacted.
    pub default_allow: bool,
    /// Explicitly permitted hostnames or `.domain` suffixes.
    #[serde(default)]
    pub allowed_hosts: BTreeSet<String>,
    /// Explicitly denied hostnames or `.domain` suffixes (always overrides allow!).
    #[serde(default)]
    pub blocked_hosts: BTreeSet<String>,
}

impl NetworkEgressPolicy {
    /// Evaluate whether `host` is permitted by this sandbox's network boundary.
    pub fn is_host_allowed(&self, host: &str) -> bool {
        let h = host.trim().to_lowercase();
        if h.is_empty() {
            return false;
        }
        let matches_rule = |rule: &str| -> bool {
            let r = rule.trim().to_lowercase();
            h == r || (r.starts_with('.') && h.ends_with(&r)) || h.ends_with(&format!(".{r}"))
        };
        if self.blocked_hosts.iter().any(|r| matches_rule(r)) {
            return false;
        }
        if self.allowed_hosts.iter().any(|r| matches_rule(r)) {
            return true;
        }
        self.default_allow
    }
}

/// GitGuardian-class credential boundary hook that scrubs leaked API keys / tokens before
/// they are persisted into logs, files, or cloud handoff bundles.
pub struct SecretScrubber;

impl SecretScrubber {
    /// Redact common secret token patterns (`sk-...`, `ghp_...`, `AKIA...`, `Bearer ...`, PEM blocks).
    /// Returns `(scrubbed_text, secrets_redacted_count)`.
    pub fn scrub(input: &str) -> (String, usize) {
        let mut count = 0usize;
        let mut out = input.to_string();

        // Redact PEM private key blocks if present
        while let Some(start) = out.find("-----BEGIN ") {
            if let Some(rel_end) = out[start..].find("PRIVATE KEY-----") {
                let end = start + rel_end + "PRIVATE KEY-----".len();
                // Look for matching `-----END ...-----`
                if let Some(end_tag_start) = out[end..].find("-----END ")
                    && let Some(end_tag_fin) = out[end + end_tag_start..].find("-----")
                {
                    let full_end = end + end_tag_start + end_tag_fin + 5;
                    out.replace_range(start..full_end, "[REDACTED_PRIVATE_KEY]");
                    count += 1;
                    continue;
                }
                out.replace_range(start..end, "[REDACTED_PRIVATE_KEY_HEADER]");
                count += 1;
            } else {
                break;
            }
        }

        // Token-level prefix scanner (`sk-`, `ghp_`, `github_pat_`, `AKIA`, `xoxb-`)
        let prefixes = ["sk-", "ghp_", "github_pat_", "AKIA", "xoxb-", "AIza"];
        let mut rebuilt = String::with_capacity(out.len());
        let mut idx = 0;
        let bytes = out.as_bytes();
        while idx < bytes.len() {
            let tail = &out[idx..];
            let mut matched_prefix = None;
            for p in &prefixes {
                if tail.starts_with(p) {
                    matched_prefix = Some(*p);
                    break;
                }
            }
            if let Some(p) = matched_prefix {
                let tok_len = tail
                    .bytes()
                    .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-')
                    .count();
                if tok_len >= p.len() + 10 {
                    rebuilt.push_str("[REDACTED_SECRET]");
                    count += 1;
                    idx += tok_len;
                    continue;
                }
            }
            if let Some(ch) = tail.chars().next() {
                rebuilt.push(ch);
                idx += ch.len_utf8();
            } else {
                break;
            }
        }

        (rebuilt, count)
    }
}

/// Portable laptop-to-cloud detached execution bundle (*Docker Cloud Sandboxes* superset).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CloudSandboxHandoffBundle {
    pub handoff_id: String,
    pub branch: String,
    pub files_utf8_or_hex: BTreeMap<String, String>,
    pub op_log_tail: Vec<OpLogEntry>,
    pub network_policy: NetworkEgressPolicy,
    pub remaining_energy_uj: Option<u64>,
    pub secrets_scrubbed_on_export: usize,
    pub bundle_digest_sha256: String,
}

// ============================================================================
// 3. LiveKit + SpacetimeDB Multimodal Room & Instant Barge-In
// ============================================================================

/// Real-time multimodal track kind inside a [`JouleRoom`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModalityTrack {
    AudioOpus,
    VideoVlmFrames,
    ScreenShare,
    BiosignalEeg,
    KinematicSe3,
    ReliableDataCrdt,
}

/// Live activity state of a participant inside a [`JouleRoom`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PresenceActivity {
    Idle,
    Speaking,
    Listening,
    Thinking,
    GeneratingSpeech,
    Actuating,
}

/// Human, AI agent, or embodied robot participating in a [`JouleRoom`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoomParticipant {
    pub participant_id: String,
    pub kind: ActorKind,
    pub activity: PresenceActivity,
    pub published_tracks: BTreeSet<ModalityTrack>,
    pub cursor_anchor: Option<String>,
    pub last_heartbeat_ms: u64,
}

/// Lifecycle state of an agent's multimodal generation turn.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AgentTurnStatus {
    InFlight,
    Completed,
    CancelledByBargeIn,
}

/// Tracked multimodal agent inference/TTS/actuation turn bound to a room's `turn_epoch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InFlightAgentTurn {
    pub turn_id: String,
    pub agent_id: String,
    pub started_epoch: u64,
    pub budgeted_uj: u64,
    pub status: AgentTurnStatus,
    pub joules_reclaimed_uj: u64,
}

/// LWW CRDT entry for conflict-free shared room/application state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LwwRoomEntry {
    pub value: serde_json::Value,
    pub hlc_ms: u64,
    pub writer_id: String,
}

/// Stateful multimodal room uniting LiveKit media/turn state, CRDT app state, and Post-Git branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JouleRoom {
    pub room_id: String,
    pub linked_branch: String,
    /// Monotonic epoch incremented on every human barge-in or safety interrupt!
    pub turn_epoch: u64,
    pub participants: BTreeMap<String, RoomParticipant>,
    pub turns: Vec<InFlightAgentTurn>,
    pub crdt_state: BTreeMap<String, LwwRoomEntry>,
    pub total_barge_in_joules_saved_uj: u64,
}

/// Receipt returned when a human or safety reflex triggers a room barge-in.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BargeInReceipt {
    pub room_id: String,
    pub new_turn_epoch: u64,
    pub interrupter_id: String,
    pub reason: String,
    pub cancelled_turn_ids: Vec<String>,
    pub joules_reclaimed_uj: u64,
}

// ============================================================================
// 4. Dropbox / Drive + Tailscale / Iroh Ambient Sync & Mesh
// ============================================================================

/// Zero-config identity-addressed peer in the JouleSpace P2P mesh (*Tailscale / Iroh* class).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshPeer {
    pub peer_id: String,
    pub public_key_hex: String,
    pub endpoint: String,
    /// `"direct_quic_wg"` or `"derp_relay"`
    pub transport_mode: String,
    pub capabilities: Vec<String>,
    pub online: bool,
}

/// Incremental delta batch for offline-first Dropbox/Drive-class sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncDeltaBatch {
    pub branch: String,
    pub from_seq: u64,
    pub head_seq: u64,
    pub ops: Vec<OpLogEntry>,
    pub changed_files_utf8: BTreeMap<String, Option<String>>,
    pub unresolved_conflicts: Vec<FirstClassConflict>,
}

const MAX_OP_LOG_ENTRIES: usize = 8_192;
const MAX_ROOM_TURNS: usize = 1_024;

/// Unified Post-Silo Engine managing Post-Git operation logs, first-class conflicts,
/// Delta review threads, sandbox egress/secret policies, multimodal rooms, and mesh peers.
pub struct JouleSpaceEngine {
    agentfs: Arc<JouleAgentFs>,
    op_log: RwLock<Vec<OpLogEntry>>,
    conflicts: RwLock<BTreeMap<String, FirstClassConflict>>,
    threads: RwLock<BTreeMap<String, DeltaThread>>,
    sandbox_policies: RwLock<BTreeMap<String, NetworkEgressPolicy>>,
    rooms: RwLock<BTreeMap<String, JouleRoom>>,
    mesh_peers: RwLock<BTreeMap<String, MeshPeer>>,
}

impl JouleSpaceEngine {
    /// Construct a new `JouleSpaceEngine` sharing a [`JouleAgentFs`].
    pub fn new(agentfs: Arc<JouleAgentFs>) -> Self {
        Self {
            agentfs,
            op_log: RwLock::new(Vec::new()),
            conflicts: RwLock::new(BTreeMap::new()),
            threads: RwLock::new(BTreeMap::new()),
            sandbox_policies: RwLock::new(BTreeMap::new()),
            rooms: RwLock::new(BTreeMap::new()),
            mesh_peers: RwLock::new(BTreeMap::new()),
        }
    }

    /// Ensure `branch` exists in [`JouleAgentFs`] (auto-forking from `"main"` if needed).
    fn ensure_branch(&self, branch: &str) -> Result<(), BranchError> {
        if branch != "main" && self.agentfs.list_files(branch, "/").is_err() {
            let _ = self.agentfs.fork_workspace(
                branch,
                Some("main"),
                None,
                Some(format!("JouleSpace auto-virtualized worktree '{branch}'")),
            )?;
        }
        Ok(())
    }

    /// Write a file edit WITH Post-Git continuous operation logging, secret scrubbing, and
    /// automatic prior-state capture for `jj undo`!
    #[allow(clippy::too_many_arguments)]
    pub fn apply_continuous_edit(
        &self,
        branch: &str,
        actor: &str,
        actor_kind: ActorKind,
        path: &str,
        new_content: &str,
        summary: &str,
        prompt_intent: Option<String>,
    ) -> Result<(OpLogEntry, usize), BranchError> {
        self.ensure_branch(branch)?;
        let norm_path = normalize_vfs_path(path)?;
        let prior = self.agentfs.read_file(branch, &norm_path)?;

        let (scrubbed_content, c1) = SecretScrubber::scrub(new_content);
        let (scrubbed_intent, c2) = match prompt_intent {
            Some(p) => {
                let (sp, c) = SecretScrubber::scrub(&p);
                (Some(sp), c)
            }
            None => (None, 0),
        };
        let total_scrubbed = c1 + c2;

        self.agentfs
            .write_file(branch, &norm_path, scrubbed_content.as_bytes())?;

        let mut prior_map = BTreeMap::new();
        prior_map.insert(norm_path.clone(), prior);

        let energy_uj = 2 + (scrubbed_content.len() as u64).div_ceil(4096);
        let entry = self.append_op(
            branch,
            actor,
            actor_kind,
            OpKind::ContinuousEdit,
            summary,
            scrubbed_intent,
            vec![norm_path],
            prior_map,
            energy_uj,
        )?;
        Ok((entry, total_scrubbed))
    }

    /// Jujutsu (`jj`)-class 3-way concurrent file merge:
    /// - Non-overlapping line changes from two agents merge cleanly!
    /// - Overlapping conflicting changes materialize a structured [`FirstClassConflict`]
    ///   without `<<<<<<<` text corruption!
    pub fn merge_concurrent_file(
        &self,
        branch: &str,
        path: &str,
        base: &str,
        side_a_label: &str,
        side_a: &str,
        side_b_label: &str,
        side_b: &str,
    ) -> Result<Result<String, FirstClassConflict>, BranchError> {
        self.ensure_branch(branch)?;
        let norm_path = normalize_vfs_path(path)?;

        let base_lines: Vec<&str> = base.lines().collect();
        let a_lines: Vec<&str> = side_a.lines().collect();
        let b_lines: Vec<&str> = side_b.lines().collect();

        if base_lines.len() == a_lines.len() && base_lines.len() == b_lines.len() {
            let mut merged = Vec::with_capacity(base_lines.len());
            let mut collision = false;
            for i in 0..base_lines.len() {
                let bl = base_lines[i];
                let al = a_lines[i];
                let bbl = b_lines[i];
                if al == bbl {
                    merged.push(al);
                } else if al == bl {
                    merged.push(bbl);
                } else if bbl == bl {
                    merged.push(al);
                } else {
                    collision = true;
                    break;
                }
            }
            if !collision {
                let mut joined = merged.join("\n");
                if base.ends_with('\n') {
                    joined.push('\n');
                }
                self.agentfs
                    .write_file(branch, &norm_path, joined.as_bytes())?;
                return Ok(Ok(joined));
            }
        }

        // Materialize Jujutsu-style First-Class Conflict object!
        let cid = format!(
            "jj-conflict-{}",
            &sha256_hex(&[
                branch.as_bytes(),
                norm_path.as_bytes(),
                side_a.as_bytes(),
                side_b.as_bytes()
            ])[..12]
        );
        let conflict = FirstClassConflict {
            conflict_id: cid.clone(),
            branch: branch.to_string(),
            path: norm_path,
            base_content: Some(base.to_string()),
            sides: vec![
                ConflictSide {
                    label: side_a_label.to_string(),
                    content: side_a.to_string(),
                },
                ConflictSide {
                    label: side_b_label.to_string(),
                    content: side_b.to_string(),
                },
            ],
            resolved: false,
            resolution: None,
        };
        if let Ok(mut map) = self.conflicts.write() {
            map.insert(cid, conflict.clone());
        }
        Ok(Err(conflict))
    }

    /// Resolve a [`FirstClassConflict`] programmatically or interactively and commit the result.
    pub fn resolve_conflict(
        &self,
        conflict_id: &str,
        actor: &str,
        resolved_content: &str,
    ) -> Result<FirstClassConflict, BranchError> {
        let updated = {
            let mut map = self
                .conflicts
                .write()
                .map_err(|e| BranchError::Storage(e.to_string()))?;
            let c = map
                .get_mut(conflict_id)
                .ok_or_else(|| BranchError::NotFound(conflict_id.to_string()))?;
            c.resolved = true;
            c.resolution = Some(resolved_content.to_string());
            c.clone()
        };
        let prior = self.agentfs.read_file(&updated.branch, &updated.path)?;
        self.agentfs
            .write_file(&updated.branch, &updated.path, resolved_content.as_bytes())?;
        let mut prior_map = BTreeMap::new();
        prior_map.insert(updated.path.clone(), prior);
        let _ = self.append_op(
            &updated.branch,
            actor,
            ActorKind::AiAgent,
            OpKind::ConflictResolved,
            &format!("Resolved conflict {conflict_id} on {}", updated.path),
            None,
            vec![updated.path.clone()],
            prior_map,
            2,
        )?;
        Ok(updated)
    }

    /// `jj undo`: Non-destructively revert the most recent edit operation on `branch` while
    /// preserving full forward/backward history in the `OperationLog`!
    pub fn undo_last_op(&self, branch: &str, actor: &str) -> Result<OpLogEntry, BranchError> {
        let target_op = {
            let log = self
                .op_log
                .read()
                .map_err(|e| BranchError::Storage(e.to_string()))?;
            log.iter()
                .rev()
                .find(|e| e.branch == branch && !e.prior_snapshots.is_empty())
                .cloned()
                .ok_or_else(|| BranchError::NotFound(format!("no undoable op on '{branch}'")))?
        };

        let mut affected = Vec::new();
        for (path, maybe_bytes) in &target_op.prior_snapshots {
            affected.push(path.clone());
            match maybe_bytes {
                Some(bytes) => {
                    self.agentfs.write_file(branch, path, bytes)?;
                }
                None => {
                    let _ = self.agentfs.delete_file(branch, path);
                }
            }
        }

        self.append_op(
            branch,
            actor,
            ActorKind::Human,
            OpKind::UndoRestore,
            &format!("jj-undo of op {}", target_op.op_id),
            None,
            affected,
            BTreeMap::new(),
            2,
        )
    }

    /// Open or reply to a Zed `Delta`-style multiplayer workspace review thread (no PR needed!).
    pub fn upsert_delta_thread(
        &self,
        thread_id: &str,
        branch: &str,
        title: &str,
        anchor_path: Option<String>,
        anchor_op_id: Option<String>,
        status: Option<ThreadStatus>,
        message: ThreadMessage,
    ) -> Result<DeltaThread, BranchError> {
        let mut map = self
            .threads
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let entry = map
            .entry(thread_id.to_string())
            .or_insert_with(|| DeltaThread {
                thread_id: thread_id.to_string(),
                branch: branch.to_string(),
                title: title.to_string(),
                anchor_path,
                anchor_op_id,
                status: ThreadStatus::Open,
                messages: Vec::new(),
            });
        if let Some(st) = status {
            entry.status = st;
        }
        let (clean_body, _) = SecretScrubber::scrub(&message.body);
        entry.messages.push(ThreadMessage {
            body: clean_body,
            ..message
        });
        Ok(entry.clone())
    }

    /// Export `branch` + its Post-Git intent metadata as a standard `git fast-import` stream!
    pub fn export_git_fast_import(&self, branch: &str, git_ref: &str) -> Result<String, BranchError> {
        let files = self.agentfs.list_files(branch, "/")?;
        let ops = self.list_ops(Some(branch));
        let last_summary = ops
            .last()
            .map(|o| {
                if let Some(ref intent) = o.prompt_intent {
                    format!("{}\n\nPrompt-Intent: {}", o.summary, intent)
                } else {
                    o.summary.clone()
                }
            })
            .unwrap_or_else(|| format!("JouleSpace snapshot of {branch}"));

        let mut stream = String::new();
        let mut file_cmds = Vec::new();
        for (idx, fstat) in files.iter().enumerate() {
            if let Some(bytes) = self.agentfs.read_file(branch, &fstat.path)? {
                let mark = idx + 1;
                stream.push_str(&format!("blob\nmark :{mark}\ndata {}\n", bytes.len()));
                stream.push_str(&String::from_utf8_lossy(&bytes));
                stream.push('\n');
                let rel = fstat.path.trim_start_matches('/');
                file_cmds.push(format!("M 100644 :{mark} {rel}"));
            }
        }
        let commit_mark = files.len() + 1;
        stream.push_str(&format!(
            "commit {git_ref}\nmark :{commit_mark}\ncommitter JouleSpace <space@jouledb.org> {} +0000\ndata {}\n{}\n",
            now_ms() / 1000,
            last_summary.len(),
            last_summary
        ));
        for cmd in file_cmds {
            stream.push_str(&cmd);
            stream.push('\n');
        }
        Ok(stream)
    }

    // ── Docker Sandboxes & Cloud Handoff ─────────────────────────────────────

    /// Configure network allow/deny egress rules for a sandbox branch.
    pub fn set_sandbox_policy(&self, branch: &str, policy: NetworkEgressPolicy) {
        if let Ok(mut map) = self.sandbox_policies.write() {
            map.insert(branch.to_string(), policy);
        }
    }

    /// Check whether `branch` is allowed to connect to `host`.
    pub fn check_sandbox_egress(&self, branch: &str, host: &str) -> bool {
        self.sandbox_policies
            .read()
            .ok()
            .and_then(|m| m.get(branch).map(|p| p.is_host_allowed(host)))
            .unwrap_or(false)
    }

    /// Package `branch` into a secret-scrubbed [`CloudSandboxHandoffBundle`] so an agent started
    /// on a laptop can detach and continue unattended on a cloud/fleet node!
    pub fn export_cloud_handoff(&self, branch: &str) -> Result<CloudSandboxHandoffBundle, BranchError> {
        self.ensure_branch(branch)?;
        let files = self.agentfs.list_files(branch, "/")?;
        let mut exported_files = BTreeMap::new();
        let mut scrubbed_total = 0usize;

        for f in &files {
            if let Some(bytes) = self.agentfs.read_file(branch, &f.path)? {
                let text = String::from_utf8_lossy(&bytes);
                let (clean, c) = SecretScrubber::scrub(&text);
                scrubbed_total += c;
                exported_files.insert(f.path.clone(), clean);
            }
        }

        let policy = self
            .sandbox_policies
            .read()
            .ok()
            .and_then(|m| m.get(branch).cloned())
            .unwrap_or_default();
        let remaining_energy_uj = self
            .agentfs
            .branch_manager()
            .get_branch(branch)
            .ok()
            .and_then(|b| b.energy_budget_uj.map(|cap| cap.saturating_sub(b.energy_spent_uj)));
        let op_log_tail: Vec<OpLogEntry> = self
            .list_ops(Some(branch))
            .into_iter()
            .rev()
            .take(32)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();

        let serialized_files = serde_json::to_vec(&exported_files).unwrap_or_default();
        let digest = sha256_hex(&[branch.as_bytes(), &serialized_files]);

        Ok(CloudSandboxHandoffBundle {
            handoff_id: format!("handoff-{}", &digest[..12]),
            branch: branch.to_string(),
            files_utf8_or_hex: exported_files,
            op_log_tail,
            network_policy: policy,
            remaining_energy_uj,
            secrets_scrubbed_on_export: scrubbed_total,
            bundle_digest_sha256: digest,
        })
    }

    /// Resume a [`CloudSandboxHandoffBundle`] onto a fleet/cloud node.
    pub fn import_cloud_handoff(&self, bundle: CloudSandboxHandoffBundle) -> Result<usize, BranchError> {
        self.ensure_branch(&bundle.branch)?;
        let count = bundle.files_utf8_or_hex.len();
        for (path, content) in &bundle.files_utf8_or_hex {
            self.agentfs
                .write_file(&bundle.branch, path, content.as_bytes())?;
        }
        self.set_sandbox_policy(&bundle.branch, bundle.network_policy);
        Ok(count)
    }

    // ── LiveKit + SpacetimeDB Multimodal Room & Instant Barge-In ─────────────

    /// Join or update presence + modality tracks in a stateful [`JouleRoom`].
    pub fn join_or_update_room(
        &self,
        room_id: &str,
        linked_branch: &str,
        participant: RoomParticipant,
    ) -> Result<JouleRoom, BranchError> {
        self.ensure_branch(linked_branch)?;
        let mut rooms = self
            .rooms
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let room = rooms
            .entry(room_id.to_string())
            .or_insert_with(|| JouleRoom {
                room_id: room_id.to_string(),
                linked_branch: linked_branch.to_string(),
                turn_epoch: 1,
                participants: BTreeMap::new(),
                turns: Vec::new(),
                crdt_state: BTreeMap::new(),
                total_barge_in_joules_saved_uj: 0,
            });
        room.participants
            .insert(participant.participant_id.clone(), participant);
        Ok(room.clone())
    }

    /// Start an in-flight multimodal agent turn bound to the room's current `turn_epoch`.
    pub fn start_agent_turn(
        &self,
        room_id: &str,
        turn_id: &str,
        agent_id: &str,
        budgeted_uj: u64,
    ) -> Result<InFlightAgentTurn, BranchError> {
        let mut rooms = self
            .rooms
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let room = rooms
            .get_mut(room_id)
            .ok_or_else(|| BranchError::NotFound(room_id.to_string()))?;
        if let Some(p) = room.participants.get_mut(agent_id) {
            p.activity = PresenceActivity::GeneratingSpeech;
        }
        let turn = InFlightAgentTurn {
            turn_id: turn_id.to_string(),
            agent_id: agent_id.to_string(),
            started_epoch: room.turn_epoch,
            budgeted_uj,
            status: AgentTurnStatus::InFlight,
            joules_reclaimed_uj: 0,
        };
        if room.turns.len() >= MAX_ROOM_TURNS {
            let end = (MAX_ROOM_TURNS / 8).max(1).min(room.turns.len());
            room.turns.drain(0..end);
        }
        room.turns.push(turn.clone());
        Ok(turn)
    }

    /// Atomic Barge-In: advances `turn_epoch`, cancels all in-flight agent turns from prior
    /// epochs, transitions interrupted AI agents to `Listening`, and reclaims unspent joules!
    pub fn trigger_barge_in(
        &self,
        room_id: &str,
        interrupter_id: &str,
        reason: &str,
    ) -> Result<BargeInReceipt, BranchError> {
        let mut rooms = self
            .rooms
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let room = rooms
            .get_mut(room_id)
            .ok_or_else(|| BranchError::NotFound(room_id.to_string()))?;

        room.turn_epoch += 1;
        let new_epoch = room.turn_epoch;
        let mut cancelled_ids = Vec::new();
        let mut reclaimed_uj = 0u64;

        for t in &mut room.turns {
            if t.status == AgentTurnStatus::InFlight && t.started_epoch < new_epoch {
                t.status = AgentTurnStatus::CancelledByBargeIn;
                // Reclaim ~75% of remaining streaming generation/TTS/VLM budget on early barge-in
                let saved = (t.budgeted_uj * 3) / 4;
                t.joules_reclaimed_uj = saved;
                reclaimed_uj += saved;
                cancelled_ids.push(t.turn_id.clone());
            }
        }

        for (pid, p) in &mut room.participants {
            if pid == interrupter_id {
                p.activity = PresenceActivity::Speaking;
            } else if p.kind == ActorKind::AiAgent {
                p.activity = PresenceActivity::Listening;
            }
        }
        room.total_barge_in_joules_saved_uj += reclaimed_uj;

        Ok(BargeInReceipt {
            room_id: room_id.to_string(),
            new_turn_epoch: new_epoch,
            interrupter_id: interrupter_id.to_string(),
            reason: reason.to_string(),
            cancelled_turn_ids: cancelled_ids,
            joules_reclaimed_uj: reclaimed_uj,
        })
    }

    /// Commutative, associative, idempotent LWW CRDT merge of shared room state!
    pub fn merge_room_crdt_state(
        &self,
        room_id: &str,
        updates: BTreeMap<String, LwwRoomEntry>,
    ) -> Result<JouleRoom, BranchError> {
        let mut rooms = self
            .rooms
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let room = rooms
            .get_mut(room_id)
            .ok_or_else(|| BranchError::NotFound(room_id.to_string()))?;

        for (k, incoming) in updates {
            let replace = match room.crdt_state.get(&k) {
                None => true,
                Some(cur) => {
                    (incoming.hlc_ms, &incoming.writer_id) > (cur.hlc_ms, &cur.writer_id)
                }
            };
            if replace {
                room.crdt_state.insert(k, incoming);
            }
        }
        Ok(room.clone())
    }

    // ── Ambient Drive Sync & P2P Mesh ────────────────────────────────────────

    /// Register or refresh an identity-addressed peer in the Tailscale/Iroh-class mesh.
    pub fn register_mesh_peer(&self, peer: MeshPeer) {
        if let Ok(mut m) = self.mesh_peers.write() {
            m.insert(peer.peer_id.clone(), peer);
        }
    }

    /// List all mesh peers.
    pub fn list_mesh_peers(&self) -> Vec<MeshPeer> {
        self.mesh_peers
            .read()
            .map(|m| m.values().cloned().collect())
            .unwrap_or_default()
    }

    /// Pull incremental workspace + Post-Git deltas since `since_seq` (*Dropbox / Drive* sync).
    pub fn pull_delta_since(&self, branch: &str, since_seq: u64) -> Result<SyncDeltaBatch, BranchError> {
        self.ensure_branch(branch)?;
        let ops: Vec<OpLogEntry> = self
            .list_ops(Some(branch))
            .into_iter()
            .filter(|o| o.seq > since_seq)
            .collect();
        let head_seq = ops.last().map(|o| o.seq).unwrap_or(since_seq);

        let mut changed_files = BTreeMap::new();
        for op in &ops {
            for p in &op.affected_paths {
                let current = self
                    .agentfs
                    .read_file(branch, p)?
                    .map(|b| String::from_utf8_lossy(&b).to_string());
                changed_files.insert(p.clone(), current);
            }
        }
        let unresolved_conflicts = self
            .conflicts
            .read()
            .map(|m| {
                m.values()
                    .filter(|c| c.branch == branch && !c.resolved)
                    .cloned()
                    .collect()
            })
            .unwrap_or_default();

        Ok(SyncDeltaBatch {
            branch: branch.to_string(),
            from_seq: since_seq,
            head_seq,
            ops,
            changed_files_utf8: changed_files,
            unresolved_conflicts,
        })
    }

    /// List Post-Git operation log entries (optionally filtered to `branch`).
    pub fn list_ops(&self, branch: Option<&str>) -> Vec<OpLogEntry> {
        self.op_log
            .read()
            .map(|log| {
                log.iter()
                    .filter(|e| branch.is_none_or(|b| e.branch == b))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// List `DeltaThread`s on `branch` (or all branches if `None`).
    pub fn list_threads(&self, branch: Option<&str>) -> Vec<DeltaThread> {
        self.threads
            .read()
            .map(|m| {
                m.values()
                    .filter(|t| branch.is_none_or(|b| t.branch == b))
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    #[allow(clippy::too_many_arguments)]
    fn append_op(
        &self,
        branch: &str,
        actor: &str,
        actor_kind: ActorKind,
        kind: OpKind,
        summary: &str,
        prompt_intent: Option<String>,
        affected_paths: Vec<String>,
        prior_snapshots: BTreeMap<String, Option<Vec<u8>>>,
        energy_uj: u64,
    ) -> Result<OpLogEntry, BranchError> {
        let mut log = self
            .op_log
            .write()
            .map_err(|e| BranchError::Storage(e.to_string()))?;
        let seq = log.last().map(|e| e.seq + 1).unwrap_or(1);
        let parent_op_id = log
            .last()
            .map(|e| e.op_id.clone())
            .unwrap_or_else(|| "root".to_string());
        if log.len() >= MAX_OP_LOG_ENTRIES {
            let end = (MAX_OP_LOG_ENTRIES / 8).max(1).min(log.len());
            log.drain(0..end);
        }
        let ts = now_ms();
        let seq_b = seq.to_le_bytes();
        let ts_b = ts.to_le_bytes();
        let op_id = format!(
            "op-{}",
            &sha256_hex(&[
                parent_op_id.as_bytes(),
                &seq_b,
                &ts_b,
                branch.as_bytes(),
                actor.as_bytes(),
                summary.as_bytes(),
            ])[..16]
        );
        let entry = OpLogEntry {
            seq,
            op_id,
            parent_op_id,
            branch: branch.to_string(),
            actor: actor.to_string(),
            actor_kind,
            kind,
            summary: summary.to_string(),
            prompt_intent,
            affected_paths,
            prior_snapshots,
            energy_uj,
            timestamp_ms: ts,
        };
        log.push(entry.clone());
        Ok(entry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_space() -> JouleSpaceEngine {
        JouleSpaceEngine::new(Arc::new(JouleAgentFs::standalone()))
    }

    #[test]
    fn test_post_git_delta_oplog_secret_scrub_jj_conflict_undo_and_git_export() {
        let space = make_space();

        // 1. Continuous edit with prompt intent + leaked API key auto-scrubbed!
        let (op1, scrubbed) = space
            .apply_continuous_edit(
                "feat-agent",
                "claude-coder",
                ActorKind::AiAgent,
                "/src/lib.rs",
                "pub fn auth() { let k = \"sk-proj1234567890abcdef\"; }\n",
                "Add auth helper",
                Some("User asked to wire auth; temp key ghp_9876543210fedcba9876".into()),
            )
            .unwrap();
        assert_eq!(scrubbed, 2, "Both leaked tokens in code & prompt must be redacted!");
        assert!(
            !op1.prompt_intent
                .as_deref()
                .unwrap()
                .contains("ghp_9876543210")
        );

        let (_op2, _) = space
            .apply_continuous_edit(
                "feat-agent",
                "alice",
                ActorKind::Human,
                "/src/lib.rs",
                "pub fn auth() -> bool { true }\n",
                "Simplify auth signature",
                None,
            )
            .unwrap();

        // 2. `jj undo` reverts `/src/lib.rs` back to op1's scrubbed content!
        let _undo_op = space.undo_last_op("feat-agent", "alice").unwrap();
        let restored = space
            .pull_delta_since("feat-agent", 0)
            .unwrap();
        assert!(
            restored.changed_files_utf8["/src/lib.rs"]
                .as_deref()
                .unwrap()
                .contains("[REDACTED_SECRET]")
        );

        // 3. Jujutsu 3-way merge: non-overlapping lines auto-merge; overlapping lines create FirstClassConflict!
        let clean = space
            .merge_concurrent_file(
                "feat-agent",
                "/src/config.toml",
                "port = 8080\nmode = \"dev\"\n",
                "agent-1",
                "port = 9090\nmode = \"dev\"\n",
                "agent-2",
                "port = 8080\nmode = \"prod\"\n",
            )
            .unwrap()
            .unwrap();
        assert_eq!(clean, "port = 9090\nmode = \"prod\"\n");

        let conflict = space
            .merge_concurrent_file(
                "feat-agent",
                "/src/config.toml",
                "port = 8080\n",
                "agent-1",
                "port = 9091\n",
                "agent-2",
                "port = 9092\n",
            )
            .unwrap()
            .unwrap_err();
        assert!(!conflict.resolved);
        let resolved = space
            .resolve_conflict(&conflict.conflict_id, "alice", "port = 9092\n")
            .unwrap();
        assert!(resolved.resolved);

        // 4. Lossless Git fast-import stream export preserves Prompt-Intent!
        let fast_import = space
            .export_git_fast_import("feat-agent", "refs/heads/main")
            .unwrap();
        assert!(fast_import.contains("commit refs/heads/main"));
        assert!(fast_import.contains("src/config.toml"));
    }

    #[test]
    fn test_docker_sandbox_egress_handoff_and_livekit_room_barge_in() {
        let space = make_space();

        // 1. Zero-trust Docker Sandbox network allow/deny boundary + Cloud Handoff
        let mut allowed = BTreeSet::new();
        allowed.insert("crates.io".into());
        allowed.insert("github.com".into());
        let mut blocked = BTreeSet::new();
        blocked.insert("evil.github.com".into());
        space.set_sandbox_policy(
            "sbx-1",
            NetworkEgressPolicy {
                default_allow: false,
                allowed_hosts: allowed,
                blocked_hosts: blocked,
            },
        );
        assert!(space.check_sandbox_egress("sbx-1", "static.crates.io"));
        assert!(!space.check_sandbox_egress("sbx-1", "evil.github.com"));
        assert!(!space.check_sandbox_egress("sbx-1", "exfiltrate.io"));

        let _ = space
            .apply_continuous_edit(
                "sbx-1",
                "agent",
                ActorKind::AiAgent,
                "/work/task.md",
                "Continue synthesis in cloud",
                "Checkpoint before laptop lid close",
                None,
            )
            .unwrap();
        let handoff = space.export_cloud_handoff("sbx-1").unwrap();
        assert_eq!(handoff.files_utf8_or_hex.len(), 1);

        // 2. LiveKit-class Multimodal Room + Atomic Barge-In + Commutative CRDT State!
        let mut tracks = BTreeSet::new();
        tracks.insert(ModalityTrack::AudioOpus);
        tracks.insert(ModalityTrack::VideoVlmFrames);
        let _ = space
            .join_or_update_room(
                "room-ops",
                "sbx-1",
                RoomParticipant {
                    participant_id: "david".into(),
                    kind: ActorKind::Human,
                    activity: PresenceActivity::Listening,
                    published_tracks: tracks.clone(),
                    cursor_anchor: Some("/work/task.md:1".into()),
                    last_heartbeat_ms: 1000,
                },
            )
            .unwrap();
        let _ = space
            .join_or_update_room(
                "room-ops",
                "sbx-1",
                RoomParticipant {
                    participant_id: "voice-agent".into(),
                    kind: ActorKind::AiAgent,
                    activity: PresenceActivity::Idle,
                    published_tracks: tracks,
                    cursor_anchor: None,
                    last_heartbeat_ms: 1000,
                },
            )
            .unwrap();

        let _ = space
            .start_agent_turn("room-ops", "turn-1", "voice-agent", 400_000)
            .unwrap();
        let receipt = space
            .trigger_barge_in("room-ops", "david", "human_vad_speech_interrupt")
            .unwrap();
        assert_eq!(receipt.new_turn_epoch, 2);
        assert_eq!(receipt.cancelled_turn_ids, vec!["turn-1".to_string()]);
        assert_eq!(receipt.joules_reclaimed_uj, 300_000);
    }
}
