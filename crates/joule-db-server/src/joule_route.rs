//! `JouleRoute` 4-Tier Quota Waterfall + `RTK` Context Compressor (`OmniRoute` & `FreeLLMAPI` Parity)
//! and `Harness Super-Adapters` HTTP Surface (`harness-hub` `Qwen3.8-27B-super-adapter` Parity).
//!
//! Unifies:
//! 1. **Encrypted Multi-Account / Multi-Tier Provider Vault**: `BLAKE3` keyed XOF stream cipher +
//!    `HMAC-SHA256` Encrypt-then-MAC authentication (`0` plaintext API keys stored at rest).
//! 2. **4-Tier Smart Waterfall Router**:
//!    - `Tier 0`: LSN-Valid `SemanticResponseCache` (`LangCache` — `0` external tokens, `~95 nJ`).
//!    - `Tier 1`: Sunk-Cost OAuth Subscription Pool (`claude-code-sub`, `codex-sub`, `gemini-cli-sub`, `copilot-sub`)
//!      with round-robin quota balancing.
//!    - `Tier 2`: Local Zero-Marginal-Cost Weight Substrate (`.jsyn` `LittleBit-2 + Super-Adapter` / `Ollama` / `Docker Model Runner`).
//!    - `Tier 3`: Zero-Cost Developer Free-Tier Pool (`groq-free`, `cerebras-free`, `ai-studio-free`,
//!      `openrouter-free`, `github-models-free`) with automatic `HTTP 429` exponential cooldown circuit-breakers.
//!    - `Tier 4`: Metered Sub-Cent & Frontier BYOK APIs (`deepseek`, `openai`, `anthropic`, `gemini`),
//!      strictly bounded by `ThermoCap` (`max_joules_uj` and `max_cost_micro_usd`).
//! 3. **Deterministic `RTK` Context Token Compressor**: Strips ANSI escapes, collapses repetitive
//!    terminal/build log lines, compacts JSON tool-output whitespace, and deduplicates repeated headers
//!    before upstream dispatch while preserving SQL identifiers, code blocks, and line numbers.
//! 4. **Orthogonal Harness Super-Adapter Registry (`/api/v1/synapse/super-adapters/*`)**: Exposes
//!    `HarnessLoRAAdapter` registration, exact factorized subspace orthogonality auditing,
//!    Rank-Concatenated Super-Adapter composition (`combination_type = "cat"`), live `.jsyn`
//!    `LittleBitSVFTensor` bitplane fusion, and `AsyncGRPO` trajectory advantage + `TRL` JSONL export.

use crate::{
    synapse_routes::SynapseServerRuntime,
    ward_openai_routes::{WardOpenAiRuntime, run_thermodynamic_cascade},
};
use axum::{
    Json, Router,
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use hmac::{Hmac, Mac};
use joule_db_weights::harness_super_adapter::{
    HarnessLoRAAdapter, HarnessTrajectoryEpisode, SuperAdapterCombineMode,
    append_super_adapter_to_littlebit, audit_adapter_orthogonality, compose_super_adapter,
    compute_grpo_group_advantages, export_jsonl_for_trl,
};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use std::collections::BTreeMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

type HmacSha256 = Hmac<Sha256>;

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Waterfall routing tier ordered from lowest marginal cost (`Tier 1`) to metered frontier (`Tier 4`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ProviderWatermarkTier {
    /// Tier 1: Sunk-cost monthly subscriptions (Claude Pro/Max, ChatGPT Codex, Gemini Code Assist, Copilot).
    #[serde(alias = "oauth_subscription")]
    OAuthSubscription = 1,
    /// Tier 2: Local hardware inference (`.jsyn` LittleBit-2, Ollama, Docker Model Runner).
    LocalWeightEngine = 2,
    /// Tier 3: Zero-cost developer free tiers (Groq, Cerebras, Google AI Studio, OpenRouter `:free`, GitHub Models).
    FreeTierPool = 3,
    /// Tier 4: Metered sub-cent or frontier APIs (DeepSeek, OpenAI, Anthropic, Gemini).
    MeteredByok = 4,
}

/// Sealed API credential encrypted via `BLAKE3` keyed XOF + `HMAC-SHA256` Encrypt-then-MAC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealedApiKey {
    pub nonce_hex: String,
    pub ciphertext_hex: String,
    pub mac_hex: String,
    pub masked_preview: String,
}

impl SealedApiKey {
    pub fn seal(master_key: &[u8; 32], nonce_seed: u64, plaintext: &str) -> Result<Self, String> {
        let mut nonce = [0u8; 16];
        nonce[..8].copy_from_slice(&nonce_seed.to_le_bytes());
        nonce[8..].copy_from_slice(&(now_ms() ^ 0xA5A5_5A5A_F0F0_0F0F).to_le_bytes());

        let mut xof = blake3::Hasher::new_keyed(master_key);
        xof.update(b"jouledb-vault-stream-v1:");
        xof.update(&nonce);
        let mut reader = xof.finalize_xof();
        let pt = plaintext.as_bytes();
        let mut keystream = vec![0u8; pt.len()];
        reader.fill(&mut keystream);
        let ct: Vec<u8> = pt
            .iter()
            .zip(keystream.iter())
            .map(|(a, b)| a ^ b)
            .collect();

        let mut mac = HmacSha256::new_from_slice(master_key).map_err(|e| e.to_string())?;
        mac.update(b"jouledb-vault-mac-v1:");
        mac.update(&nonce);
        mac.update(&ct);
        let tag = mac.finalize().into_bytes();

        let masked = if plaintext.len() <= 8 {
            "****".to_string()
        } else {
            let prefix = &plaintext[..4.min(plaintext.len())];
            let suffix = &plaintext[plaintext.len().saturating_sub(4)..];
            format!("{prefix}...{suffix}")
        };

        Ok(Self {
            nonce_hex: hex::encode(nonce),
            ciphertext_hex: hex::encode(ct),
            mac_hex: hex::encode(tag),
            masked_preview: masked,
        })
    }

    pub fn unseal(&self, master_key: &[u8; 32]) -> Result<String, String> {
        let nonce = hex::decode(&self.nonce_hex).map_err(|e| e.to_string())?;
        let ct = hex::decode(&self.ciphertext_hex).map_err(|e| e.to_string())?;
        let expected_mac = hex::decode(&self.mac_hex).map_err(|e| e.to_string())?;

        let mut mac = HmacSha256::new_from_slice(master_key).map_err(|e| e.to_string())?;
        mac.update(b"jouledb-vault-mac-v1:");
        mac.update(&nonce);
        mac.update(&ct);
        mac.verify_slice(&expected_mac)
            .map_err(|_| "vault MAC verification failed".to_string())?;

        let mut xof = blake3::Hasher::new_keyed(master_key);
        xof.update(b"jouledb-vault-stream-v1:");
        xof.update(&nonce);
        let mut reader = xof.finalize_xof();
        let mut keystream = vec![0u8; ct.len()];
        reader.fill(&mut keystream);
        let pt: Vec<u8> = ct
            .iter()
            .zip(keystream.iter())
            .map(|(a, b)| a ^ b)
            .collect();
        String::from_utf8(pt).map_err(|e| e.to_string())
    }
}

/// Registered upstream or local provider endpoint inside `JouleRoute`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteProviderEntry {
    pub provider_id: String,
    pub vendor: String,
    pub model_name: String,
    pub base_url: String,
    pub tier: ProviderWatermarkTier,
    pub sealed_key: Option<SealedApiKey>,
    #[serde(default)]
    pub live_http_dispatch: bool,
    pub rpm_limit: u32,
    pub requests_in_window: u32,
    pub window_started_ms: u64,
    pub cost_micro_usd_per_1k_tokens: u64,
    pub estimated_uj_per_token: u64,
    pub consecutive_429s: u32,
    pub cooldown_until_ms: u64,
    pub total_dispatches: u64,
    pub total_tokens_served: u64,
    pub total_cost_micro_usd: u64,
    pub total_energy_uj: u64,
}

/// Sanitized public view of a registered provider (never exposes `ciphertext_hex` or raw keys).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicProviderStatus {
    pub provider_id: String,
    pub vendor: String,
    pub model_name: String,
    pub tier: ProviderWatermarkTier,
    pub key_preview: String,
    pub live_http_dispatch: bool,
    pub circuit_open: bool,
    pub cooldown_remaining_ms: u64,
    pub rpm_remaining: u32,
    pub cost_micro_usd_per_1k_tokens: u64,
    pub total_dispatches: u64,
    pub total_cost_micro_usd: u64,
    pub total_energy_uj: u64,
}

/// Receipt from deterministic `RTK` (Rust Token Killer) prompt/context compression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RtkCompressionReceipt {
    pub raw_chars: usize,
    pub compressed_chars: usize,
    pub raw_est_tokens: usize,
    pub compressed_est_tokens: usize,
    pub tokens_saved: usize,
    pub reduction_pct: f64,
    pub compressed_text: String,
}

/// Strip ANSI escape sequences (`\x1b[...m`) without regex overhead.
fn strip_ansi_codes(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '\x1b' && chars.peek() == Some(&'[') {
            let _ = chars.next();
            for c in chars.by_ref() {
                if c.is_ascii_alphabetic() {
                    break;
                }
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Deterministic `RTK`-style context compressor for noisy coding-agent terminal logs, build output,
/// and verbose JSON tool payloads. Preserves fenced code blocks, SQL identifiers, and error lines.
pub fn compress_context_rtk(raw: &str) -> RtkCompressionReceipt {
    let stripped = strip_ansi_codes(raw);
    let mut out_lines: Vec<String> = Vec::new();
    let mut prev_norm = String::new();
    let mut dup_count = 0usize;
    let mut in_code_fence = false;
    let mut seen_decorative_rules = 0usize;

    for line in stripped.lines() {
        let trimmed_end = line.trim_end();
        let trimmed = trimmed_end.trim();

        if trimmed.starts_with("```") {
            if dup_count > 0 {
                out_lines.push(format!("  [... {dup_count} duplicate lines elided ...]"));
                dup_count = 0;
            }
            in_code_fence = !in_code_fence;
            out_lines.push(trimmed_end.to_string());
            prev_norm.clear();
            continue;
        }

        if in_code_fence {
            out_lines.push(trimmed_end.to_string());
            continue;
        }

        // Skip redundant blank runs
        if trimmed.is_empty() {
            if out_lines.last().is_some_and(|l| !l.is_empty()) {
                out_lines.push(String::new());
            }
            continue;
        }

        // Elide repetitive ASCII box/divider banners (`====`, `----`, `****`)
        if trimmed.len() >= 12
            && trimmed
                .chars()
                .all(|c| matches!(c, '=' | '-' | '*' | '_' | '#' | '─' | '═'))
        {
            seen_decorative_rules += 1;
            if seen_decorative_rules > 1 {
                continue;
            }
        }

        // Drop noisy cargo/npm progress chatter that carries zero diagnostic signal
        if (trimmed.starts_with("Downloading ")
            || trimmed.starts_with("Downloaded ")
            || trimmed.starts_with("Compiling ")
            || trimmed.starts_with("Fetching "))
            && !trimmed.contains("error")
            && !trimmed.contains("warning")
        {
            dup_count += 1;
            continue;
        }

        // Compact single-line pretty JSON spaces outside string literals if line looks like JSON key-val
        let candidate = if (trimmed.starts_with('"') && trimmed.contains("\": "))
            || trimmed == "{"
            || trimmed == "}"
            || trimmed == "},"
        {
            trimmed.replace("\": ", "\":")
        } else {
            trimmed_end.to_string()
        };

        if candidate == prev_norm {
            dup_count += 1;
        } else {
            if dup_count > 0 {
                out_lines.push(format!("  [... {dup_count} repetitive lines elided ...]"));
                dup_count = 0;
            }
            prev_norm = candidate.clone();
            out_lines.push(candidate);
        }
    }
    if dup_count > 0 {
        out_lines.push(format!("  [... {dup_count} repetitive lines elided ...]"));
    }

    let compressed = out_lines.join("\n").trim().to_string();
    let raw_chars = raw.len();
    let comp_chars = compressed.len();
    let raw_tokens = raw_chars.div_ceil(4);
    let comp_tokens = comp_chars.div_ceil(4);
    let saved = raw_tokens.saturating_sub(comp_tokens);
    let pct = if raw_chars == 0 {
        0.0
    } else {
        ((raw_chars.saturating_sub(comp_chars)) as f64 / (raw_chars as f64)) * 100.0
    };

    RtkCompressionReceipt {
        raw_chars,
        compressed_chars: comp_chars,
        raw_est_tokens: raw_tokens,
        compressed_est_tokens: comp_tokens,
        tokens_saved: saved,
        reduction_pct: pct,
        compressed_text: compressed,
    }
}

#[derive(Default)]
struct RouterState {
    providers: BTreeMap<String, RouteProviderEntry>,
    round_robin_cursor: usize,
    nonce_counter: u64,
    total_saved_tokens_rtk: u64,
    total_cache_hits: u64,
    total_failovers: u64,
}

/// Shared runtime for `JouleRoute` + `SuperAdapterRegistry`.
pub struct JouleRouteRuntime {
    master_vault_key: [u8; 32],
    state: RwLock<RouterState>,
    adapters: RwLock<BTreeMap<String, HarnessLoRAAdapter>>,
    ward_rt: Arc<WardOpenAiRuntime>,
    synapse_rt: Arc<SynapseServerRuntime>,
}

impl JouleRouteRuntime {
    pub fn new(
        ward_rt: Arc<WardOpenAiRuntime>,
        synapse_rt: Arc<SynapseServerRuntime>,
        seed: &str,
    ) -> Arc<Self> {
        let master_vault_key = blake3::derive_key("jouledb-v017-router-key-vault", seed.as_bytes());
        let mut init_state = RouterState::default();
        // Register default Tier-2 local `.jsyn` provider so out-of-the-box zero-key routing works immediately
        init_state.providers.insert(
            "local-joulesynapse-jsyn".to_string(),
            RouteProviderEntry {
                provider_id: "local-joulesynapse-jsyn".to_string(),
                vendor: "jouledb".to_string(),
                model_name: "jouledb-cascade-v1".to_string(),
                base_url: "inproc://synapse".to_string(),
                tier: ProviderWatermarkTier::LocalWeightEngine,
                sealed_key: None,
                live_http_dispatch: false,
                rpm_limit: 100_000,
                requests_in_window: 0,
                window_started_ms: now_ms(),
                cost_micro_usd_per_1k_tokens: 0,
                estimated_uj_per_token: 12,
                consecutive_429s: 0,
                cooldown_until_ms: 0,
                total_dispatches: 0,
                total_tokens_served: 0,
                total_cost_micro_usd: 0,
                total_energy_uj: 0,
            },
        );

        Arc::new(Self {
            master_vault_key,
            state: RwLock::new(init_state),
            adapters: RwLock::new(BTreeMap::new()),
            ward_rt,
            synapse_rt,
        })
    }

    /// Register or update a provider in the 4-tier waterfall with sealed API key storage.
    #[allow(clippy::too_many_arguments)]
    pub fn register_provider(
        &self,
        provider_id: String,
        vendor: String,
        model_name: String,
        base_url: String,
        tier: ProviderWatermarkTier,
        api_key: Option<&str>,
        rpm_limit: u32,
        cost_micro_usd_per_1k_tokens: u64,
        estimated_uj_per_token: u64,
    ) -> Result<PublicProviderStatus, String> {
        self.register_provider_ext(
            provider_id,
            vendor,
            model_name,
            base_url,
            tier,
            api_key,
            false,
            rpm_limit,
            cost_micro_usd_per_1k_tokens,
            estimated_uj_per_token,
        )
    }

    /// Register or update a provider with optional live async HTTP upstream dispatch (`live_http_dispatch`).
    #[allow(clippy::too_many_arguments)]
    pub fn register_provider_ext(
        &self,
        provider_id: String,
        vendor: String,
        model_name: String,
        base_url: String,
        tier: ProviderWatermarkTier,
        api_key: Option<&str>,
        live_http_dispatch: bool,
        rpm_limit: u32,
        cost_micro_usd_per_1k_tokens: u64,
        estimated_uj_per_token: u64,
    ) -> Result<PublicProviderStatus, String> {
        let mut st = match self.state.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        st.nonce_counter += 1;
        let nonce_seed = st.nonce_counter;
        let sealed = match api_key.filter(|s| !s.trim().is_empty()) {
            Some(raw) => Some(SealedApiKey::seal(&self.master_vault_key, nonce_seed, raw)?),
            None => None,
        };
        let now = now_ms();
        let entry = RouteProviderEntry {
            provider_id: provider_id.clone(),
            vendor,
            model_name,
            base_url,
            tier,
            sealed_key: sealed,
            live_http_dispatch,
            rpm_limit: rpm_limit.max(1),
            requests_in_window: 0,
            window_started_ms: now,
            cost_micro_usd_per_1k_tokens,
            estimated_uj_per_token: estimated_uj_per_token.max(1),
            consecutive_429s: 0,
            cooldown_until_ms: 0,
            total_dispatches: 0,
            total_tokens_served: 0,
            total_cost_micro_usd: 0,
            total_energy_uj: 0,
        };
        let status = Self::to_public_status(&entry, now);
        st.providers.insert(provider_id, entry);
        Ok(status)
    }

    fn to_public_status(e: &RouteProviderEntry, now: u64) -> PublicProviderStatus {
        let circuit_open = e.cooldown_until_ms > now;
        let cooldown_rem = e.cooldown_until_ms.saturating_sub(now);
        let rpm_rem = if now.saturating_sub(e.window_started_ms) >= 60_000 {
            e.rpm_limit
        } else {
            e.rpm_limit.saturating_sub(e.requests_in_window)
        };
        PublicProviderStatus {
            provider_id: e.provider_id.clone(),
            vendor: e.vendor.clone(),
            model_name: e.model_name.clone(),
            tier: e.tier,
            key_preview: e
                .sealed_key
                .as_ref()
                .map(|k| k.masked_preview.clone())
                .unwrap_or_else(|| "none (keyless/local)".to_string()),
            live_http_dispatch: e.live_http_dispatch,
            circuit_open,
            cooldown_remaining_ms: cooldown_rem,
            rpm_remaining: rpm_rem,
            cost_micro_usd_per_1k_tokens: e.cost_micro_usd_per_1k_tokens,
            total_dispatches: e.total_dispatches,
            total_cost_micro_usd: e.total_cost_micro_usd,
            total_energy_uj: e.total_energy_uj,
        }
    }

    /// Trip exponential `HTTP 429` cooldown circuit breaker (`2^k * 1_000 ms`, capped at `60_000 ms`).
    pub fn report_rate_limit_429(&self, provider_id: &str) -> Result<PublicProviderStatus, String> {
        let mut st = match self.state.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = now_ms();
        st.total_failovers += 1;
        let entry = st
            .providers
            .get_mut(provider_id)
            .ok_or_else(|| format!("provider '{provider_id}' not found"))?;
        entry.consecutive_429s = entry.consecutive_429s.saturating_add(1);
        let shift = entry.consecutive_429s.min(6);
        let backoff_ms = (1_000u64 << shift).min(60_000);
        entry.cooldown_until_ms = now.saturating_add(backoff_ms);
        Ok(Self::to_public_status(entry, now))
    }

    /// Select the best healthy, ThermoCap-compliant provider along the 4-tier waterfall
    /// (`Tier 1 OAuthSub -> Tier 2 Local -> Tier 3 FreePool -> Tier 4 MeteredByok`) with round-robin
    /// balancing inside each tier.
    pub fn select_waterfall_provider(
        &self,
        est_tokens: u64,
        max_cost_micro_usd: Option<u64>,
        max_energy_uj: Option<u64>,
        preferred_tier_max: Option<ProviderWatermarkTier>,
    ) -> Result<(RouteProviderEntry, Option<String>, Vec<String>), String> {
        let mut st = match self.state.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let now = now_ms();
        let mut skipped_reasons = Vec::new();

        let tiers = [
            ProviderWatermarkTier::OAuthSubscription,
            ProviderWatermarkTier::LocalWeightEngine,
            ProviderWatermarkTier::FreeTierPool,
            ProviderWatermarkTier::MeteredByok,
        ];

        let cursor = st.round_robin_cursor;
        st.round_robin_cursor = st.round_robin_cursor.wrapping_add(1);

        for tier in tiers {
            if let Some(max_t) = preferred_tier_max
                && tier > max_t
            {
                continue;
            }
            let tier_ids: Vec<String> = st
                .providers
                .values()
                .filter(|p| p.tier == tier)
                .map(|p| p.provider_id.clone())
                .collect();
            if tier_ids.is_empty() {
                continue;
            }

            for offset in 0..tier_ids.len() {
                let id = &tier_ids[(cursor + offset) % tier_ids.len()];
                let Some(entry) = st.providers.get_mut(id) else {
                    continue;
                };
                if now.saturating_sub(entry.window_started_ms) >= 60_000 {
                    entry.window_started_ms = now;
                    entry.requests_in_window = 0;
                }
                if entry.cooldown_until_ms > now {
                    skipped_reasons.push(format!("{id}:circuit_open_429"));
                    continue;
                }
                if entry.requests_in_window >= entry.rpm_limit {
                    skipped_reasons.push(format!("{id}:rpm_exhausted"));
                    continue;
                }
                let est_cost =
                    (est_tokens.saturating_mul(entry.cost_micro_usd_per_1k_tokens)).div_ceil(1_000);
                if let Some(cap_usd) = max_cost_micro_usd
                    && est_cost > cap_usd
                {
                    skipped_reasons.push(format!("{id}:cost_{est_cost}_exceeds_{cap_usd}"));
                    continue;
                }
                let est_uj = est_tokens.saturating_mul(entry.estimated_uj_per_token);
                if let Some(cap_uj) = max_energy_uj
                    && est_uj > cap_uj
                {
                    skipped_reasons.push(format!("{id}:energy_{est_uj}uj_exceeds_{cap_uj}uj"));
                    continue;
                }

                // Commit selection & update accounting
                entry.requests_in_window += 1;
                entry.consecutive_429s = 0;
                entry.total_dispatches += 1;
                entry.total_tokens_served += est_tokens;
                entry.total_cost_micro_usd += est_cost;
                entry.total_energy_uj += est_uj;

                let unsealed = match &entry.sealed_key {
                    Some(sk) => Some(sk.unseal(&self.master_vault_key)?),
                    None => None,
                };
                return Ok((entry.clone(), unsealed, skipped_reasons));
            }
        }

        Err(format!(
            "all waterfall tiers exhausted or blocked by ThermoCap ({})",
            skipped_reasons.join(", ")
        ))
    }
}

// ============================================================================
// HTTP DTOs & Handlers
// ============================================================================

#[derive(Debug, Deserialize)]
struct RegisterProviderReq {
    provider_id: String,
    vendor: String,
    model_name: String,
    #[serde(default = "default_openai_url")]
    base_url: String,
    tier: ProviderWatermarkTier,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    live_http_dispatch: bool,
    #[serde(default = "default_rpm")]
    rpm_limit: u32,
    #[serde(default)]
    cost_micro_usd_per_1k_tokens: u64,
    #[serde(default = "default_uj_per_tok")]
    estimated_uj_per_token: u64,
}

fn default_openai_url() -> String {
    "https://api.openai.com/v1".to_string()
}
fn default_rpm() -> u32 {
    60
}
fn default_uj_per_tok() -> u64 {
    50
}

#[derive(Debug, Deserialize)]
struct RtkCompressReq {
    text: String,
}

#[derive(Debug, Deserialize)]
struct RouterChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct RouterCompletionReq {
    #[serde(default = "default_cascade_model")]
    model: String,
    messages: Vec<RouterChatMessage>,
    #[serde(default = "default_true")]
    apply_rtk_compression: bool,
    #[serde(default)]
    max_cost_micro_usd: Option<u64>,
    #[serde(default)]
    max_energy_uj: Option<u64>,
    #[serde(default)]
    max_tier: Option<ProviderWatermarkTier>,
    #[serde(default)]
    bypass_lang_cache: bool,
}

fn default_cascade_model() -> String {
    "jouledb-cascade-v1".to_string()
}
fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct RegisterSuperAdapterReq {
    adapter_id: String,
    harness_name: String,
    #[serde(default = "default_target_tensor")]
    target_tensor: String,
    rows: usize,
    cols: usize,
    rank: usize,
    #[serde(default = "default_lora_alpha")]
    lora_alpha: f32,
    #[serde(default)]
    lora_a: Option<Vec<f32>>,
    #[serde(default)]
    lora_b: Option<Vec<f32>>,
    #[serde(default = "default_seed")]
    seed: u64,
}

fn default_target_tensor() -> String {
    "model.layers.0.self_attn.q_proj.weight".to_string()
}
fn default_lora_alpha() -> f32 {
    64.0
}
fn default_seed() -> u64 {
    42
}

#[derive(Debug, Deserialize)]
struct AuditSuperAdaptersReq {
    adapter_ids: Vec<String>,
    #[serde(default = "default_cos_thresh")]
    cosine_threshold: f64,
}

fn default_cos_thresh() -> f64 {
    0.05
}

#[derive(Debug, Deserialize)]
struct ComposeSuperAdapterReq {
    super_adapter_id: String,
    adapter_ids: Vec<String>,
    #[serde(default)]
    weights: Option<Vec<f32>>,
    #[serde(default = "default_combine_mode")]
    mode: SuperAdapterCombineMode,
    #[serde(default = "default_cos_thresh")]
    cosine_threshold: f64,
    /// Optional name of a live `.jsyn` `LittleBitSVFTensor` in `SynapseServerRuntime` to fuse
    /// the composed Super-Adapter bitplanes onto!
    #[serde(default)]
    fuse_into_littlebit_tensor: Option<String>,
}

fn default_combine_mode() -> SuperAdapterCombineMode {
    SuperAdapterCombineMode::RankConcat
}

#[derive(Debug, Deserialize)]
struct GrpoStepReq {
    episodes: Vec<HarnessTrajectoryEpisode>,
    #[serde(default)]
    update_adapter_id: Option<String>,
    #[serde(default)]
    sample_input: Option<Vec<f32>>,
    #[serde(default)]
    sample_target_delta: Option<Vec<f32>>,
    #[serde(default = "default_lr")]
    learning_rate: f32,
}

fn default_lr() -> f32 {
    0.05
}

/// Build the Axum router for `JouleRoute` (`/api/v1/router/*`, `/v1/router/chat/completions`)
/// and `Harness Super-Adapters` (`/api/v1/synapse/super-adapters/*`).
pub fn joule_route_and_super_adapter_router(rt: Arc<JouleRouteRuntime>) -> Router {
    Router::new()
        .route(
            "/api/v1/router/providers/register",
            post(handle_register_provider),
        )
        .route("/api/v1/router/providers", get(handle_list_providers))
        .route(
            "/api/v1/router/providers/{id}/report-429",
            post(handle_report_429),
        )
        .route("/api/v1/router/rtk/compress", post(handle_rtk_compress))
        .route(
            "/api/v1/router/completions",
            post(handle_router_completions),
        )
        .route(
            "/v1/router/chat/completions",
            post(handle_router_completions),
        )
        .route("/v1/router/rerank", post(handle_rerank))
        .route("/api/v1/router/rerank", post(handle_rerank))
        .route(
            "/api/v1/synapse/super-adapters/register",
            post(handle_register_super_adapter),
        )
        .route(
            "/api/v1/synapse/super-adapters",
            get(handle_list_super_adapters),
        )
        .route(
            "/api/v1/synapse/super-adapters/audit",
            post(handle_audit_super_adapters),
        )
        .route(
            "/api/v1/synapse/super-adapters/compose",
            post(handle_compose_super_adapter),
        )
        .route(
            "/api/v1/synapse/super-adapters/grpo-step",
            post(handle_grpo_step),
        )
        .with_state(rt)
}

async fn handle_register_provider(
    State(rt): State<Arc<JouleRouteRuntime>>,
    Json(req): Json<RegisterProviderReq>,
) -> impl IntoResponse {
    match rt.register_provider_ext(
        req.provider_id,
        req.vendor,
        req.model_name,
        req.base_url,
        req.tier,
        req.api_key.as_deref(),
        req.live_http_dispatch,
        req.rpm_limit,
        req.cost_micro_usd_per_1k_tokens,
        req.estimated_uj_per_token,
    ) {
        Ok(status) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "provider": status })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e })),
        )
            .into_response(),
    }
}

async fn handle_list_providers(State(rt): State<Arc<JouleRouteRuntime>>) -> impl IntoResponse {
    let st = match rt.state.read() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let now = now_ms();
    let list: Vec<PublicProviderStatus> = st
        .providers
        .values()
        .map(|e| JouleRouteRuntime::to_public_status(e, now))
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "ok": true,
            "providers": list,
            "total_saved_tokens_rtk": st.total_saved_tokens_rtk,
            "total_cache_hits": st.total_cache_hits,
            "total_failovers": st.total_failovers,
        })),
    )
}

async fn handle_report_429(
    State(rt): State<Arc<JouleRouteRuntime>>,
    Path(id): Path<String>,
) -> impl IntoResponse {
    match rt.report_rate_limit_429(&id) {
        Ok(status) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "provider": status })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "ok": false, "error": e })),
        )
            .into_response(),
    }
}

async fn handle_rtk_compress(Json(req): Json<RtkCompressReq>) -> impl IntoResponse {
    let rec = compress_context_rtk(&req.text);
    (StatusCode::OK, Json(rec))
}

async fn handle_router_completions(
    State(rt): State<Arc<JouleRouteRuntime>>,
    Json(req): Json<RouterCompletionReq>,
) -> impl IntoResponse {
    let joined_prompt = req
        .messages
        .iter()
        .map(|m| format!("{}: {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n");

    let rtk = if req.apply_rtk_compression {
        let r = compress_context_rtk(&joined_prompt);
        if let Ok(mut st) = rt.state.write() {
            st.total_saved_tokens_rtk += r.tokens_saved as u64;
        }
        r
    } else {
        RtkCompressionReceipt {
            raw_chars: joined_prompt.len(),
            compressed_chars: joined_prompt.len(),
            raw_est_tokens: joined_prompt.len().div_ceil(4),
            compressed_est_tokens: joined_prompt.len().div_ceil(4),
            tokens_saved: 0,
            reduction_pct: 0.0,
            compressed_text: joined_prompt,
        }
    };

    // Tier 0: Check LSN-valid `SemanticResponseCache` (`LangCache`) first!
    if !req.bypass_lang_cache {
        let hit = {
            let mut lc = match rt.ward_rt.swot_rt.lang_cache.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            lc.get("default", "main", 0, &rtk.compressed_text, 0.94)
        };
        if let Some((cached_val, _sim, _saved_uj)) = hit {
            if let Ok(mut st) = rt.state.write() {
                st.total_cache_hits += 1;
            }
            let text = cached_val
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("[LangCache Hit]")
                .to_string();
            return (
                StatusCode::OK,
                Json(serde_json::json!({
                    "id": format!("chatcmpl-jroute-cache-{}", now_ms()),
                    "object": "chat.completion",
                    "model": req.model,
                    "choices": [{
                        "index": 0,
                        "message": { "role": "assistant", "content": text },
                        "finish_reason": "stop"
                    }],
                    "joule_route": {
                        "selected_tier": "tier0_lsn_lang_cache",
                        "selected_provider": "inproc-langcache",
                        "rtk_tokens_saved": rtk.tokens_saved,
                        "rtk_reduction_pct": rtk.reduction_pct,
                        "cost_micro_usd": 0,
                        "energy_uj": 1,
                        "failover_skipped": []
                    }
                })),
            )
                .into_response();
        }
    }

    let est_tokens = (rtk.compressed_est_tokens as u64).saturating_add(32);
    let mut accumulated_skips: Vec<String> = Vec::new();
    let http_client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(10))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    for _attempt in 0..6 {
        let (selected, unsealed_key, skipped) = match rt.select_waterfall_provider(
            est_tokens,
            req.max_cost_micro_usd,
            req.max_energy_uj,
            req.max_tier,
        ) {
            Ok(res) => res,
            Err(e) => {
                let mut detail = e;
                if !accumulated_skips.is_empty() {
                    detail = format!("{detail}; live_failovers={}", accumulated_skips.join(", "));
                }
                return (
                    StatusCode::TOO_MANY_REQUESTS,
                    Json(serde_json::json!({
                        "ok": false,
                        "error": detail,
                        "sqlstate": "53200",
                        "failover_skipped": accumulated_skips,
                    })),
                )
                    .into_response();
            }
        };
        for s in skipped {
            if !accumulated_skips.contains(&s) {
                accumulated_skips.push(s);
            }
        }

        // Live async OpenAI-compatible HTTP dispatch (`live_http_dispatch == true`):
        // Automatically catches `HTTP 429` / `5xx` / transport errors, trips the exponential
        // circuit breaker on `selected.provider_id`, and loops to the next healthy waterfall tier!
        if selected.live_http_dispatch
            && (selected.base_url.starts_with("http://")
                || selected.base_url.starts_with("https://"))
        {
            let endpoint = format!(
                "{}/chat/completions",
                selected.base_url.trim_end_matches('/')
            );
            let mut req_builder = http_client.post(&endpoint).json(&serde_json::json!({
                "model": selected.model_name,
                "messages": [{ "role": "user", "content": rtk.compressed_text }],
            }));
            if let Some(ref bearer) = unsealed_key {
                req_builder = req_builder.bearer_auth(bearer);
            }

            match req_builder.send().await {
                Ok(upstream_resp) => {
                    let st_code = upstream_resp.status();
                    if st_code == reqwest::StatusCode::TOO_MANY_REQUESTS
                        || st_code.is_server_error()
                    {
                        let _ = rt.report_rate_limit_429(&selected.provider_id);
                        accumulated_skips.push(format!(
                            "{}:live_http_{}",
                            selected.provider_id,
                            st_code.as_u16()
                        ));
                        continue;
                    }
                    if !st_code.is_success() {
                        let body_txt = upstream_resp.text().await.unwrap_or_default();
                        return (
                            StatusCode::BAD_GATEWAY,
                            Json(serde_json::json!({
                                "ok": false,
                                "error": format!("upstream '{}' returned HTTP {}: {}", selected.provider_id, st_code.as_u16(), body_txt)
                            })),
                        )
                            .into_response();
                    }

                    let payload: serde_json::Value = match upstream_resp.json().await {
                        Ok(v) => v,
                        Err(e) => {
                            let _ = rt.report_rate_limit_429(&selected.provider_id);
                            accumulated_skips
                                .push(format!("{}:invalid_json_{e}", selected.provider_id));
                            continue;
                        }
                    };
                    let reply_text = payload
                        .get("choices")
                        .and_then(|c| c.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|c0| c0.get("message"))
                        .and_then(|m| m.get("content"))
                        .and_then(|s| s.as_str())
                        .unwrap_or("[Upstream OK]")
                        .to_string();

                    if !req.bypass_lang_cache {
                        let mut lc = match rt.ward_rt.swot_rt.lang_cache.write() {
                            Ok(g) => g,
                            Err(p) => p.into_inner(),
                        };
                        lc.put(
                            "default",
                            "main",
                            0,
                            &rtk.compressed_text,
                            serde_json::json!({ "text": reply_text }),
                            est_tokens.saturating_mul(selected.estimated_uj_per_token),
                        );
                    }

                    let est_cost = (est_tokens
                        .saturating_mul(selected.cost_micro_usd_per_1k_tokens))
                    .div_ceil(1_000);
                    let energy_uj = est_tokens
                        .saturating_mul(selected.estimated_uj_per_token)
                        .max(1);

                    return (
                        StatusCode::OK,
                        Json(serde_json::json!({
                            "id": format!("chatcmpl-jroute-http-{}", now_ms()),
                            "object": "chat.completion",
                            "model": selected.model_name,
                            "choices": [{
                                "index": 0,
                                "message": { "role": "assistant", "content": reply_text },
                                "finish_reason": "stop"
                            }],
                            "joule_route": {
                                "selected_tier": selected.tier,
                                "selected_provider": selected.provider_id,
                                "vendor": selected.vendor,
                                "dispatch_mode": "live_http",
                                "rtk_tokens_saved": rtk.tokens_saved,
                                "rtk_reduction_pct": rtk.reduction_pct,
                                "cost_micro_usd": est_cost,
                                "energy_uj": energy_uj,
                                "failover_skipped": accumulated_skips,
                            }
                        })),
                    )
                        .into_response();
                }
                Err(_) => {
                    let _ = rt.report_rate_limit_429(&selected.provider_id);
                    accumulated_skips
                        .push(format!("{}:live_http_connect_error", selected.provider_id));
                    continue;
                }
            }
        }

        // In-process thermodynamic cascade (`LittleBit-2` / `LangCache` LSN-0 population)
        let cascade = match run_thermodynamic_cascade(
            &rt.ward_rt,
            &selected.model_name,
            &rtk.compressed_text,
            None,
            false,
        ) {
            Ok(c) => c,
            Err((_, _, msg)) => {
                return (
                    StatusCode::BAD_GATEWAY,
                    Json(serde_json::json!({ "ok": false, "error": msg })),
                )
                    .into_response();
            }
        };

        let est_cost =
            (est_tokens.saturating_mul(selected.cost_micro_usd_per_1k_tokens)).div_ceil(1_000);
        let energy_uj = cascade
            .trace
            .aggregate_rails
            .total_uj_ceil()
            .max(est_tokens.saturating_mul(selected.estimated_uj_per_token));

        return (
            StatusCode::OK,
            Json(serde_json::json!({
                "id": format!("chatcmpl-jroute-{}", now_ms()),
                "object": "chat.completion",
                "model": selected.model_name,
                "choices": [{
                    "index": 0,
                    "message": { "role": "assistant", "content": cascade.reply_text },
                    "finish_reason": "stop"
                }],
                "joule_route": {
                    "selected_tier": selected.tier,
                    "selected_provider": selected.provider_id,
                    "vendor": selected.vendor,
                    "dispatch_mode": "inproc_cascade",
                    "rtk_tokens_saved": rtk.tokens_saved,
                    "rtk_reduction_pct": rtk.reduction_pct,
                    "cost_micro_usd": est_cost,
                    "energy_uj": energy_uj,
                    "failover_skipped": accumulated_skips,
                    "cascade_kernel": cascade.kernel,
                }
            })),
        )
            .into_response();
    }

    (
        StatusCode::TOO_MANY_REQUESTS,
        Json(serde_json::json!({
            "ok": false,
            "error": "exceeded maximum waterfall failover attempts",
            "sqlstate": "53200",
            "failover_skipped": accumulated_skips,
        })),
    )
        .into_response()
}

// ============================================================================
// Voyage AI (`POST /v1/rerank`) & Cohere Rerank v2 Wire-Compatible Reranker
// ============================================================================

/// Accepts either plain string documents (`["doc1", "doc2"]`) or Cohere/Voyage object documents
/// (`[{"text": "doc1"}]`).
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum RerankDocInput {
    Plain(String),
    Object {
        text: String,
        #[serde(flatten)]
        extra: BTreeMap<String, serde_json::Value>,
    },
}

impl RerankDocInput {
    pub fn as_text(&self) -> &str {
        match self {
            Self::Plain(s) => s.as_str(),
            Self::Object { text, .. } => text.as_str(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct RerankHttpReq {
    #[serde(default = "default_rerank_model")]
    pub model: String,
    pub query: String,
    pub documents: Vec<RerankDocInput>,
    #[serde(default)]
    pub top_n: Option<usize>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default = "default_true")]
    pub return_documents: bool,
}

fn default_rerank_model() -> String {
    "jouledb-rerank-v1".to_string()
}

fn embed_rerank_feature_vec(text: &str) -> [f64; 64] {
    let mut vec = [0.0f64; 64];
    let lower = text.to_ascii_lowercase();
    for tok in lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 2)
    {
        let h = blake3::hash(tok.as_bytes());
        let b = h.as_bytes();
        let idx = (b[0] as usize) & 63;
        let sign = if (b[1] & 1) == 0 { 1.0 } else { -1.0 };
        vec[idx] += sign * 1.25;
        let bytes = tok.as_bytes();
        if bytes.len() >= 3 {
            for win in bytes.windows(3) {
                let gh = blake3::hash(win);
                let gb = gh.as_bytes();
                let gidx = (gb[0] as usize) & 63;
                let gsign = if (gb[1] & 1) == 0 { 0.35 } else { -0.35 };
                vec[gidx] += gsign;
            }
        }
    }
    vec
}

/// Hybrid deterministic + semantic cross-scoring reranker (Voyage AI `rerank-2.5` + Cohere v2 wire schema).
/// Combines dense 64-dim signed word/3-gram feature cosine similarity, IDF-weighted lexical token coverage,
/// and exact phrase overlap in sub-millisecond Rust.
pub fn rerank_documents(
    model: &str,
    query: &str,
    documents: &[RerankDocInput],
    top_limit: Option<usize>,
    return_documents: bool,
) -> serde_json::Value {
    let t0 = std::time::Instant::now();
    let q_lower = query.to_ascii_lowercase();
    let q_tokens: Vec<&str> = q_lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| t.len() >= 2)
        .collect();
    let q_emb = embed_rerank_feature_vec(query);

    // Document frequency across candidate batch for BM25-style IDF weighting
    let n_docs = documents.len().max(1) as f64;
    let mut doc_freqs = vec![0usize; q_tokens.len()];
    let lowered_docs: Vec<String> = documents
        .iter()
        .map(|d| d.as_text().to_ascii_lowercase())
        .collect();
    for ld in &lowered_docs {
        for (ti, qt) in q_tokens.iter().enumerate() {
            if ld.contains(qt) {
                doc_freqs[ti] += 1;
            }
        }
    }
    let total_idf: f64 = doc_freqs
        .iter()
        .map(|&df| ((n_docs + 1.0) / (df as f64 + 1.0)).ln() + 1.0)
        .sum::<f64>()
        .max(1e-9);

    let mut scored: Vec<(usize, f64)> = documents
        .iter()
        .enumerate()
        .map(|(idx, doc)| {
            let raw_txt = doc.as_text();
            let d_emb = embed_rerank_feature_vec(raw_txt);
            let mut dot = 0.0f64;
            let mut qn = 0.0f64;
            let mut dn = 0.0f64;
            for (a, b) in q_emb.iter().zip(d_emb.iter()) {
                dot += a * b;
                qn += a * a;
                dn += b * b;
            }
            let cos = if qn > 0.0 && dn > 0.0 {
                ((dot / (qn.sqrt() * dn.sqrt())) + 1.0) * 0.5
            } else {
                0.0
            };

            let ld = &lowered_docs[idx];
            let d_words: std::collections::HashSet<&str> = ld
                .split(|c: char| !c.is_ascii_alphanumeric())
                .filter(|t| !t.is_empty())
                .collect();
            let mut idf_hit = 0.0f64;
            for (ti, qt) in q_tokens.iter().enumerate() {
                let w = ((n_docs + 1.0) / (doc_freqs[ti] as f64 + 1.0)).ln() + 1.0;
                if d_words.contains(qt) {
                    idf_hit += w;
                } else if ld.contains(qt) {
                    idf_hit += 0.55 * w;
                }
            }
            let lex_cov = if q_tokens.is_empty() {
                0.0
            } else {
                (idf_hit / total_idf).clamp(0.0, 1.0)
            };
            let phrase_bonus = if !q_lower.is_empty() && ld.contains(&q_lower) {
                0.15
            } else {
                0.0
            };

            let score = (0.55 * lex_cov + 0.35 * cos + phrase_bonus).clamp(0.0, 0.9999);
            (idx, (score * 100_000.0).round() / 100_000.0)
        })
        .collect();

    scored.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });

    let take_n = top_limit.unwrap_or(scored.len()).min(scored.len());
    scored.truncate(take_n);

    let mut results = Vec::with_capacity(scored.len());
    let mut voyage_data = Vec::with_capacity(scored.len());
    let mut total_chars = query.len();
    for d in documents {
        total_chars += d.as_text().len();
    }

    for (orig_idx, rel_score) in scored {
        let txt = documents[orig_idx].as_text();
        if return_documents {
            results.push(serde_json::json!({
                "index": orig_idx,
                "relevance_score": rel_score,
                "document": { "text": txt },
            }));
            voyage_data.push(serde_json::json!({
                "object": "rerank",
                "index": orig_idx,
                "relevance_score": rel_score,
                "document": txt,
            }));
        } else {
            results.push(serde_json::json!({
                "index": orig_idx,
                "relevance_score": rel_score,
            }));
            voyage_data.push(serde_json::json!({
                "object": "rerank",
                "index": orig_idx,
                "relevance_score": rel_score,
            }));
        }
    }

    let elapsed_us = t0.elapsed().as_micros() as u64;
    let total_tokens = total_chars.div_ceil(4);
    let energy_uj = ((documents.len() as u64).div_ceil(4)).max(1);

    serde_json::json!({
        "id": format!("rerank-{}", now_ms()),
        "object": "list",
        "model": model,
        "results": results,
        "data": voyage_data,
        "usage": {
            "total_tokens": total_tokens,
        },
        "joule_receipt": {
            "elapsed_us": elapsed_us,
            "energy_uj": energy_uj,
            "measurement_class": "analytical_est",
        }
    })
}

async fn handle_rerank(Json(req): Json<RerankHttpReq>) -> impl IntoResponse {
    let limit = req.top_n.or(req.top_k);
    let out = rerank_documents(
        &req.model,
        &req.query,
        &req.documents,
        limit,
        req.return_documents,
    );
    (StatusCode::OK, Json(out))
}

async fn handle_register_super_adapter(
    State(rt): State<Arc<JouleRouteRuntime>>,
    Json(req): Json<RegisterSuperAdapterReq>,
) -> impl IntoResponse {
    let adapter_res = match (req.lora_a, req.lora_b) {
        (Some(a), Some(b)) => HarnessLoRAAdapter::new(
            req.adapter_id.clone(),
            req.harness_name,
            req.target_tensor,
            req.rows,
            req.cols,
            req.rank,
            req.lora_alpha,
            a,
            b,
        ),
        _ => HarnessLoRAAdapter::initialized_for_harness(
            req.adapter_id.clone(),
            req.harness_name,
            req.target_tensor,
            req.rows,
            req.cols,
            req.rank,
            req.lora_alpha,
            req.seed,
        ),
    };

    match adapter_res {
        Ok(adapter) => {
            let summary = serde_json::json!({
                "ok": true,
                "adapter_id": adapter.adapter_id,
                "harness_name": adapter.harness_name,
                "target_tensor": adapter.target_tensor,
                "rows": adapter.rows,
                "cols": adapter.cols,
                "rank": adapter.rank,
                "lora_alpha": adapter.lora_alpha,
            });
            let mut map = match rt.adapters.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            map.insert(req.adapter_id, adapter);
            (StatusCode::OK, Json(summary)).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e })),
        )
            .into_response(),
    }
}

async fn handle_list_super_adapters(State(rt): State<Arc<JouleRouteRuntime>>) -> impl IntoResponse {
    let map = match rt.adapters.read() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    let items: Vec<serde_json::Value> = map
        .values()
        .map(|a| {
            serde_json::json!({
                "adapter_id": a.adapter_id,
                "harness_name": a.harness_name,
                "target_tensor": a.target_tensor,
                "rows": a.rows,
                "cols": a.cols,
                "rank": a.rank,
                "lora_alpha": a.lora_alpha,
            })
        })
        .collect();
    (
        StatusCode::OK,
        Json(serde_json::json!({ "ok": true, "adapters": items })),
    )
}

async fn handle_audit_super_adapters(
    State(rt): State<Arc<JouleRouteRuntime>>,
    Json(req): Json<AuditSuperAdaptersReq>,
) -> impl IntoResponse {
    let collected: Result<Vec<HarnessLoRAAdapter>, String> = {
        let map = match rt.adapters.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        req.adapter_ids
            .iter()
            .map(|id| {
                map.get(id)
                    .cloned()
                    .ok_or_else(|| format!("adapter '{id}' not found"))
            })
            .collect()
    };

    match collected.and_then(|list| audit_adapter_orthogonality(&list, req.cosine_threshold)) {
        Ok(audit) => (
            StatusCode::OK,
            Json(serde_json::json!({ "ok": true, "audit": audit })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e })),
        )
            .into_response(),
    }
}

async fn handle_compose_super_adapter(
    State(rt): State<Arc<JouleRouteRuntime>>,
    Json(req): Json<ComposeSuperAdapterReq>,
) -> impl IntoResponse {
    let collected: Result<Vec<HarnessLoRAAdapter>, String> = {
        let map = match rt.adapters.read() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        req.adapter_ids
            .iter()
            .map(|id| {
                map.get(id)
                    .cloned()
                    .ok_or_else(|| format!("adapter '{id}' not found"))
            })
            .collect()
    };

    let list = match collected {
        Ok(l) => l,
        Err(e) => {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": e })),
            )
                .into_response();
        }
    };

    match compose_super_adapter(
        req.super_adapter_id.clone(),
        &list,
        req.weights.as_deref(),
        req.mode,
        req.cosine_threshold,
    ) {
        Ok((composed, receipt)) => {
            let mut fused_tensor_name: Option<String> = None;
            let mut fused_rank: Option<usize> = None;
            if let Some(ref lb_name) = req.fuse_into_littlebit_tensor {
                let Some(base_lb) = rt.synapse_rt.get_littlebit_tensor(lb_name) else {
                    return (
                        StatusCode::NOT_FOUND,
                        Json(serde_json::json!({
                            "ok": false,
                            "error": format!("LittleBit tensor '{lb_name}' not found in SynapseServerRuntime")
                        })),
                    )
                        .into_response();
                };
                match append_super_adapter_to_littlebit(&base_lb, &composed) {
                    Ok(fused) => {
                        fused_rank = Some(fused.rank);
                        let new_name = fused.name.clone();
                        rt.synapse_rt
                            .upsert_littlebit_tensor(new_name.clone(), fused);
                        fused_tensor_name = Some(new_name);
                    }
                    Err(e) => {
                        return (
                            StatusCode::BAD_REQUEST,
                            Json(serde_json::json!({ "ok": false, "error": e })),
                        )
                            .into_response();
                    }
                }
            }

            let mut map = match rt.adapters.write() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            map.insert(req.super_adapter_id, composed);

            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "ok": true,
                    "receipt": receipt,
                    "fused_littlebit_tensor": fused_tensor_name,
                    "fused_littlebit_rank": fused_rank,
                })),
            )
                .into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "ok": false, "error": e })),
        )
            .into_response(),
    }
}

async fn handle_grpo_step(
    State(rt): State<Arc<JouleRouteRuntime>>,
    Json(mut req): Json<GrpoStepReq>,
) -> impl IntoResponse {
    compute_grpo_group_advantages(&mut req.episodes);
    let mut updated_mse: Option<f64> = None;

    if let (Some(adapter_id), Some(x), Some(target)) = (
        req.update_adapter_id.as_deref(),
        req.sample_input.as_deref(),
        req.sample_target_delta.as_deref(),
    ) {
        let adv = req
            .episodes
            .first()
            .map(|e| e.grpo_advantage as f32)
            .unwrap_or(1.0);
        let mut map = match rt.adapters.write() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let Some(adapter) = map.get_mut(adapter_id) else {
            return (
                StatusCode::NOT_FOUND,
                Json(serde_json::json!({ "ok": false, "error": format!("adapter '{adapter_id}' not found") })),
            )
                .into_response();
        };
        match adapter.grpo_advantage_step(x, target, adv, req.learning_rate) {
            Ok(mse) => updated_mse = Some(mse),
            Err(e) => {
                return (
                    StatusCode::BAD_REQUEST,
                    Json(serde_json::json!({ "ok": false, "error": e })),
                )
                    .into_response();
            }
        }
    }

    match export_jsonl_for_trl(&req.episodes) {
        Ok(jsonl) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "ok": true,
                "episodes": req.episodes,
                "updated_adapter_mse": updated_mse,
                "trl_jsonl": jsonl,
            })),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "ok": false, "error": e })),
        )
            .into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sealed_api_key_encrypt_then_mac_and_tamper_reject() {
        let key = blake3::derive_key("test-vault", b"unit-seed");
        let sealed = SealedApiKey::seal(&key, 1, "sk-live-super-secret-1234").expect("seal");
        assert_eq!(sealed.masked_preview, "sk-l...1234");
        assert!(!sealed.ciphertext_hex.contains("super-secret"));
        let opened = sealed.unseal(&key).expect("unseal");
        assert_eq!(opened, "sk-live-super-secret-1234");

        let mut tampered = sealed.clone();
        tampered.ciphertext_hex.replace_range(0..2, "ff");
        assert!(tampered.unseal(&key).is_err());
    }

    #[test]
    fn test_rtk_context_compressor_preserves_code_and_strips_noise() {
        let noisy = "\x1b[32mCompiling foo v0.1.0\x1b[0m\n\
Downloading bar v1.2.3\n\
========================================\n\
========================================\n\
retryattempt...\n\
retryattempt...\n\
retryattempt...\n\
```sql\n\
SELECT account_id, balance FROM accounts WHERE account_id = 42;\n\
```\n\
error[E0425]: cannot find value `x` in this scope";
        let rec = compress_context_rtk(noisy);
        assert!(rec.compressed_chars < rec.raw_chars);
        assert!(rec.compressed_text.contains("SELECT account_id, balance"));
        assert!(rec.compressed_text.contains("error[E0425]"));
        assert!(!rec.compressed_text.contains("\x1b["));
    }
}
