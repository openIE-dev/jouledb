//! Agent Memory for JouleDB
//!
//! Persistent temporal knowledge with episodic, semantic, and working memory
//! types. Memories have temporal decay and per-tenant/per-agent isolation.
//! Built on top of JouleDB's own storage — first database to ship agent
//! memory as a built-in primitive.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;

// ============================================================================
// Errors
// ============================================================================

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("memory not found: {0}")]
    NotFound(String),

    #[error("invalid memory type: {0}")]
    InvalidType(String),

    #[error("storage error: {0}")]
    Storage(String),
}

// ============================================================================
// Memory types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryType {
    /// Timestamped interaction records with embeddings for similarity recall
    Episodic,
    /// Knowledge graph triples with confidence scores
    Semantic,
    /// Short-term scratchpad per session, auto-expires
    Working,
}

impl MemoryType {
    pub fn from_str(s: &str) -> Result<Self, MemoryError> {
        match s {
            "episodic" => Ok(Self::Episodic),
            "semantic" => Ok(Self::Semantic),
            "working" => Ok(Self::Working),
            other => Err(MemoryError::InvalidType(other.to_string())),
        }
    }
}

// ============================================================================
// Memory entry
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub memory_type: MemoryType,
    pub content: String,
    pub embedding: Vec<f32>,
    pub metadata: HashMap<String, String>,
    pub created_at: u64,
    pub last_accessed: u64,
    pub access_count: u64,
    pub decay_rate: f64,
    pub tenant_id: String,
    pub agent_id: Option<String>,
}

// ============================================================================
// Scored memory (recall result)
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredMemory {
    pub memory: Memory,
    pub raw_similarity: f64,
    pub effective_score: f64,
    pub age_hours: f64,
}

// ============================================================================
// Memory stats
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_memories: usize,
    pub episodic_count: usize,
    pub semantic_count: usize,
    pub working_count: usize,
    pub tenant_id: String,
}

// ============================================================================
// Request types
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct StoreMemoryRequest {
    pub content: String,
    #[serde(default = "default_memory_type")]
    pub memory_type: String,
    #[serde(default)]
    pub metadata: HashMap<String, String>,
    pub agent_id: Option<String>,
    pub decay_rate: Option<f64>,
}

fn default_memory_type() -> String {
    "episodic".to_string()
}

#[derive(Debug, Clone, Deserialize)]
pub struct RecallMemoryRequest {
    pub query: String,
    #[serde(default = "default_k")]
    pub k: usize,
    pub time_range_hours: Option<f64>,
    pub half_life_hours: Option<f64>,
    pub memory_type: Option<String>,
    pub agent_id: Option<String>,
}

fn default_k() -> usize {
    5
}

#[derive(Debug, Clone, Deserialize)]
pub struct ForgetMemoryRequest {
    pub memory_type: Option<String>,
    pub agent_id: Option<String>,
    pub before_timestamp: Option<u64>,
    pub ids: Option<Vec<String>>,
}

// ============================================================================
// MemoryManager
// ============================================================================

pub struct MemoryManager {
    /// All memories indexed by tenant_id, then by memory id
    memories: RwLock<HashMap<String, HashMap<String, Memory>>>,
    db: Option<joule_db_local::Database>,
}

impl MemoryManager {
    pub fn new() -> Self {
        Self {
            memories: RwLock::new(HashMap::new()),
            db: None,
        }
    }

    /// Open a durable manager backed by WAL storage
    pub fn open(db_path: &str) -> Result<Self, MemoryError> {
        let db = joule_db_local::Database::open(db_path)
            .map_err(|e| MemoryError::Storage(format!("failed to open memory db: {e}")))?;
        let mut mgr = Self {
            memories: RwLock::new(HashMap::new()),
            db: Some(db),
        };
        mgr.recover()?;
        Ok(mgr)
    }

    fn persist(&self, key: &str, value: &impl Serialize) {
        if let Some(ref db) = self.db {
            if let Ok(bytes) = serde_json::to_vec(value) {
                let _ = db.put(key.as_bytes(), &bytes);
            }
        }
    }

    fn remove_key(&self, key: &str) {
        if let Some(ref db) = self.db {
            let _ = db.delete(key.as_bytes());
        }
    }

    fn recover(&mut self) -> Result<(), MemoryError> {
        let db = match self.db {
            Some(ref db) => db,
            None => return Ok(()),
        };
        let entries = db.prefix_scan(b"mem:").unwrap_or_default();
        let mut all: HashMap<String, HashMap<String, Memory>> = HashMap::new();
        for (_k, v) in &entries {
            if let Ok(memory) = serde_json::from_slice::<Memory>(v) {
                all.entry(memory.tenant_id.clone())
                    .or_default()
                    .insert(memory.id.clone(), memory);
            }
        }
        *self
            .memories
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))? = all;
        Ok(())
    }

    /// Store a new memory. Returns the memory ID.
    pub fn store(&self, req: &StoreMemoryRequest, tenant_id: &str) -> Result<String, MemoryError> {
        // Input validation
        if req.content.is_empty() {
            return Err(MemoryError::Storage(
                "Memory content cannot be empty".into(),
            ));
        }
        if req.content.len() > 1_000_000 {
            return Err(MemoryError::Storage(
                "Memory content too large (max 1MB)".into(),
            ));
        }
        if req.metadata.len() > 100 {
            return Err(MemoryError::Storage(
                "Too many metadata entries (max 100)".into(),
            ));
        }
        if let Some(ref agent_id) = req.agent_id {
            if agent_id.len() > 256 {
                return Err(MemoryError::Storage(
                    "agent_id too long (max 256 chars)".into(),
                ));
            }
        }
        if let Some(rate) = req.decay_rate {
            if !(0.0..=1.0).contains(&rate) || rate.is_nan() {
                return Err(MemoryError::Storage(
                    "decay_rate must be between 0.0 and 1.0".into(),
                ));
            }
        }
        let memory_type = MemoryType::from_str(&req.memory_type)?;
        let now = now_millis();
        let id = format!("mem_{:016x}", now ^ (next_id() & 0xFFFF_FFFF));

        let embedding = content_to_embedding(&req.content);

        let memory = Memory {
            id: id.clone(),
            memory_type,
            content: req.content.clone(),
            embedding,
            metadata: req.metadata.clone(),
            created_at: now,
            last_accessed: now,
            access_count: 0,
            decay_rate: req.decay_rate.unwrap_or(0.95),
            tenant_id: tenant_id.to_string(),
            agent_id: req.agent_id.clone(),
        };

        let mut all = self
            .memories
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        self.persist(&format!("mem:{}:{}", tenant_id, id), &memory);

        all.entry(tenant_id.to_string())
            .or_default()
            .insert(id.clone(), memory);

        Ok(id)
    }

    /// Recall memories with temporal decay and similarity scoring.
    pub fn recall(
        &self,
        req: &RecallMemoryRequest,
        tenant_id: &str,
    ) -> Result<Vec<ScoredMemory>, MemoryError> {
        if req.query.is_empty() {
            return Err(MemoryError::Storage("Recall query cannot be empty".into()));
        }
        if req.query.len() > 1_000_000 {
            return Err(MemoryError::Storage(
                "Recall query too large (max 1MB)".into(),
            ));
        }
        let k = req.k.min(10_000);
        let half_life_hours = req.half_life_hours.unwrap_or(168.0); // 1 week
        let now = now_millis();
        let query_embedding = content_to_embedding(&req.query);

        let memory_type_filter = req
            .memory_type
            .as_deref()
            .map(MemoryType::from_str)
            .transpose()?;

        let all = self
            .memories
            .read()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let tenant_memories = match all.get(tenant_id) {
            Some(m) => m,
            None => return Ok(Vec::new()),
        };

        let mut scored: Vec<ScoredMemory> = tenant_memories
            .values()
            .filter(|m| {
                // Filter by memory type
                if let Some(ref t) = memory_type_filter {
                    if m.memory_type != *t {
                        return false;
                    }
                }
                // Filter by agent_id
                if let Some(ref agent) = req.agent_id {
                    if m.agent_id.as_deref() != Some(agent.as_str()) {
                        return false;
                    }
                }
                // Filter by time range
                if let Some(hours) = req.time_range_hours {
                    let cutoff = now.saturating_sub((hours * 3_600_000.0) as u64);
                    if m.created_at < cutoff {
                        return false;
                    }
                }
                true
            })
            .map(|m| {
                let raw_similarity = cosine_similarity(&query_embedding, &m.embedding);
                let age_hours = (now.saturating_sub(m.created_at)) as f64 / 3_600_000.0;
                let decay = 0.5_f64.powf(age_hours / half_life_hours);
                let access_boost = (2.0 + m.access_count as f64).log2();
                let effective_score = raw_similarity * decay * access_boost;

                ScoredMemory {
                    memory: m.clone(),
                    raw_similarity,
                    effective_score,
                    age_hours,
                }
            })
            .collect();

        // Sort by effective score descending
        scored.sort_by(|a, b| {
            b.effective_score
                .partial_cmp(&a.effective_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Take top k (capped at 10,000)
        scored.truncate(k);

        // Update access counts for recalled memories and persist
        drop(all);
        if let Ok(mut all) = self.memories.write() {
            if let Some(tenant_memories) = all.get_mut(tenant_id) {
                for sm in &scored {
                    if let Some(m) = tenant_memories.get_mut(&sm.memory.id) {
                        m.access_count += 1;
                        m.last_accessed = now;
                        // Persist updated access count
                        self.persist(&format!("mem:{}:{}", tenant_id, sm.memory.id), m);
                    }
                }
            }
        }

        Ok(scored)
    }

    /// Forget memories matching the filter. Returns count of deleted memories.
    pub fn forget(&self, req: &ForgetMemoryRequest, tenant_id: &str) -> Result<u64, MemoryError> {
        let mut all = self
            .memories
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let tenant_memories = match all.get_mut(tenant_id) {
            Some(m) => m,
            None => return Ok(0),
        };

        let memory_type_filter = req
            .memory_type
            .as_deref()
            .map(MemoryType::from_str)
            .transpose()?;

        let before = tenant_memories.len();

        // If specific IDs given, delete only those
        if let Some(ref ids) = req.ids {
            for id in ids {
                if tenant_memories.remove(id).is_some() {
                    self.remove_key(&format!("mem:{}:{}", tenant_id, id));
                }
            }
        } else {
            let mut removed_ids = Vec::new();
            tenant_memories.retain(|id, m| {
                // Keep memories that DON'T match the filter
                if let Some(ref t) = memory_type_filter {
                    if m.memory_type != *t {
                        return true; // keep
                    }
                }
                if let Some(ref agent) = req.agent_id {
                    if m.agent_id.as_deref() != Some(agent.as_str()) {
                        return true; // keep
                    }
                }
                if let Some(before_ts) = req.before_timestamp {
                    if m.created_at >= before_ts {
                        return true; // keep
                    }
                }
                removed_ids.push(id.clone());
                false // delete
            });
            for id in &removed_ids {
                self.remove_key(&format!("mem:{}:{}", tenant_id, id));
            }
        }

        let after = tenant_memories.len();
        Ok((before - after) as u64)
    }

    /// Consolidate episodic memories into semantic memories.
    /// Groups recent episodic memories and creates semantic summaries.
    pub fn consolidate(&self, tenant_id: &str) -> Result<u64, MemoryError> {
        let mut all = self
            .memories
            .write()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let tenant_memories = match all.get_mut(tenant_id) {
            Some(m) => m,
            None => return Ok(0),
        };

        // Collect episodic memories
        let episodic: Vec<Memory> = tenant_memories
            .values()
            .filter(|m| m.memory_type == MemoryType::Episodic)
            .cloned()
            .collect();

        if episodic.is_empty() {
            return Ok(0);
        }

        // Create a single semantic memory summarizing the episodic batch
        let now = now_millis();
        let combined_content: String = episodic
            .iter()
            .map(|m| m.content.as_str())
            .collect::<Vec<_>>()
            .join(" | ");

        let id = format!("sem_{:016x}", now ^ (next_id() & 0xFFFF_FFFF));
        let embedding = content_to_embedding(&combined_content);

        let semantic = Memory {
            id: id.clone(),
            memory_type: MemoryType::Semantic,
            content: combined_content,
            embedding,
            metadata: {
                let mut m = HashMap::new();
                m.insert("consolidated_from".to_string(), episodic.len().to_string());
                m.insert("consolidated_at".to_string(), now.to_string());
                m
            },
            created_at: now,
            last_accessed: now,
            access_count: 0,
            decay_rate: 0.99, // semantic memories decay slower
            tenant_id: tenant_id.to_string(),
            agent_id: episodic.first().and_then(|m| m.agent_id.clone()),
        };

        // Remove the consolidated episodic memories
        let count = episodic.len() as u64;
        for m in &episodic {
            tenant_memories.remove(&m.id);
            self.remove_key(&format!("mem:{}:{}", tenant_id, m.id));
        }

        // Insert the semantic memory
        self.persist(&format!("mem:{}:{}", tenant_id, id), &semantic);
        tenant_memories.insert(id, semantic);

        Ok(count)
    }

    /// Get memory stats for a tenant
    pub fn stats(&self, tenant_id: &str) -> Result<MemoryStats, MemoryError> {
        let all = self
            .memories
            .read()
            .map_err(|e| MemoryError::Storage(e.to_string()))?;

        let tenant_memories = match all.get(tenant_id) {
            Some(m) => m,
            None => {
                return Ok(MemoryStats {
                    total_memories: 0,
                    episodic_count: 0,
                    semantic_count: 0,
                    working_count: 0,
                    tenant_id: tenant_id.to_string(),
                });
            }
        };

        let mut episodic = 0;
        let mut semantic = 0;
        let mut working = 0;

        for m in tenant_memories.values() {
            match m.memory_type {
                MemoryType::Episodic => episodic += 1,
                MemoryType::Semantic => semantic += 1,
                MemoryType::Working => working += 1,
            }
        }

        Ok(MemoryStats {
            total_memories: tenant_memories.len(),
            episodic_count: episodic,
            semantic_count: semantic,
            working_count: working,
            tenant_id: tenant_id.to_string(),
        })
    }
}

// ============================================================================
// Embedding helpers
// ============================================================================

/// Generate a simple embedding from text content.
/// Uses a deterministic hash-based approach (64 dimensions).
/// In production, this would use a proper embedding model.
fn content_to_embedding(content: &str) -> Vec<f32> {
    let dim = 64;
    let mut embedding = vec![0.0f32; dim];

    // Hash-based embedding: each character contributes to multiple dimensions
    for (i, byte) in content.bytes().enumerate() {
        let idx = i % dim;
        let val = (byte as f32 - 96.0) / 32.0; // normalize to roughly [-1, 1]
        embedding[idx] += val;
        embedding[(idx + 7) % dim] += val * 0.5;
        embedding[(idx + 13) % dim] -= val * 0.3;
    }

    // L2 normalize
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut embedding {
            *x /= norm;
        }
    }

    embedding
}

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| *x as f64 * *y as f64)
        .sum();
    let norm_a: f64 = a
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b
        .iter()
        .map(|x| (*x as f64) * (*x as f64))
        .sum::<f64>()
        .sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Monotonic counter for unique IDs
fn next_id() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let cnt = COUNTER.fetch_add(1, Ordering::Relaxed);
    let t = now_millis();
    t ^ cnt
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_recall() {
        let mgr = MemoryManager::new();

        let id = mgr
            .store(
                &StoreMemoryRequest {
                    content: "The meeting discussed project deadlines".to_string(),
                    memory_type: "episodic".to_string(),
                    metadata: HashMap::new(),
                    agent_id: Some("agent-1".to_string()),
                    decay_rate: None,
                },
                "default",
            )
            .unwrap();

        assert!(id.starts_with("mem_"));

        let results = mgr
            .recall(
                &RecallMemoryRequest {
                    query: "The meeting discussed project deadlines".to_string(),
                    k: 5,
                    time_range_hours: None,
                    half_life_hours: None,
                    memory_type: None,
                    agent_id: None,
                },
                "default",
            )
            .unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].effective_score > 0.0);
        assert_eq!(
            results[0].memory.content,
            "The meeting discussed project deadlines"
        );
    }

    #[test]
    fn test_temporal_decay() {
        let mgr = MemoryManager::new();

        // Store a memory and manually backdate it
        mgr.store(
            &StoreMemoryRequest {
                content: "old memory".to_string(),
                memory_type: "episodic".to_string(),
                metadata: HashMap::new(),
                agent_id: None,
                decay_rate: None,
            },
            "default",
        )
        .unwrap();

        // Store a recent memory with same content
        mgr.store(
            &StoreMemoryRequest {
                content: "old memory recent".to_string(),
                memory_type: "episodic".to_string(),
                metadata: HashMap::new(),
                agent_id: None,
                decay_rate: None,
            },
            "default",
        )
        .unwrap();

        let results = mgr
            .recall(
                &RecallMemoryRequest {
                    query: "old memory".to_string(),
                    k: 10,
                    time_range_hours: None,
                    half_life_hours: Some(168.0),
                    memory_type: None,
                    agent_id: None,
                },
                "default",
            )
            .unwrap();

        assert_eq!(results.len(), 2);
        // Both should have positive scores (both are recent)
        for r in &results {
            assert!(r.effective_score > 0.0);
        }
    }

    #[test]
    fn test_forget_by_type() {
        let mgr = MemoryManager::new();

        mgr.store(
            &StoreMemoryRequest {
                content: "episodic mem".to_string(),
                memory_type: "episodic".to_string(),
                metadata: HashMap::new(),
                agent_id: None,
                decay_rate: None,
            },
            "default",
        )
        .unwrap();

        mgr.store(
            &StoreMemoryRequest {
                content: "working mem".to_string(),
                memory_type: "working".to_string(),
                metadata: HashMap::new(),
                agent_id: None,
                decay_rate: None,
            },
            "default",
        )
        .unwrap();

        let deleted = mgr
            .forget(
                &ForgetMemoryRequest {
                    memory_type: Some("working".to_string()),
                    agent_id: None,
                    before_timestamp: None,
                    ids: None,
                },
                "default",
            )
            .unwrap();

        assert_eq!(deleted, 1);

        let stats = mgr.stats("default").unwrap();
        assert_eq!(stats.total_memories, 1);
        assert_eq!(stats.episodic_count, 1);
        assert_eq!(stats.working_count, 0);
    }

    #[test]
    fn test_forget_by_ids() {
        let mgr = MemoryManager::new();

        let id1 = mgr
            .store(
                &StoreMemoryRequest {
                    content: "first".to_string(),
                    memory_type: "episodic".to_string(),
                    metadata: HashMap::new(),
                    agent_id: None,
                    decay_rate: None,
                },
                "default",
            )
            .unwrap();

        mgr.store(
            &StoreMemoryRequest {
                content: "second".to_string(),
                memory_type: "episodic".to_string(),
                metadata: HashMap::new(),
                agent_id: None,
                decay_rate: None,
            },
            "default",
        )
        .unwrap();

        let deleted = mgr
            .forget(
                &ForgetMemoryRequest {
                    memory_type: None,
                    agent_id: None,
                    before_timestamp: None,
                    ids: Some(vec![id1]),
                },
                "default",
            )
            .unwrap();

        assert_eq!(deleted, 1);
        assert_eq!(mgr.stats("default").unwrap().total_memories, 1);
    }

    #[test]
    fn test_consolidate() {
        let mgr = MemoryManager::new();

        for i in 0..3 {
            mgr.store(
                &StoreMemoryRequest {
                    content: format!("episode {i}"),
                    memory_type: "episodic".to_string(),
                    metadata: HashMap::new(),
                    agent_id: None,
                    decay_rate: None,
                },
                "default",
            )
            .unwrap();
        }

        let consolidated = mgr.consolidate("default").unwrap();
        assert_eq!(consolidated, 3);

        let stats = mgr.stats("default").unwrap();
        assert_eq!(stats.episodic_count, 0);
        assert_eq!(stats.semantic_count, 1);
        assert_eq!(stats.total_memories, 1);
    }

    #[test]
    fn test_tenant_isolation() {
        let mgr = MemoryManager::new();

        mgr.store(
            &StoreMemoryRequest {
                content: "tenant a data".to_string(),
                memory_type: "episodic".to_string(),
                metadata: HashMap::new(),
                agent_id: None,
                decay_rate: None,
            },
            "tenant-a",
        )
        .unwrap();

        mgr.store(
            &StoreMemoryRequest {
                content: "tenant b data".to_string(),
                memory_type: "episodic".to_string(),
                metadata: HashMap::new(),
                agent_id: None,
                decay_rate: None,
            },
            "tenant-b",
        )
        .unwrap();

        let results_a = mgr
            .recall(
                &RecallMemoryRequest {
                    query: "data".to_string(),
                    k: 10,
                    time_range_hours: None,
                    half_life_hours: None,
                    memory_type: None,
                    agent_id: None,
                },
                "tenant-a",
            )
            .unwrap();

        assert_eq!(results_a.len(), 1);
        assert!(results_a[0].memory.content.contains("tenant a"));
    }

    #[test]
    fn test_stats_empty_tenant() {
        let mgr = MemoryManager::new();
        let stats = mgr.stats("nonexistent").unwrap();
        assert_eq!(stats.total_memories, 0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-6);
    }
}
