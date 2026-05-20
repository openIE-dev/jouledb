//! Model caching with LRU eviction, warm-up inference, memory budgets,
//! and version management.
//!
//! Manages a pool of loaded inference models to amortise loading cost.
//! Supports memory-budget-aware LRU eviction, warm-up runs to prime
//! JIT caches, and model versioning for safe hot-swaps.

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ── Model Metadata ─────────────────────────────────────────────

/// Metadata describing a cached model.
#[derive(Debug, Clone)]
pub struct ModelMeta {
    pub name: String,
    pub version: u64,
    pub size_bytes: usize,
    pub param_count: usize,
    pub format: ModelFormat,
    pub tags: HashMap<String, String>,
}

/// Supported model serialisation formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFormat {
    Onnx,
    SafeTensors,
    TorchScript,
    Custom,
}

impl fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ModelFormat::Onnx => write!(f, "ONNX"),
            ModelFormat::SafeTensors => write!(f, "SafeTensors"),
            ModelFormat::TorchScript => write!(f, "TorchScript"),
            ModelFormat::Custom => write!(f, "Custom"),
        }
    }
}

impl ModelMeta {
    pub fn new(name: impl Into<String>, version: u64, size_bytes: usize) -> Self {
        Self {
            name: name.into(),
            version,
            size_bytes,
            param_count: 0,
            format: ModelFormat::Custom,
            tags: HashMap::new(),
        }
    }

    pub fn with_param_count(mut self, count: usize) -> Self {
        self.param_count = count;
        self
    }

    pub fn with_format(mut self, fmt: ModelFormat) -> Self {
        self.format = fmt;
        self
    }

    pub fn with_tag(mut self, key: impl Into<String>, val: impl Into<String>) -> Self {
        self.tags.insert(key.into(), val.into());
        self
    }

    /// Unique cache key: name + version.
    pub fn cache_key(&self) -> String {
        format!("{}@v{}", self.name, self.version)
    }
}

impl fmt::Display for ModelMeta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelMeta('{}', v{}, {}B, fmt={})",
            self.name, self.version, self.size_bytes, self.format
        )
    }
}

// ── Cache Entry ────────────────────────────────────────────────

/// An entry in the model cache.
#[derive(Debug)]
struct CacheEntry {
    meta: ModelMeta,
    /// Simulated model payload (weights hash for tracking).
    payload_hash: u64,
    /// When this entry was loaded.
    loaded_at: Instant,
    /// Last time this model was used for inference.
    last_used: Instant,
    /// Number of inference calls since load.
    inference_count: u64,
    /// Whether warm-up has been performed.
    warmed_up: bool,
    /// Average inference latency (microseconds).
    avg_latency_us: f64,
}

impl CacheEntry {
    fn new(meta: ModelMeta, payload_hash: u64) -> Self {
        let now = Instant::now();
        Self {
            meta,
            payload_hash,
            loaded_at: now,
            last_used: now,
            inference_count: 0,
            warmed_up: false,
            avg_latency_us: 0.0,
        }
    }

    fn touch(&mut self) {
        self.last_used = Instant::now();
        self.inference_count += 1;
    }

    fn record_latency(&mut self, latency_us: f64) {
        let n = self.inference_count as f64;
        if n <= 1.0 {
            self.avg_latency_us = latency_us;
        } else {
            // Running average.
            self.avg_latency_us =
                self.avg_latency_us * ((n - 1.0) / n) + latency_us / n;
        }
    }

    fn idle_duration(&self) -> Duration {
        self.last_used.elapsed()
    }

    fn size(&self) -> usize {
        self.meta.size_bytes
    }
}

// ── Eviction Policy ────────────────────────────────────────────

/// Eviction strategy for the model cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Evict least-recently-used first.
    Lru,
    /// Evict largest models first.
    LargestFirst,
    /// Evict least-frequently-used first.
    Lfu,
    /// Evict oldest loaded model first.
    Fifo,
}

impl fmt::Display for EvictionPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EvictionPolicy::Lru => write!(f, "LRU"),
            EvictionPolicy::LargestFirst => write!(f, "largest-first"),
            EvictionPolicy::Lfu => write!(f, "LFU"),
            EvictionPolicy::Fifo => write!(f, "FIFO"),
        }
    }
}

// ── Cache Stats ────────────────────────────────────────────────

/// Aggregate statistics for the model cache.
#[derive(Debug, Clone)]
pub struct CacheStats {
    pub total_models: usize,
    pub total_memory_bytes: usize,
    pub budget_bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            return 0.0;
        }
        self.hits as f64 / total as f64
    }

    pub fn utilisation(&self) -> f64 {
        if self.budget_bytes == 0 {
            return 0.0;
        }
        self.total_memory_bytes as f64 / self.budget_bytes as f64
    }
}

impl fmt::Display for CacheStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CacheStats(models={}, mem={}/{}B, hit_rate={:.2}%, evictions={})",
            self.total_models,
            self.total_memory_bytes,
            self.budget_bytes,
            self.hit_rate() * 100.0,
            self.evictions
        )
    }
}

// ── ModelCache ─────────────────────────────────────────────────

/// Memory-budget-aware model cache with eviction support.
#[derive(Debug)]
pub struct ModelCache {
    entries: HashMap<String, CacheEntry>,
    budget_bytes: usize,
    used_bytes: usize,
    policy: EvictionPolicy,
    hits: u64,
    misses: u64,
    evictions: u64,
    warmup_iterations: u32,
}

impl ModelCache {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            entries: HashMap::new(),
            budget_bytes,
            used_bytes: 0,
            policy: EvictionPolicy::Lru,
            hits: 0,
            misses: 0,
            evictions: 0,
            warmup_iterations: 3,
        }
    }

    pub fn with_policy(mut self, policy: EvictionPolicy) -> Self {
        self.policy = policy;
        self
    }

    pub fn with_warmup_iterations(mut self, n: u32) -> Self {
        self.warmup_iterations = n;
        self
    }

    pub fn with_budget(mut self, bytes: usize) -> Self {
        self.budget_bytes = bytes;
        self
    }

    /// Load a model into the cache, evicting if necessary.
    pub fn load(&mut self, meta: ModelMeta, payload_hash: u64) -> Result<(), String> {
        let key = meta.cache_key();

        if meta.size_bytes > self.budget_bytes {
            return Err(format!(
                "model '{}' ({}B) exceeds total budget ({}B)",
                key, meta.size_bytes, self.budget_bytes
            ));
        }

        // Evict until there is room.
        while self.used_bytes + meta.size_bytes > self.budget_bytes {
            if !self.evict_one() {
                return Err("cannot free enough space".into());
            }
        }

        if let Some(old) = self.entries.get(&key) {
            self.used_bytes -= old.size();
        }

        let size = meta.size_bytes;
        self.entries.insert(key, CacheEntry::new(meta, payload_hash));
        self.used_bytes += size;
        Ok(())
    }

    /// Look up a model; returns `true` if found (cache hit).
    pub fn get(&mut self, name: &str, version: u64) -> bool {
        let key = format!("{name}@v{version}");
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.touch();
            self.hits += 1;
            true
        } else {
            self.misses += 1;
            false
        }
    }

    /// Record an inference latency for a cached model.
    pub fn record_inference(&mut self, name: &str, version: u64, latency_us: f64) {
        let key = format!("{name}@v{version}");
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.touch();
            entry.record_latency(latency_us);
        }
    }

    /// Mark a model as warmed up.
    pub fn mark_warmed_up(&mut self, name: &str, version: u64) {
        let key = format!("{name}@v{version}");
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.warmed_up = true;
        }
    }

    /// Remove a specific model from the cache.
    pub fn unload(&mut self, name: &str, version: u64) -> bool {
        let key = format!("{name}@v{version}");
        if let Some(entry) = self.entries.remove(&key) {
            self.used_bytes -= entry.size();
            true
        } else {
            false
        }
    }

    /// Number of cached models.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current memory usage in bytes.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }

    /// Remaining budget in bytes.
    pub fn available_bytes(&self) -> usize {
        self.budget_bytes.saturating_sub(self.used_bytes)
    }

    /// Aggregate cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            total_models: self.entries.len(),
            total_memory_bytes: self.used_bytes,
            budget_bytes: self.budget_bytes,
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
        }
    }

    /// List all cached model keys.
    pub fn keys(&self) -> Vec<String> {
        self.entries.keys().cloned().collect()
    }

    /// Evict one model according to the configured policy.
    fn evict_one(&mut self) -> bool {
        let victim_key = match self.policy {
            EvictionPolicy::Lru => {
                self.entries
                    .iter()
                    .max_by_key(|(_, e)| e.idle_duration())
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::LargestFirst => {
                self.entries
                    .iter()
                    .max_by_key(|(_, e)| e.size())
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::Lfu => {
                self.entries
                    .iter()
                    .min_by_key(|(_, e)| e.inference_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::Fifo => {
                self.entries
                    .iter()
                    .min_by_key(|(_, e)| e.loaded_at)
                    .map(|(k, _)| k.clone())
            }
        };

        if let Some(key) = victim_key {
            if let Some(entry) = self.entries.remove(&key) {
                self.used_bytes -= entry.size();
                self.evictions += 1;
                return true;
            }
        }
        false
    }
}

impl fmt::Display for ModelCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ModelCache(models={}, used={}/{}B, policy={})",
            self.len(),
            self.used_bytes,
            self.budget_bytes,
            self.policy
        )
    }
}

// ── Version Registry ───────────────────────────────────────────

/// Tracks model versions for safe hot-swap.
#[derive(Debug)]
pub struct VersionRegistry {
    versions: HashMap<String, Vec<VersionEntry>>,
}

#[derive(Debug, Clone)]
struct VersionEntry {
    version: u64,
    payload_hash: u64,
    registered_at: Instant,
    active: bool,
}

impl VersionRegistry {
    pub fn new() -> Self {
        Self { versions: HashMap::new() }
    }

    /// Register a new version for a model.
    pub fn register(&mut self, name: impl Into<String>, version: u64, payload_hash: u64) {
        let entries = self.versions.entry(name.into()).or_default();
        // Deactivate previous versions.
        for e in entries.iter_mut() {
            e.active = false;
        }
        entries.push(VersionEntry {
            version,
            payload_hash,
            registered_at: Instant::now(),
            active: true,
        });
    }

    /// Get the currently active version for a model.
    pub fn active_version(&self, name: &str) -> Option<u64> {
        self.versions
            .get(name)
            .and_then(|entries| entries.iter().find(|e| e.active).map(|e| e.version))
    }

    /// List all versions for a model.
    pub fn all_versions(&self, name: &str) -> Vec<u64> {
        self.versions
            .get(name)
            .map(|entries| entries.iter().map(|e| e.version).collect())
            .unwrap_or_default()
    }

    /// Number of tracked model families.
    pub fn model_count(&self) -> usize {
        self.versions.len()
    }
}

impl Default for VersionRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for VersionRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VersionRegistry(models={})", self.versions.len())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(name: &str, version: u64, size: usize) -> ModelMeta {
        ModelMeta::new(name, version, size)
    }

    #[test]
    fn test_model_meta_cache_key() {
        let m = make_meta("resnet50", 3, 100);
        assert_eq!(m.cache_key(), "resnet50@v3");
    }

    #[test]
    fn test_model_meta_builder() {
        let m = ModelMeta::new("bert", 1, 500)
            .with_param_count(1000)
            .with_format(ModelFormat::Onnx)
            .with_tag("task", "nlp");
        assert_eq!(m.param_count, 1000);
        assert_eq!(m.format, ModelFormat::Onnx);
        assert_eq!(m.tags.get("task").unwrap(), "nlp");
    }

    #[test]
    fn test_cache_load_and_get() {
        let mut cache = ModelCache::new(1000);
        cache.load(make_meta("m1", 1, 200), 111).unwrap();
        assert!(cache.get("m1", 1));
        assert!(!cache.get("m1", 2));
    }

    #[test]
    fn test_cache_hit_miss_counting() {
        let mut cache = ModelCache::new(1000);
        cache.load(make_meta("m1", 1, 100), 1).unwrap();
        cache.get("m1", 1); // hit
        cache.get("m1", 1); // hit
        cache.get("m2", 1); // miss
        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
    }

    #[test]
    fn test_cache_eviction_lru() {
        let mut cache = ModelCache::new(300).with_policy(EvictionPolicy::Lru);
        cache.load(make_meta("a", 1, 150), 1).unwrap();
        cache.load(make_meta("b", 1, 150), 2).unwrap();
        // Touch 'b' so 'a' is LRU.
        cache.get("b", 1);
        // Load 'c' which should evict 'a'.
        cache.load(make_meta("c", 1, 150), 3).unwrap();
        assert!(!cache.get("a", 1));
        assert!(cache.get("b", 1));
    }

    #[test]
    fn test_cache_eviction_largest() {
        let mut cache = ModelCache::new(400).with_policy(EvictionPolicy::LargestFirst);
        cache.load(make_meta("small", 1, 100), 1).unwrap();
        cache.load(make_meta("big", 1, 300), 2).unwrap();
        // Load one more → evict biggest.
        cache.load(make_meta("med", 1, 200), 3).unwrap();
        assert!(cache.get("small", 1));
        assert!(!cache.get("big", 1));
    }

    #[test]
    fn test_cache_budget_exceeded() {
        let mut cache = ModelCache::new(100);
        let result = cache.load(make_meta("huge", 1, 200), 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_unload() {
        let mut cache = ModelCache::new(1000);
        cache.load(make_meta("m", 1, 100), 1).unwrap();
        assert!(cache.unload("m", 1));
        assert!(!cache.get("m", 1));
        assert_eq!(cache.used_bytes(), 0);
    }

    #[test]
    fn test_cache_available_bytes() {
        let mut cache = ModelCache::new(1000);
        assert_eq!(cache.available_bytes(), 1000);
        cache.load(make_meta("m", 1, 300), 1).unwrap();
        assert_eq!(cache.available_bytes(), 700);
    }

    #[test]
    fn test_cache_keys() {
        let mut cache = ModelCache::new(1000);
        cache.load(make_meta("a", 1, 100), 1).unwrap();
        cache.load(make_meta("b", 2, 100), 2).unwrap();
        let keys = cache.keys();
        assert_eq!(keys.len(), 2);
    }

    #[test]
    fn test_cache_record_inference() {
        let mut cache = ModelCache::new(1000);
        cache.load(make_meta("m", 1, 100), 1).unwrap();
        cache.get("m", 1);
        cache.record_inference("m", 1, 500.0);
        cache.record_inference("m", 1, 300.0);
    }

    #[test]
    fn test_cache_warmup() {
        let mut cache = ModelCache::new(1000).with_warmup_iterations(5);
        cache.load(make_meta("m", 1, 100), 1).unwrap();
        cache.mark_warmed_up("m", 1);
    }

    #[test]
    fn test_version_registry() {
        let mut reg = VersionRegistry::new();
        reg.register("resnet", 1, 100);
        assert_eq!(reg.active_version("resnet"), Some(1));

        reg.register("resnet", 2, 200);
        assert_eq!(reg.active_version("resnet"), Some(2));
        assert_eq!(reg.all_versions("resnet"), vec![1, 2]);
    }

    #[test]
    fn test_version_registry_unknown() {
        let reg = VersionRegistry::new();
        assert_eq!(reg.active_version("nonexist"), None);
        assert!(reg.all_versions("nonexist").is_empty());
    }

    #[test]
    fn test_version_registry_count() {
        let mut reg = VersionRegistry::new();
        reg.register("a", 1, 1);
        reg.register("b", 1, 2);
        assert_eq!(reg.model_count(), 2);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            total_models: 2,
            total_memory_bytes: 500,
            budget_bytes: 1000,
            hits: 8,
            misses: 2,
            evictions: 1,
        };
        assert!((stats.hit_rate() - 0.8).abs() < 1e-10);
        assert!((stats.utilisation() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_display_impls() {
        assert!(format!("{}", ModelFormat::Onnx).contains("ONNX"));
        assert!(format!("{}", EvictionPolicy::Lru).contains("LRU"));

        let meta = make_meta("test", 1, 100);
        assert!(format!("{meta}").contains("test"));

        let cache = ModelCache::new(1000);
        assert!(format!("{cache}").contains("ModelCache"));

        let stats = CacheStats {
            total_models: 0, total_memory_bytes: 0, budget_bytes: 100,
            hits: 0, misses: 0, evictions: 0,
        };
        assert!(format!("{stats}").contains("CacheStats"));

        let reg = VersionRegistry::new();
        assert!(format!("{reg}").contains("VersionRegistry"));
    }

    #[test]
    fn test_cache_is_empty() {
        let cache = ModelCache::new(1000);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_version_registry_default() {
        let reg = VersionRegistry::default();
        assert_eq!(reg.model_count(), 0);
    }
}
