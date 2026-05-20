//! Memory Manager for JouleDB
//!
//! This module provides memory tracking and eviction policies for managing
//! memory usage in analytical workloads.
//!
//! ## Features
//!
//! - **Memory Tracking**: Monitor memory usage across stores
//! - **Budget Enforcement**: Set limits and enforce them
//! - **Eviction Policies**: LRU, FIFO, and priority-based eviction
//! - **Compaction Triggers**: Automatic compaction on memory pressure
//!
//! ## Example
//!
//! ```rust,ignore
//! let mut manager = MemoryManager::with_budget(1024 * 1024 * 1024); // 1GB
//!
//! // Register stores
//! manager.register("main", store.memory_usage());
//!
//! // Check memory pressure
//! if manager.memory_pressure() > 0.8 {
//!     manager.trigger_eviction();
//! }
//! ```

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

/// Memory budget and tracking manager
pub struct MemoryManager {
    /// Maximum memory budget in bytes
    budget_bytes: usize,
    /// Current tracked memory usage
    current_usage: AtomicUsize,
    /// Per-store memory tracking
    store_usage: HashMap<String, StoreMemoryInfo>,
    /// Eviction policy
    policy: EvictionPolicy,
    /// Access history for LRU tracking
    access_history: VecDeque<AccessRecord>,
    /// Maximum access history entries
    max_history: usize,
    /// Compaction threshold (0.0 to 1.0)
    compaction_threshold: f64,
}

impl MemoryManager {
    /// Create a memory manager with the given budget
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            current_usage: AtomicUsize::new(0),
            store_usage: HashMap::new(),
            policy: EvictionPolicy::LRU,
            access_history: VecDeque::new(),
            max_history: 10000,
            compaction_threshold: 0.8,
        }
    }

    /// Create with custom eviction policy
    pub fn with_policy(budget_bytes: usize, policy: EvictionPolicy) -> Self {
        let mut manager = Self::with_budget(budget_bytes);
        manager.policy = policy;
        manager
    }

    /// Get the memory budget in bytes
    pub fn budget(&self) -> usize {
        self.budget_bytes
    }

    /// Set the memory budget
    pub fn set_budget(&mut self, budget_bytes: usize) {
        self.budget_bytes = budget_bytes;
    }

    /// Get current memory usage in bytes
    pub fn current_usage(&self) -> usize {
        self.current_usage.load(Ordering::Relaxed)
    }

    /// Get available memory in bytes
    pub fn available(&self) -> usize {
        self.budget_bytes.saturating_sub(self.current_usage())
    }

    /// Get memory pressure as a ratio (0.0 to 1.0+)
    pub fn memory_pressure(&self) -> f64 {
        if self.budget_bytes == 0 {
            return 1.0;
        }
        self.current_usage() as f64 / self.budget_bytes as f64
    }

    /// Check if memory is under pressure (above compaction threshold)
    pub fn is_under_pressure(&self) -> bool {
        self.memory_pressure() >= self.compaction_threshold
    }

    /// Register a store's memory usage
    pub fn register(&mut self, name: &str, usage_bytes: usize) {
        self.current_usage.fetch_add(usage_bytes, Ordering::Relaxed);
        self.store_usage.insert(
            name.to_string(),
            StoreMemoryInfo {
                name: name.to_string(),
                usage_bytes,
                last_access: Instant::now(),
                access_count: 1,
                priority: StorePriority::Normal,
            },
        );
    }

    /// Update a store's memory usage
    pub fn update_usage(&mut self, name: &str, new_usage_bytes: usize) {
        if let Some(info) = self.store_usage.get_mut(name) {
            let old_usage = info.usage_bytes;
            info.usage_bytes = new_usage_bytes;
            info.last_access = Instant::now();
            info.access_count += 1;

            // Update global usage
            if new_usage_bytes > old_usage {
                self.current_usage
                    .fetch_add(new_usage_bytes - old_usage, Ordering::Relaxed);
            } else {
                self.current_usage
                    .fetch_sub(old_usage - new_usage_bytes, Ordering::Relaxed);
            }
        }
    }

    /// Record an access for LRU tracking
    pub fn record_access(&mut self, store_name: &str) {
        let now = Instant::now();

        if let Some(info) = self.store_usage.get_mut(store_name) {
            info.last_access = now;
            info.access_count += 1;
        }

        // Add to history
        self.access_history.push_back(AccessRecord {
            store_name: store_name.to_string(),
            timestamp: now,
        });

        // Trim history if too long
        while self.access_history.len() > self.max_history {
            self.access_history.pop_front();
        }
    }

    /// Set store priority for eviction decisions
    pub fn set_priority(&mut self, name: &str, priority: StorePriority) {
        if let Some(info) = self.store_usage.get_mut(name) {
            info.priority = priority;
        }
    }

    /// Get eviction candidates based on current policy
    pub fn get_eviction_candidates(&self, target_bytes: usize) -> Vec<EvictionCandidate> {
        let mut candidates: Vec<_> = self
            .store_usage
            .values()
            .filter(|info| info.priority != StorePriority::Pinned)
            .map(|info| EvictionCandidate {
                store_name: info.name.clone(),
                usage_bytes: info.usage_bytes,
                last_access: info.last_access,
                access_count: info.access_count,
                priority: info.priority,
            })
            .collect();

        // Sort by eviction policy
        match self.policy {
            EvictionPolicy::LRU => {
                candidates.sort_by(|a, b| a.last_access.cmp(&b.last_access));
            }
            EvictionPolicy::LFU => {
                candidates.sort_by_key(|c| c.access_count);
            }
            EvictionPolicy::FIFO => {
                // Already in insertion order from HashMap iteration
            }
            EvictionPolicy::SizeFirst => {
                candidates.sort_by(|a, b| b.usage_bytes.cmp(&a.usage_bytes));
            }
            EvictionPolicy::PriorityBased => {
                candidates.sort_by_key(|c| c.priority);
            }
        }

        // Collect candidates until we reach target bytes
        let mut collected_bytes = 0;
        let mut selected = Vec::new();

        for candidate in candidates {
            if collected_bytes >= target_bytes {
                break;
            }
            collected_bytes += candidate.usage_bytes;
            selected.push(candidate);
        }

        selected
    }

    /// Calculate how many bytes need to be freed to get below threshold
    pub fn bytes_to_free(&self) -> usize {
        let target = (self.budget_bytes as f64 * (self.compaction_threshold - 0.1)) as usize;
        self.current_usage().saturating_sub(target)
    }

    /// Unregister a store (after eviction)
    pub fn unregister(&mut self, name: &str) {
        if let Some(info) = self.store_usage.remove(name) {
            self.current_usage
                .fetch_sub(info.usage_bytes, Ordering::Relaxed);
        }
    }

    /// Get statistics about memory usage
    pub fn stats(&self) -> MemoryStats {
        let store_stats: Vec<_> = self
            .store_usage
            .values()
            .map(|info| (info.name.clone(), info.usage_bytes))
            .collect();

        MemoryStats {
            budget_bytes: self.budget_bytes,
            current_usage_bytes: self.current_usage(),
            available_bytes: self.available(),
            pressure: self.memory_pressure(),
            store_count: self.store_usage.len(),
            stores: store_stats,
            policy: self.policy,
        }
    }

    /// Get a summary string for logging
    pub fn summary(&self) -> String {
        let pressure = self.memory_pressure();
        let status = if pressure >= 1.0 {
            "OVER BUDGET"
        } else if pressure >= self.compaction_threshold {
            "HIGH PRESSURE"
        } else if pressure >= 0.5 {
            "MODERATE"
        } else {
            "OK"
        };

        format!(
            "Memory: {:.1}MB / {:.1}MB ({:.1}%) - {} [{}]",
            self.current_usage() as f64 / 1_000_000.0,
            self.budget_bytes as f64 / 1_000_000.0,
            pressure * 100.0,
            status,
            format!("{:?}", self.policy),
        )
    }
}

impl Default for MemoryManager {
    fn default() -> Self {
        // Default to 1GB budget
        Self::with_budget(1024 * 1024 * 1024)
    }
}

/// Eviction policy for the memory manager
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used - evict oldest accessed first
    LRU,
    /// Least Frequently Used - evict least accessed first
    LFU,
    /// First In First Out - evict oldest registered first
    FIFO,
    /// Size First - evict largest stores first
    SizeFirst,
    /// Priority Based - evict by priority level
    PriorityBased,
}

/// Store priority for eviction decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StorePriority {
    /// Can be evicted any time
    Low = 0,
    /// Normal priority
    Normal = 1,
    /// Prefer to keep
    High = 2,
    /// Never evict (critical data)
    Pinned = 3,
}

/// Memory info for a single store
#[derive(Debug, Clone)]
struct StoreMemoryInfo {
    name: String,
    usage_bytes: usize,
    last_access: Instant,
    access_count: u64,
    priority: StorePriority,
}

/// Access record for LRU tracking
#[derive(Debug)]
struct AccessRecord {
    store_name: String,
    timestamp: Instant,
}

/// Candidate for eviction
#[derive(Debug, Clone)]
pub struct EvictionCandidate {
    pub store_name: String,
    pub usage_bytes: usize,
    pub last_access: Instant,
    pub access_count: u64,
    pub priority: StorePriority,
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub budget_bytes: usize,
    pub current_usage_bytes: usize,
    pub available_bytes: usize,
    pub pressure: f64,
    pub store_count: usize,
    pub stores: Vec<(String, usize)>,
    pub policy: EvictionPolicy,
}

/// Estimate memory usage of common structures
pub struct MemoryEstimator;

impl MemoryEstimator {
    /// Estimate memory for a Vec<f64>
    pub fn vec_f64(len: usize) -> usize {
        std::mem::size_of::<Vec<f64>>() + len * std::mem::size_of::<f64>()
    }

    /// Estimate memory for a Vec<u64>
    pub fn vec_u64(len: usize) -> usize {
        std::mem::size_of::<Vec<u64>>() + len * std::mem::size_of::<u64>()
    }

    /// Estimate memory for a HashMap<String, T>
    pub fn hashmap_string<T>(entries: usize, avg_key_len: usize) -> usize {
        // HashMap overhead + entries * (key + value + bucket overhead)
        64 + entries * (avg_key_len + std::mem::size_of::<T>() + 16)
    }

    /// Estimate memory for a columnar store
    pub fn columnar_store(num_columns: usize, avg_rows_per_column: usize) -> usize {
        // ColumnarStore overhead
        let base = std::mem::size_of::<crate::ColumnarStore>();

        // Per column: Vec<f64> values + Vec<RecordId> record_ids + stats
        let per_column =
            Self::vec_f64(avg_rows_per_column) + Self::vec_u64(avg_rows_per_column) + 64; // min, max, sum, tombstones overhead

        base + num_columns * per_column
    }

    /// Estimate memory for an AmorphicStore record
    pub fn amorphic_record(avg_fields: usize, avg_string_len: usize) -> usize {
        // Record overhead + fields
        64 + avg_fields * (16 + avg_string_len + 8) // key + value
    }
}

// =============================================================================
// AUTO-EVICTION INTEGRATION
// =============================================================================

/// Trait for stores that support memory-managed eviction
pub trait MemoryManagedStore {
    /// Get current memory usage in bytes
    fn memory_usage(&self) -> usize;

    /// Evict data to free up memory, returns bytes actually freed
    fn evict(&mut self, target_bytes: usize) -> usize;

    /// Compact internal structures (e.g., remove tombstones)
    fn compact(&mut self) -> usize;

    /// Get a unique identifier for this store
    fn store_id(&self) -> &str;
}

/// Result of an eviction operation
#[derive(Debug, Clone)]
pub struct EvictionResult {
    /// Bytes freed by eviction
    pub bytes_freed: usize,
    /// Number of items evicted
    pub items_evicted: usize,
    /// Time taken in milliseconds
    pub duration_ms: f64,
    /// Whether the eviction was triggered automatically
    pub was_automatic: bool,
}

/// Configuration for auto-eviction
#[derive(Debug, Clone)]
pub struct AutoEvictionConfig {
    /// Enable automatic eviction
    pub enabled: bool,
    /// Threshold to trigger eviction (0.0 to 1.0)
    pub trigger_threshold: f64,
    /// Target memory usage after eviction (0.0 to 1.0)
    pub target_threshold: f64,
    /// Minimum bytes to free per eviction
    pub min_eviction_bytes: usize,
    /// Maximum items to evict per operation
    pub max_eviction_items: usize,
    /// Enable compaction during eviction
    pub compact_on_evict: bool,
}

impl Default for AutoEvictionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            trigger_threshold: 0.85,
            target_threshold: 0.70,
            min_eviction_bytes: 1024 * 1024, // 1MB minimum
            max_eviction_items: 1000,
            compact_on_evict: true,
        }
    }
}

/// Memory-managed store wrapper with auto-eviction
pub struct ManagedStore<S: MemoryManagedStore> {
    /// The underlying store
    store: S,
    /// Memory manager
    memory: MemoryManager,
    /// Auto-eviction configuration
    config: AutoEvictionConfig,
    /// Eviction statistics
    stats: ManagedStoreStats,
}

/// Statistics for managed store
#[derive(Debug, Clone, Default)]
pub struct ManagedStoreStats {
    /// Total evictions performed
    pub total_evictions: u64,
    /// Total bytes evicted
    pub total_bytes_evicted: usize,
    /// Total compactions performed
    pub total_compactions: u64,
    /// Total bytes freed by compaction
    pub total_bytes_compacted: usize,
    /// Auto-eviction trigger count
    pub auto_eviction_triggers: u64,
}

impl<S: MemoryManagedStore> ManagedStore<S> {
    /// Create a new managed store with default configuration
    pub fn new(store: S, budget_bytes: usize) -> Self {
        let mut memory = MemoryManager::with_budget(budget_bytes);
        let usage = store.memory_usage();
        memory.register(store.store_id(), usage);

        Self {
            store,
            memory,
            config: AutoEvictionConfig::default(),
            stats: ManagedStoreStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(store: S, budget_bytes: usize, config: AutoEvictionConfig) -> Self {
        let mut managed = Self::new(store, budget_bytes);
        managed.config = config;
        managed
    }

    /// Get reference to underlying store
    pub fn store(&self) -> &S {
        &self.store
    }

    /// Get mutable reference to underlying store
    pub fn store_mut(&mut self) -> &mut S {
        &mut self.store
    }

    /// Get memory manager reference
    pub fn memory_manager(&self) -> &MemoryManager {
        &self.memory
    }

    /// Get current memory usage
    pub fn memory_usage(&self) -> usize {
        self.store.memory_usage()
    }

    /// Get memory pressure (0.0 to 1.0+)
    pub fn memory_pressure(&self) -> f64 {
        self.memory.memory_pressure()
    }

    /// Check if auto-eviction should be triggered
    pub fn should_evict(&self) -> bool {
        self.config.enabled && self.memory_pressure() >= self.config.trigger_threshold
    }

    /// Update memory tracking (call after writes)
    pub fn track_memory_change(&mut self) {
        let usage = self.store.memory_usage();
        self.memory.update_usage(self.store.store_id(), usage);
    }

    /// Record access for LRU tracking
    pub fn record_access(&mut self) {
        self.memory.record_access(self.store.store_id());
    }

    /// Trigger eviction if needed, returns result if eviction occurred
    pub fn check_and_evict(&mut self) -> Option<EvictionResult> {
        if !self.should_evict() {
            return None;
        }

        self.stats.auto_eviction_triggers += 1;
        Some(self.evict())
    }

    /// Force eviction to reach target threshold
    pub fn evict(&mut self) -> EvictionResult {
        let start = Instant::now();

        // Calculate target
        let target_usage = (self.memory.budget() as f64 * self.config.target_threshold) as usize;
        let current_usage = self.memory_usage();
        let target_free = current_usage.saturating_sub(target_usage);
        let target_free = target_free.max(self.config.min_eviction_bytes);

        // Evict from store
        let bytes_freed = self.store.evict(target_free);
        let mut items_evicted = 0;

        // Optionally compact
        let compact_freed = if self.config.compact_on_evict {
            let freed = self.store.compact();
            self.stats.total_compactions += 1;
            self.stats.total_bytes_compacted += freed;
            freed
        } else {
            0
        };

        let total_freed = bytes_freed + compact_freed;

        // Update tracking
        self.track_memory_change();

        // Update stats
        self.stats.total_evictions += 1;
        self.stats.total_bytes_evicted += total_freed;

        if total_freed > 0 {
            items_evicted = (total_freed / 100).max(1); // Estimate
        }

        EvictionResult {
            bytes_freed: total_freed,
            items_evicted,
            duration_ms: start.elapsed().as_secs_f64() * 1000.0,
            was_automatic: true,
        }
    }

    /// Manually trigger compaction
    pub fn compact(&mut self) -> usize {
        let freed = self.store.compact();
        self.stats.total_compactions += 1;
        self.stats.total_bytes_compacted += freed;
        self.track_memory_change();
        freed
    }

    /// Get managed store statistics
    pub fn stats(&self) -> &ManagedStoreStats {
        &self.stats
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "ManagedStore[{}]: {} (evictions={}, compactions={})",
            self.store.store_id(),
            self.memory.summary(),
            self.stats.total_evictions,
            self.stats.total_compactions
        )
    }
}

/// Helper to create eviction callbacks for async eviction
pub type EvictionCallback = Box<dyn Fn(EvictionResult) + Send + Sync>;

/// Builder for ManagedStore
pub struct ManagedStoreBuilder<S: MemoryManagedStore> {
    store: S,
    budget_bytes: usize,
    config: AutoEvictionConfig,
    policy: EvictionPolicy,
}

impl<S: MemoryManagedStore> ManagedStoreBuilder<S> {
    /// Create a new builder
    pub fn new(store: S) -> Self {
        Self {
            store,
            budget_bytes: 1024 * 1024 * 1024, // 1GB default
            config: AutoEvictionConfig::default(),
            policy: EvictionPolicy::LRU,
        }
    }

    /// Set memory budget
    pub fn budget(mut self, bytes: usize) -> Self {
        self.budget_bytes = bytes;
        self
    }

    /// Set eviction policy
    pub fn policy(mut self, policy: EvictionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Enable or disable auto-eviction
    pub fn auto_eviction(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }

    /// Set trigger threshold
    pub fn trigger_threshold(mut self, threshold: f64) -> Self {
        self.config.trigger_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set target threshold
    pub fn target_threshold(mut self, threshold: f64) -> Self {
        self.config.target_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Enable compaction during eviction
    pub fn compact_on_evict(mut self, enabled: bool) -> Self {
        self.config.compact_on_evict = enabled;
        self
    }

    /// Build the managed store
    pub fn build(self) -> ManagedStore<S> {
        let mut memory = MemoryManager::with_policy(self.budget_bytes, self.policy);
        let usage = self.store.memory_usage();
        memory.register(self.store.store_id(), usage);

        ManagedStore {
            store: self.store,
            memory,
            config: self.config,
            stats: ManagedStoreStats::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_manager_basic() {
        let mut manager = MemoryManager::with_budget(1000);

        assert_eq!(manager.budget(), 1000);
        assert_eq!(manager.current_usage(), 0);
        assert_eq!(manager.available(), 1000);
        assert!((manager.memory_pressure() - 0.0).abs() < 0.01);

        // Register a store
        manager.register("store1", 400);
        assert_eq!(manager.current_usage(), 400);
        assert!((manager.memory_pressure() - 0.4).abs() < 0.01);

        // Update usage
        manager.update_usage("store1", 600);
        assert_eq!(manager.current_usage(), 600);

        // Add another store
        manager.register("store2", 300);
        assert_eq!(manager.current_usage(), 900);
        assert!((manager.memory_pressure() - 0.9).abs() < 0.01);
    }

    #[test]
    fn test_eviction_candidates_lru() {
        let mut manager = MemoryManager::with_policy(1000, EvictionPolicy::LRU);

        manager.register("old_store", 200);
        std::thread::sleep(std::time::Duration::from_millis(10));
        manager.register("new_store", 200);

        // Old store should be evicted first
        let candidates = manager.get_eviction_candidates(200);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].store_name, "old_store");
    }

    #[test]
    fn test_eviction_candidates_size_first() {
        let mut manager = MemoryManager::with_policy(1000, EvictionPolicy::SizeFirst);

        manager.register("small", 100);
        manager.register("large", 500);
        manager.register("medium", 250);

        // Largest should be evicted first
        let candidates = manager.get_eviction_candidates(500);
        assert_eq!(candidates[0].store_name, "large");
    }

    #[test]
    fn test_pinned_stores_not_evicted() {
        let mut manager = MemoryManager::with_budget(1000);

        manager.register("pinned", 400);
        manager.set_priority("pinned", StorePriority::Pinned);
        manager.register("normal", 400);

        let candidates = manager.get_eviction_candidates(400);
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].store_name, "normal");
    }

    #[test]
    fn test_memory_pressure() {
        let mut manager = MemoryManager::with_budget(1000);
        manager.compaction_threshold = 0.8;

        manager.register("store", 700);
        assert!(!manager.is_under_pressure());

        manager.update_usage("store", 850);
        assert!(manager.is_under_pressure());
    }

    #[test]
    fn test_memory_estimator() {
        let vec_size = MemoryEstimator::vec_f64(1000);
        assert!(vec_size >= 8000); // At least 8 bytes per f64

        let store_size = MemoryEstimator::columnar_store(10, 1000);
        assert!(store_size > 0);
    }

    #[test]
    fn test_unregister() {
        let mut manager = MemoryManager::with_budget(1000);

        manager.register("store1", 400);
        manager.register("store2", 300);
        assert_eq!(manager.current_usage(), 700);

        manager.unregister("store1");
        assert_eq!(manager.current_usage(), 300);
    }

    // Test auto-eviction integration
    struct MockStore {
        id: String,
        memory_bytes: usize,
        items: Vec<u8>,
    }

    impl MockStore {
        fn new(id: &str, initial_bytes: usize) -> Self {
            Self {
                id: id.to_string(),
                memory_bytes: initial_bytes,
                items: vec![0u8; initial_bytes],
            }
        }
    }

    impl MemoryManagedStore for MockStore {
        fn memory_usage(&self) -> usize {
            self.memory_bytes
        }

        fn evict(&mut self, target_bytes: usize) -> usize {
            let to_evict = target_bytes.min(self.memory_bytes / 2); // Max evict half
            self.memory_bytes = self.memory_bytes.saturating_sub(to_evict);
            self.items.truncate(self.memory_bytes);
            to_evict
        }

        fn compact(&mut self) -> usize {
            // Simulate 10% space savings
            let freed = self.memory_bytes / 10;
            self.memory_bytes = self.memory_bytes.saturating_sub(freed);
            self.items.truncate(self.memory_bytes);
            freed
        }

        fn store_id(&self) -> &str {
            &self.id
        }
    }

    #[test]
    fn test_managed_store_creation() {
        let store = MockStore::new("test", 1000);
        let managed = ManagedStore::new(store, 2000);

        assert_eq!(managed.memory_usage(), 1000);
        assert!((managed.memory_pressure() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_managed_store_eviction_trigger() {
        let store = MockStore::new("test", 1700); // 85% of 2000
        let mut managed = ManagedStore::new(store, 2000);

        // Should trigger eviction at 85%
        assert!(managed.should_evict());

        let result = managed.check_and_evict();
        assert!(result.is_some());

        let result = result.unwrap();
        assert!(result.bytes_freed > 0);
        assert!(managed.memory_pressure() < 0.85);
    }

    #[test]
    fn test_managed_store_no_eviction_needed() {
        let store = MockStore::new("test", 500); // 25% of 2000
        let managed = ManagedStore::new(store, 2000);

        assert!(!managed.should_evict());
    }

    #[test]
    fn test_managed_store_builder() {
        let store = MockStore::new("test", 1000);
        let managed = ManagedStoreBuilder::new(store)
            .budget(5000)
            .policy(EvictionPolicy::SizeFirst)
            .trigger_threshold(0.9)
            .target_threshold(0.7)
            .auto_eviction(true)
            .compact_on_evict(true)
            .build();

        assert_eq!(managed.memory_manager().budget(), 5000);
        assert!(!managed.should_evict()); // 20% < 90%
    }

    #[test]
    fn test_managed_store_stats() {
        let store = MockStore::new("test", 1800);
        let mut managed = ManagedStore::new(store, 2000);

        // Trigger eviction
        let _ = managed.evict();

        let stats = managed.stats();
        assert_eq!(stats.total_evictions, 1);
        assert!(stats.total_bytes_evicted > 0);
    }

    #[test]
    fn test_managed_store_compact() {
        let store = MockStore::new("test", 1000);
        let mut managed = ManagedStore::new(store, 2000);

        let freed = managed.compact();
        assert_eq!(freed, 100); // 10% of 1000

        let stats = managed.stats();
        assert_eq!(stats.total_compactions, 1);
        assert_eq!(stats.total_bytes_compacted, 100);
    }

    #[test]
    fn test_auto_eviction_config() {
        let config = AutoEvictionConfig::default();
        assert!(config.enabled);
        assert!((config.trigger_threshold - 0.85).abs() < 0.01);
        assert!((config.target_threshold - 0.70).abs() < 0.01);
    }
}
