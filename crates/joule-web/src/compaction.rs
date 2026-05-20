//! Compaction strategies — size-tiered compaction, leveled compaction,
//! time-window compaction, compaction scheduling, write amplification
//! estimation, space amplification, compaction statistics.

// ── Errors ───────────────────────────────────────────────────────────────────

/// Errors returned by compaction operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompactionError {
    /// No tables available for compaction.
    NothingToCompact,
    /// Invalid configuration.
    InvalidConfig(String),
    /// Level out of range.
    LevelOutOfRange(usize),
    /// Compaction task failed.
    TaskFailed(String),
}

impl std::fmt::Display for CompactionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NothingToCompact => write!(f, "nothing to compact"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
            Self::LevelOutOfRange(l) => write!(f, "level {l} out of range"),
            Self::TaskFailed(msg) => write!(f, "compaction task failed: {msg}"),
        }
    }
}

impl std::error::Error for CompactionError {}

// ── Table Info ───────────────────────────────────────────────────────────────

/// Metadata for a single SSTable (or equivalent sorted-run file).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableInfo {
    /// Unique table ID.
    pub id: u64,
    /// Level this table belongs to.
    pub level: usize,
    /// Size in bytes.
    pub size_bytes: u64,
    /// Number of entries.
    pub entry_count: u64,
    /// Minimum key (for overlap detection).
    pub min_key: Vec<u8>,
    /// Maximum key (for overlap detection).
    pub max_key: Vec<u8>,
    /// Creation timestamp (epoch millis).
    pub created_at: u64,
    /// Number of tombstones in the table.
    pub tombstone_count: u64,
}

impl TableInfo {
    /// Whether this table's key range overlaps with another.
    pub fn overlaps(&self, other: &TableInfo) -> bool {
        self.min_key <= other.max_key && other.min_key <= self.max_key
    }

    /// Tombstone ratio.
    pub fn tombstone_ratio(&self) -> f64 {
        if self.entry_count == 0 {
            return 0.0;
        }
        self.tombstone_count as f64 / self.entry_count as f64
    }
}

// ── Compaction Task ──────────────────────────────────────────────────────────

/// A compaction task describing which tables to merge.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactionTask {
    /// Unique task ID.
    pub task_id: u64,
    /// Strategy that generated this task.
    pub strategy: StrategyType,
    /// Input tables to be merged.
    pub input_tables: Vec<TableInfo>,
    /// Target level for the output.
    pub output_level: usize,
    /// Estimated output size.
    pub estimated_output_bytes: u64,
    /// Priority (higher = more urgent).
    pub priority: u32,
}

impl CompactionTask {
    /// Total input bytes.
    pub fn input_bytes(&self) -> u64 {
        self.input_tables.iter().map(|t| t.size_bytes).sum()
    }

    /// Total input entries.
    pub fn input_entries(&self) -> u64 {
        self.input_tables.iter().map(|t| t.entry_count).sum()
    }
}

// ── Strategy Type ────────────────────────────────────────────────────────────

/// Type of compaction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrategyType {
    SizeTiered,
    Leveled,
    TimeWindow,
}

impl std::fmt::Display for StrategyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeTiered => write!(f, "size-tiered"),
            Self::Leveled => write!(f, "leveled"),
            Self::TimeWindow => write!(f, "time-window"),
        }
    }
}

// ── Size-Tiered Compaction ──────────────────────────────────────────────────

/// Size-tiered compaction: groups similarly-sized tables and merges them.
#[derive(Debug, Clone)]
pub struct SizeTieredConfig {
    /// Minimum number of similar-sized tables to trigger compaction.
    pub min_threshold: usize,
    /// Maximum number of tables to include in one compaction.
    pub max_threshold: usize,
    /// Size ratio tolerance: tables within this ratio are "similar".
    /// E.g., 1.5 means a table up to 1.5x the size of the smallest in the
    /// group is still considered similar.
    pub bucket_ratio: f64,
    /// Minimum table size to consider for compaction.
    pub min_table_size: u64,
}

impl Default for SizeTieredConfig {
    fn default() -> Self {
        Self {
            min_threshold: 4,
            max_threshold: 32,
            bucket_ratio: 1.5,
            min_table_size: 0,
        }
    }
}

/// Pick tables for size-tiered compaction.
pub fn size_tiered_pick(
    tables: &[TableInfo],
    config: &SizeTieredConfig,
) -> Result<Vec<TableInfo>, CompactionError> {
    if tables.len() < config.min_threshold {
        return Err(CompactionError::NothingToCompact);
    }

    // Sort by size.
    let mut sorted: Vec<&TableInfo> = tables
        .iter()
        .filter(|t| t.size_bytes >= config.min_table_size)
        .collect();
    sorted.sort_by_key(|t| t.size_bytes);

    // Find the best bucket of similar-sized tables.
    let mut best_bucket: Vec<&TableInfo> = Vec::new();

    for i in 0..sorted.len() {
        let base_size = sorted[i].size_bytes.max(1);
        let mut bucket: Vec<&TableInfo> = vec![sorted[i]];

        for item in sorted.iter().skip(i + 1) {
            let ratio = item.size_bytes as f64 / base_size as f64;
            if ratio <= config.bucket_ratio {
                bucket.push(item);
                if bucket.len() >= config.max_threshold {
                    break;
                }
            } else {
                break;
            }
        }

        if bucket.len() >= config.min_threshold && bucket.len() > best_bucket.len() {
            best_bucket = bucket;
        }
    }

    if best_bucket.len() < config.min_threshold {
        return Err(CompactionError::NothingToCompact);
    }

    Ok(best_bucket.into_iter().cloned().collect())
}

// ── Leveled Compaction ──────────────────────────────────────────────────────

/// Leveled compaction: each level has a size budget; when exceeded, tables are
/// compacted into the next level.
#[derive(Debug, Clone)]
pub struct LeveledConfig {
    /// Maximum number of levels.
    pub max_levels: usize,
    /// Target size for level 0 in bytes.
    pub level0_target_bytes: u64,
    /// Size multiplier between levels (e.g. 10 means L1 is 10x L0).
    pub level_size_multiplier: u64,
    /// Maximum number of L0 tables before compaction.
    pub level0_file_trigger: usize,
}

impl Default for LeveledConfig {
    fn default() -> Self {
        Self {
            max_levels: 7,
            level0_target_bytes: 64 * 1024 * 1024, // 64 MB
            level_size_multiplier: 10,
            level0_file_trigger: 4,
        }
    }
}

/// Compute the target size for a given level.
pub fn level_target_bytes(config: &LeveledConfig, level: usize) -> u64 {
    if level == 0 {
        return config.level0_target_bytes;
    }
    let mut target = config.level0_target_bytes;
    for _ in 0..level {
        target = target.saturating_mul(config.level_size_multiplier);
    }
    target
}

/// Check if a level needs compaction under leveled strategy.
pub fn leveled_needs_compaction(
    level: usize,
    tables: &[TableInfo],
    config: &LeveledConfig,
) -> bool {
    if level == 0 {
        return tables.len() >= config.level0_file_trigger;
    }
    let total_bytes: u64 = tables.iter().map(|t| t.size_bytes).sum();
    total_bytes > level_target_bytes(config, level)
}

/// Pick a table from `level` and find overlapping tables in `next_level_tables`
/// for leveled compaction.
pub fn leveled_pick(
    level_tables: &[TableInfo],
    next_level_tables: &[TableInfo],
    level: usize,
) -> Result<CompactionTask, CompactionError> {
    if level_tables.is_empty() {
        return Err(CompactionError::NothingToCompact);
    }

    // Pick the table with the smallest key range (least overlap).
    let picked = level_tables
        .iter()
        .min_by_key(|t| t.max_key.len() + t.min_key.len())
        .unwrap()
        .clone();

    // Find overlapping tables in the next level.
    let overlapping: Vec<TableInfo> = next_level_tables
        .iter()
        .filter(|t| picked.overlaps(t))
        .cloned()
        .collect();

    let mut inputs = vec![picked];
    inputs.extend(overlapping);

    let estimated = inputs.iter().map(|t| t.size_bytes).sum();

    Ok(CompactionTask {
        task_id: 0,
        strategy: StrategyType::Leveled,
        input_tables: inputs,
        output_level: level + 1,
        estimated_output_bytes: estimated,
        priority: if level == 0 { 10 } else { 5 },
    })
}

// ── Time-Window Compaction ──────────────────────────────────────────────────

/// Time-window compaction: groups tables by creation time windows.
#[derive(Debug, Clone)]
pub struct TimeWindowConfig {
    /// Window size in milliseconds.
    pub window_size_ms: u64,
    /// Minimum tables in a window to trigger compaction.
    pub min_threshold: usize,
}

impl Default for TimeWindowConfig {
    fn default() -> Self {
        Self {
            window_size_ms: 3600 * 1000, // 1 hour
            min_threshold: 4,
        }
    }
}

/// Pick tables for time-window compaction.
pub fn time_window_pick(
    tables: &[TableInfo],
    config: &TimeWindowConfig,
) -> Result<Vec<TableInfo>, CompactionError> {
    if tables.is_empty() {
        return Err(CompactionError::NothingToCompact);
    }

    // Group tables by window.
    let mut windows: std::collections::HashMap<u64, Vec<&TableInfo>> =
        std::collections::HashMap::new();
    for table in tables {
        let window_id = table.created_at / config.window_size_ms;
        windows.entry(window_id).or_default().push(table);
    }

    // Find the largest window meeting threshold.
    let best_window = windows
        .values()
        .filter(|w| w.len() >= config.min_threshold)
        .max_by_key(|w| w.len());

    match best_window {
        Some(w) => Ok(w.iter().map(|t| (*t).clone()).collect()),
        None => Err(CompactionError::NothingToCompact),
    }
}

// ── Write & Space Amplification ─────────────────────────────────────────────

/// Estimate write amplification for a set of levels.
///
/// Write amplification = total bytes written / bytes of original data.
/// For leveled compaction, this is approximately the level size multiplier.
pub fn estimate_write_amplification(
    levels: &[Vec<TableInfo>],
    config: &LeveledConfig,
) -> f64 {
    if levels.is_empty() {
        return 1.0;
    }
    // In leveled compaction, each byte is rewritten ~multiplier times per level.
    let non_empty = levels.iter().filter(|l| !l.is_empty()).count();
    if non_empty <= 1 {
        return 1.0;
    }
    // Simplified: multiplier * (non_empty_levels - 1).
    config.level_size_multiplier as f64 * (non_empty - 1) as f64
}

/// Estimate space amplification.
///
/// Space amplification = total on-disk size / logical data size.
/// Higher means more wasted space from old versions and tombstones.
pub fn estimate_space_amplification(levels: &[Vec<TableInfo>]) -> f64 {
    let total_bytes: u64 = levels
        .iter()
        .flat_map(|l| l.iter())
        .map(|t| t.size_bytes)
        .sum();

    // Logical size = entries in the last (deepest) non-empty level.
    let deepest_bytes: u64 = levels
        .iter()
        .rev()
        .find(|l| !l.is_empty())
        .map_or(0, |l| l.iter().map(|t| t.size_bytes).sum());

    if deepest_bytes == 0 {
        return 1.0;
    }

    total_bytes as f64 / deepest_bytes as f64
}

// ── Compaction Statistics ────────────────────────────────────────────────────

/// Aggregate compaction statistics.
#[derive(Debug, Clone, Default)]
pub struct CompactionStats {
    /// Total compaction runs completed.
    pub total_runs: u64,
    /// Total bytes read during compaction.
    pub bytes_read: u64,
    /// Total bytes written during compaction.
    pub bytes_written: u64,
    /// Total tables merged.
    pub tables_merged: u64,
    /// Total tables produced.
    pub tables_produced: u64,
    /// Estimated write amplification.
    pub write_amplification: f64,
    /// Estimated space amplification.
    pub space_amplification: f64,
    /// Number of pending tasks.
    pub pending_tasks: usize,
}

// ── Compaction Scheduler ────────────────────────────────────────────────────

/// A compaction scheduler that manages pending compaction tasks.
#[derive(Debug)]
pub struct CompactionScheduler {
    /// Pending tasks sorted by priority (highest first).
    pending: Vec<CompactionTask>,
    /// Next task ID.
    next_task_id: u64,
    /// Completed task statistics.
    stats: CompactionStats,
    /// Maximum concurrent compactions.
    max_concurrent: usize,
    /// Currently running.
    running: usize,
}

impl CompactionScheduler {
    /// Create a new scheduler.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            pending: Vec::new(),
            next_task_id: 1,
            stats: CompactionStats::default(),
            max_concurrent,
            running: 0,
        }
    }

    /// Submit a compaction task.
    pub fn submit(&mut self, mut task: CompactionTask) {
        task.task_id = self.next_task_id;
        self.next_task_id += 1;
        self.pending.push(task);
        // Sort by priority descending.
        self.pending.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Dequeue the next task if there's capacity.
    pub fn next_task(&mut self) -> Option<CompactionTask> {
        if self.running >= self.max_concurrent || self.pending.is_empty() {
            return None;
        }
        self.running += 1;
        Some(self.pending.remove(0))
    }

    /// Mark a task as completed, updating stats.
    pub fn complete_task(&mut self, task: &CompactionTask, output_bytes: u64, output_tables: u64) {
        self.running = self.running.saturating_sub(1);
        self.stats.total_runs += 1;
        self.stats.bytes_read += task.input_bytes();
        self.stats.bytes_written += output_bytes;
        self.stats.tables_merged += task.input_tables.len() as u64;
        self.stats.tables_produced += output_tables;
    }

    /// Number of pending tasks.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Number of currently running tasks.
    pub fn running_count(&self) -> usize {
        self.running
    }

    /// Whether the scheduler can accept more tasks.
    pub fn has_capacity(&self) -> bool {
        self.running < self.max_concurrent
    }

    /// Get compaction statistics.
    pub fn stats(&self) -> &CompactionStats {
        &self.stats
    }

    /// Update amplification estimates.
    pub fn update_amplification(&mut self, levels: &[Vec<TableInfo>], config: &LeveledConfig) {
        self.stats.write_amplification = estimate_write_amplification(levels, config);
        self.stats.space_amplification = estimate_space_amplification(levels);
        self.stats.pending_tasks = self.pending.len();
    }

    /// Clear all pending tasks.
    pub fn clear_pending(&mut self) {
        self.pending.clear();
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_table(id: u64, size: u64, min: &[u8], max: &[u8]) -> TableInfo {
        TableInfo {
            id,
            level: 0,
            size_bytes: size,
            entry_count: size / 100,
            min_key: min.to_vec(),
            max_key: max.to_vec(),
            created_at: id * 1000,
            tombstone_count: 0,
        }
    }

    #[test]
    fn table_overlap() {
        let a = make_table(1, 100, b"a", b"m");
        let b = make_table(2, 100, b"k", b"z");
        assert!(a.overlaps(&b));
    }

    #[test]
    fn table_no_overlap() {
        let a = make_table(1, 100, b"a", b"f");
        let b = make_table(2, 100, b"m", b"z");
        assert!(!a.overlaps(&b));
    }

    #[test]
    fn tombstone_ratio() {
        let mut t = make_table(1, 1000, b"a", b"z");
        t.tombstone_count = 5;
        t.entry_count = 10;
        assert!((t.tombstone_ratio() - 0.5).abs() < 0.01);
    }

    #[test]
    fn size_tiered_pick_basic() {
        let tables: Vec<TableInfo> = (0..6)
            .map(|i| make_table(i, 100 + i * 10, b"a", b"z"))
            .collect();
        let config = SizeTieredConfig {
            min_threshold: 4,
            max_threshold: 10,
            bucket_ratio: 1.5,
            min_table_size: 0,
        };
        let picked = size_tiered_pick(&tables, &config).unwrap();
        assert!(picked.len() >= 4);
    }

    #[test]
    fn size_tiered_not_enough_tables() {
        let tables = vec![make_table(1, 100, b"a", b"z")];
        let config = SizeTieredConfig::default();
        assert_eq!(
            size_tiered_pick(&tables, &config),
            Err(CompactionError::NothingToCompact)
        );
    }

    #[test]
    fn size_tiered_varied_sizes() {
        // Two groups: small (100-150) and large (10000-15000).
        let mut tables = Vec::new();
        for i in 0..5 {
            tables.push(make_table(i, 100 + i * 10, b"a", b"z"));
        }
        tables.push(make_table(10, 10000, b"a", b"z"));
        tables.push(make_table(11, 15000, b"a", b"z"));
        let config = SizeTieredConfig {
            min_threshold: 4,
            bucket_ratio: 1.5,
            ..Default::default()
        };
        let picked = size_tiered_pick(&tables, &config).unwrap();
        // Should pick the small group.
        assert!(picked.iter().all(|t| t.size_bytes < 1000));
    }

    #[test]
    fn level_target_bytes_computation() {
        let config = LeveledConfig {
            level0_target_bytes: 100,
            level_size_multiplier: 10,
            ..Default::default()
        };
        assert_eq!(level_target_bytes(&config, 0), 100);
        assert_eq!(level_target_bytes(&config, 1), 1000);
        assert_eq!(level_target_bytes(&config, 2), 10000);
    }

    #[test]
    fn leveled_needs_compaction_l0() {
        let config = LeveledConfig {
            level0_file_trigger: 4,
            ..Default::default()
        };
        let tables: Vec<TableInfo> = (0..4)
            .map(|i| make_table(i, 100, b"a", b"z"))
            .collect();
        assert!(leveled_needs_compaction(0, &tables, &config));
    }

    #[test]
    fn leveled_needs_compaction_l1() {
        let config = LeveledConfig {
            level0_target_bytes: 100,
            level_size_multiplier: 10,
            ..Default::default()
        };
        let tables = vec![make_table(1, 2000, b"a", b"z")];
        assert!(leveled_needs_compaction(1, &tables, &config));
    }

    #[test]
    fn leveled_pick_basic() {
        let l0 = vec![make_table(1, 100, b"a", b"m")];
        let l1 = vec![
            make_table(2, 200, b"a", b"f"),
            make_table(3, 200, b"g", b"z"),
        ];
        let task = leveled_pick(&l0, &l1, 0).unwrap();
        assert_eq!(task.output_level, 1);
        assert!(task.input_tables.len() >= 2);
    }

    #[test]
    fn leveled_pick_empty() {
        let l0: Vec<TableInfo> = Vec::new();
        assert_eq!(
            leveled_pick(&l0, &[], 0),
            Err(CompactionError::NothingToCompact)
        );
    }

    #[test]
    fn time_window_pick_basic() {
        let config = TimeWindowConfig {
            window_size_ms: 1000,
            min_threshold: 3,
        };
        // 5 tables in the same window.
        let tables: Vec<TableInfo> = (0..5)
            .map(|i| {
                let mut t = make_table(i, 100, b"a", b"z");
                t.created_at = 500 + i * 100; // All within [0, 1000).
                t
            })
            .collect();
        let picked = time_window_pick(&tables, &config).unwrap();
        assert!(picked.len() >= 3);
    }

    #[test]
    fn time_window_not_enough() {
        let config = TimeWindowConfig {
            window_size_ms: 1000,
            min_threshold: 10,
        };
        let tables = vec![make_table(1, 100, b"a", b"z")];
        assert_eq!(
            time_window_pick(&tables, &config),
            Err(CompactionError::NothingToCompact)
        );
    }

    #[test]
    fn write_amplification() {
        let config = LeveledConfig {
            level_size_multiplier: 10,
            ..Default::default()
        };
        let levels = vec![
            vec![make_table(1, 100, b"a", b"z")],
            vec![make_table(2, 1000, b"a", b"z")],
            vec![make_table(3, 10000, b"a", b"z")],
        ];
        let wa = estimate_write_amplification(&levels, &config);
        assert!(wa > 1.0);
    }

    #[test]
    fn space_amplification() {
        let levels = vec![
            vec![make_table(1, 100, b"a", b"z")],
            vec![make_table(2, 500, b"a", b"z")],
        ];
        let sa = estimate_space_amplification(&levels);
        assert!(sa > 1.0);
    }

    #[test]
    fn scheduler_submit_and_dequeue() {
        let mut sched = CompactionScheduler::new(2);
        let task = CompactionTask {
            task_id: 0,
            strategy: StrategyType::SizeTiered,
            input_tables: vec![make_table(1, 100, b"a", b"z")],
            output_level: 1,
            estimated_output_bytes: 100,
            priority: 5,
        };
        sched.submit(task);
        assert_eq!(sched.pending_count(), 1);
        let t = sched.next_task().unwrap();
        assert_eq!(t.priority, 5);
        assert_eq!(sched.running_count(), 1);
    }

    #[test]
    fn scheduler_priority_ordering() {
        let mut sched = CompactionScheduler::new(2);
        for p in [1, 5, 3, 10, 2] {
            let task = CompactionTask {
                task_id: 0,
                strategy: StrategyType::Leveled,
                input_tables: Vec::new(),
                output_level: 1,
                estimated_output_bytes: 0,
                priority: p,
            };
            sched.submit(task);
        }
        let first = sched.next_task().unwrap();
        assert_eq!(first.priority, 10);
        let second = sched.next_task().unwrap();
        assert_eq!(second.priority, 5);
    }

    #[test]
    fn scheduler_capacity() {
        let mut sched = CompactionScheduler::new(1);
        sched.submit(CompactionTask {
            task_id: 0,
            strategy: StrategyType::SizeTiered,
            input_tables: Vec::new(),
            output_level: 0,
            estimated_output_bytes: 0,
            priority: 1,
        });
        sched.submit(CompactionTask {
            task_id: 0,
            strategy: StrategyType::SizeTiered,
            input_tables: Vec::new(),
            output_level: 0,
            estimated_output_bytes: 0,
            priority: 2,
        });
        let _t = sched.next_task().unwrap();
        assert!(!sched.has_capacity());
        assert!(sched.next_task().is_none());
    }

    #[test]
    fn scheduler_complete_task() {
        let mut sched = CompactionScheduler::new(2);
        let task = CompactionTask {
            task_id: 0,
            strategy: StrategyType::Leveled,
            input_tables: vec![
                make_table(1, 100, b"a", b"m"),
                make_table(2, 200, b"n", b"z"),
            ],
            output_level: 1,
            estimated_output_bytes: 250,
            priority: 5,
        };
        sched.submit(task);
        let t = sched.next_task().unwrap();
        sched.complete_task(&t, 250, 1);
        assert_eq!(sched.running_count(), 0);
        assert_eq!(sched.stats().total_runs, 1);
        assert_eq!(sched.stats().bytes_read, 300);
        assert_eq!(sched.stats().tables_merged, 2);
    }

    #[test]
    fn scheduler_clear_pending() {
        let mut sched = CompactionScheduler::new(4);
        for _ in 0..5 {
            sched.submit(CompactionTask {
                task_id: 0,
                strategy: StrategyType::TimeWindow,
                input_tables: Vec::new(),
                output_level: 0,
                estimated_output_bytes: 0,
                priority: 1,
            });
        }
        assert_eq!(sched.pending_count(), 5);
        sched.clear_pending();
        assert_eq!(sched.pending_count(), 0);
    }

    #[test]
    fn strategy_display() {
        assert_eq!(StrategyType::SizeTiered.to_string(), "size-tiered");
        assert_eq!(StrategyType::Leveled.to_string(), "leveled");
        assert_eq!(StrategyType::TimeWindow.to_string(), "time-window");
    }

    #[test]
    fn error_display() {
        let e = CompactionError::NothingToCompact;
        assert_eq!(e.to_string(), "nothing to compact");
        let e = CompactionError::LevelOutOfRange(99);
        assert!(e.to_string().contains("99"));
    }

    #[test]
    fn task_input_bytes_and_entries() {
        let task = CompactionTask {
            task_id: 1,
            strategy: StrategyType::SizeTiered,
            input_tables: vec![
                make_table(1, 500, b"a", b"m"),
                make_table(2, 300, b"n", b"z"),
            ],
            output_level: 1,
            estimated_output_bytes: 750,
            priority: 1,
        };
        assert_eq!(task.input_bytes(), 800);
        assert_eq!(task.input_entries(), 8);
    }
}
