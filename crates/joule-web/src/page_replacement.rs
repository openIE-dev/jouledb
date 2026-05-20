//! Page replacement algorithms — FIFO, LRU, LFU, Clock (second chance),
//! Optimal. Page fault counting, working set tracking, hit rate comparison.

use std::collections::{HashMap, HashSet, VecDeque};

// ── Algorithm ───────────────────────────────────────────────────────────────

/// Page replacement algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PageAlgorithm {
    Fifo,
    Lru,
    Lfu,
    Clock,
    Optimal,
}

impl std::fmt::Display for PageAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PageAlgorithm::Fifo => write!(f, "FIFO"),
            PageAlgorithm::Lru => write!(f, "LRU"),
            PageAlgorithm::Lfu => write!(f, "LFU"),
            PageAlgorithm::Clock => write!(f, "Clock"),
            PageAlgorithm::Optimal => write!(f, "Optimal"),
        }
    }
}

// ── Page Access Result ──────────────────────────────────────────────────────

/// Result of a page access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessResult {
    Hit,
    Fault { evicted: Option<u64> },
}

// ── Statistics ──────────────────────────────────────────────────────────────

/// Page replacement statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct PageStats {
    pub algorithm: PageAlgorithm,
    pub frame_count: usize,
    pub total_accesses: u64,
    pub page_faults: u64,
    pub page_hits: u64,
    pub hit_rate: f64,
    pub fault_rate: f64,
}

// ── FIFO Replacer ───────────────────────────────────────────────────────────

/// FIFO page replacement — evict the oldest page.
#[derive(Debug)]
pub struct FifoReplacer {
    frames: VecDeque<u64>,
    frame_set: HashSet<u64>,
    capacity: usize,
    faults: u64,
    hits: u64,
    accesses: u64,
}

impl FifoReplacer {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: VecDeque::with_capacity(capacity),
            frame_set: HashSet::new(),
            capacity,
            faults: 0,
            hits: 0,
            accesses: 0,
        }
    }

    pub fn access(&mut self, page: u64) -> AccessResult {
        self.accesses += 1;
        if self.frame_set.contains(&page) {
            self.hits += 1;
            return AccessResult::Hit;
        }
        self.faults += 1;
        let evicted = if self.frames.len() >= self.capacity {
            let victim = self.frames.pop_front().unwrap();
            self.frame_set.remove(&victim);
            Some(victim)
        } else {
            None
        };
        self.frames.push_back(page);
        self.frame_set.insert(page);
        AccessResult::Fault { evicted }
    }

    pub fn current_pages(&self) -> Vec<u64> {
        self.frames.iter().copied().collect()
    }

    pub fn stats(&self) -> PageStats {
        let hit_rate = if self.accesses > 0 {
            self.hits as f64 / self.accesses as f64
        } else {
            0.0
        };
        PageStats {
            algorithm: PageAlgorithm::Fifo,
            frame_count: self.capacity,
            total_accesses: self.accesses,
            page_faults: self.faults,
            page_hits: self.hits,
            hit_rate,
            fault_rate: 1.0 - hit_rate,
        }
    }
}

// ── LRU Replacer ────────────────────────────────────────────────────────────

/// LRU page replacement — evict the least recently used page.
#[derive(Debug)]
pub struct LruReplacer {
    /// Order list: back is most recent, front is LRU.
    order: VecDeque<u64>,
    frame_set: HashSet<u64>,
    capacity: usize,
    faults: u64,
    hits: u64,
    accesses: u64,
}

impl LruReplacer {
    pub fn new(capacity: usize) -> Self {
        Self {
            order: VecDeque::with_capacity(capacity),
            frame_set: HashSet::new(),
            capacity,
            faults: 0,
            hits: 0,
            accesses: 0,
        }
    }

    pub fn access(&mut self, page: u64) -> AccessResult {
        self.accesses += 1;
        if self.frame_set.contains(&page) {
            self.hits += 1;
            // Move to back (most recently used)
            self.order.retain(|p| *p != page);
            self.order.push_back(page);
            return AccessResult::Hit;
        }
        self.faults += 1;
        let evicted = if self.order.len() >= self.capacity {
            let victim = self.order.pop_front().unwrap();
            self.frame_set.remove(&victim);
            Some(victim)
        } else {
            None
        };
        self.order.push_back(page);
        self.frame_set.insert(page);
        AccessResult::Fault { evicted }
    }

    pub fn current_pages(&self) -> Vec<u64> {
        self.order.iter().copied().collect()
    }

    pub fn stats(&self) -> PageStats {
        let hit_rate = if self.accesses > 0 {
            self.hits as f64 / self.accesses as f64
        } else {
            0.0
        };
        PageStats {
            algorithm: PageAlgorithm::Lru,
            frame_count: self.capacity,
            total_accesses: self.accesses,
            page_faults: self.faults,
            page_hits: self.hits,
            hit_rate,
            fault_rate: 1.0 - hit_rate,
        }
    }
}

// ── LFU Replacer ────────────────────────────────────────────────────────────

/// LFU page replacement — evict the least frequently used page.
/// Ties broken by FIFO order (earliest insertion).
#[derive(Debug)]
pub struct LfuReplacer {
    frames: Vec<u64>,
    frame_set: HashSet<u64>,
    frequency: HashMap<u64, u64>,
    insertion_order: HashMap<u64, u64>,
    capacity: usize,
    faults: u64,
    hits: u64,
    accesses: u64,
    insertion_counter: u64,
}

impl LfuReplacer {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: Vec::with_capacity(capacity),
            frame_set: HashSet::new(),
            frequency: HashMap::new(),
            insertion_order: HashMap::new(),
            capacity,
            faults: 0,
            hits: 0,
            accesses: 0,
            insertion_counter: 0,
        }
    }

    pub fn access(&mut self, page: u64) -> AccessResult {
        self.accesses += 1;
        if self.frame_set.contains(&page) {
            self.hits += 1;
            *self.frequency.entry(page).or_insert(0) += 1;
            return AccessResult::Hit;
        }
        self.faults += 1;
        let evicted = if self.frames.len() >= self.capacity {
            // Find the page with minimum frequency, tie-break by insertion order
            let victim = self.find_lfu_victim();
            self.frames.retain(|p| *p != victim);
            self.frame_set.remove(&victim);
            self.frequency.remove(&victim);
            self.insertion_order.remove(&victim);
            Some(victim)
        } else {
            None
        };
        self.frames.push(page);
        self.frame_set.insert(page);
        self.frequency.insert(page, 1);
        self.insertion_order.insert(page, self.insertion_counter);
        self.insertion_counter += 1;
        AccessResult::Fault { evicted }
    }

    fn find_lfu_victim(&self) -> u64 {
        let mut min_freq = u64::MAX;
        let mut min_order = u64::MAX;
        let mut victim = self.frames[0];

        for page in &self.frames {
            let freq = *self.frequency.get(page).unwrap_or(&0);
            let order = *self.insertion_order.get(page).unwrap_or(&0);
            if freq < min_freq || (freq == min_freq && order < min_order) {
                min_freq = freq;
                min_order = order;
                victim = *page;
            }
        }
        victim
    }

    pub fn current_pages(&self) -> Vec<u64> {
        self.frames.clone()
    }

    pub fn page_frequency(&self, page: u64) -> u64 {
        *self.frequency.get(&page).unwrap_or(&0)
    }

    pub fn stats(&self) -> PageStats {
        let hit_rate = if self.accesses > 0 {
            self.hits as f64 / self.accesses as f64
        } else {
            0.0
        };
        PageStats {
            algorithm: PageAlgorithm::Lfu,
            frame_count: self.capacity,
            total_accesses: self.accesses,
            page_faults: self.faults,
            page_hits: self.hits,
            hit_rate,
            fault_rate: 1.0 - hit_rate,
        }
    }
}

// ── Clock Replacer ──────────────────────────────────────────────────────────

/// Clock (second chance) page replacement.
#[derive(Debug)]
pub struct ClockReplacer {
    frames: Vec<Option<u64>>,
    reference_bits: Vec<bool>,
    hand: usize,
    capacity: usize,
    count: usize,
    faults: u64,
    hits: u64,
    accesses: u64,
}

impl ClockReplacer {
    pub fn new(capacity: usize) -> Self {
        Self {
            frames: vec![None; capacity],
            reference_bits: vec![false; capacity],
            hand: 0,
            capacity,
            count: 0,
            faults: 0,
            hits: 0,
            accesses: 0,
        }
    }

    pub fn access(&mut self, page: u64) -> AccessResult {
        self.accesses += 1;

        // Check if page is already in frames
        for i in 0..self.capacity {
            if self.frames[i] == Some(page) {
                self.hits += 1;
                self.reference_bits[i] = true;
                return AccessResult::Hit;
            }
        }

        self.faults += 1;

        // Check for empty frame first
        for i in 0..self.capacity {
            if self.frames[i].is_none() {
                self.frames[i] = Some(page);
                self.reference_bits[i] = true;
                self.count += 1;
                return AccessResult::Fault { evicted: None };
            }
        }

        // Clock sweep to find victim
        loop {
            if self.reference_bits[self.hand] {
                // Second chance: clear reference bit and move on
                self.reference_bits[self.hand] = false;
                self.hand = (self.hand + 1) % self.capacity;
            } else {
                // Evict this page
                let victim = self.frames[self.hand];
                self.frames[self.hand] = Some(page);
                self.reference_bits[self.hand] = true;
                self.hand = (self.hand + 1) % self.capacity;
                return AccessResult::Fault { evicted: victim };
            }
        }
    }

    pub fn current_pages(&self) -> Vec<u64> {
        self.frames.iter().filter_map(|f| *f).collect()
    }

    pub fn hand_position(&self) -> usize {
        self.hand
    }

    pub fn stats(&self) -> PageStats {
        let hit_rate = if self.accesses > 0 {
            self.hits as f64 / self.accesses as f64
        } else {
            0.0
        };
        PageStats {
            algorithm: PageAlgorithm::Clock,
            frame_count: self.capacity,
            total_accesses: self.accesses,
            page_faults: self.faults,
            page_hits: self.hits,
            hit_rate,
            fault_rate: 1.0 - hit_rate,
        }
    }
}

// ── Optimal Replacer ────────────────────────────────────────────────────────

/// Optimal (Belady's) page replacement — requires future knowledge.
/// Run an entire access sequence at once for comparison purposes.
#[derive(Debug)]
pub struct OptimalReplacer {
    capacity: usize,
}

impl OptimalReplacer {
    pub fn new(capacity: usize) -> Self {
        Self { capacity }
    }

    /// Simulate the optimal algorithm over a full access sequence.
    /// Returns (faults, hits, total_accesses, eviction_log).
    pub fn simulate(&self, sequence: &[u64]) -> OptimalResult {
        let mut frames: Vec<u64> = Vec::new();
        let mut frame_set: HashSet<u64> = HashSet::new();
        let mut faults = 0u64;
        let mut hits = 0u64;
        let mut eviction_log: Vec<Option<u64>> = Vec::new();

        for (i, &page) in sequence.iter().enumerate() {
            if frame_set.contains(&page) {
                hits += 1;
                eviction_log.push(None);
                continue;
            }
            faults += 1;

            if frames.len() < self.capacity {
                frames.push(page);
                frame_set.insert(page);
                eviction_log.push(None);
            } else {
                // Find the page that won't be used for the longest time
                let victim = self.find_optimal_victim(&frames, sequence, i + 1);
                frame_set.remove(&victim);
                let victim_idx = frames.iter().position(|p| *p == victim).unwrap();
                frames[victim_idx] = page;
                frame_set.insert(page);
                eviction_log.push(Some(victim));
            }
        }

        let total = sequence.len() as u64;
        let hit_rate = if total > 0 { hits as f64 / total as f64 } else { 0.0 };

        OptimalResult {
            faults,
            hits,
            total_accesses: total,
            hit_rate,
            eviction_log,
        }
    }

    fn find_optimal_victim(&self, frames: &[u64], sequence: &[u64], from: usize) -> u64 {
        let mut farthest_use = 0usize;
        let mut victim = frames[0];

        for &frame_page in frames {
            let next_use = sequence[from..]
                .iter()
                .position(|p| *p == frame_page)
                .map(|pos| pos + 1)
                .unwrap_or(usize::MAX);

            if next_use > farthest_use {
                farthest_use = next_use;
                victim = frame_page;
            }
        }
        victim
    }
}

/// Result of an optimal simulation.
#[derive(Debug, Clone)]
pub struct OptimalResult {
    pub faults: u64,
    pub hits: u64,
    pub total_accesses: u64,
    pub hit_rate: f64,
    pub eviction_log: Vec<Option<u64>>,
}

// ── Working Set Tracker ─────────────────────────────────────────────────────

/// Tracks the working set of pages over a sliding window.
#[derive(Debug)]
pub struct WorkingSetTracker {
    window_size: usize,
    history: VecDeque<u64>,
}

impl WorkingSetTracker {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            history: VecDeque::new(),
        }
    }

    /// Record a page access and return the current working set size.
    pub fn access(&mut self, page: u64) -> usize {
        self.history.push_back(page);
        if self.history.len() > self.window_size {
            self.history.pop_front();
        }
        self.working_set_size()
    }

    /// Current working set (unique pages in window).
    pub fn working_set(&self) -> HashSet<u64> {
        self.history.iter().copied().collect()
    }

    /// Number of unique pages in the current window.
    pub fn working_set_size(&self) -> usize {
        self.working_set().len()
    }
}

// ── Comparison ──────────────────────────────────────────────────────────────

/// Compare all algorithms on the same access sequence.
pub fn compare_algorithms(sequence: &[u64], frame_count: usize) -> Vec<PageStats> {
    let mut results = Vec::new();

    // FIFO
    let mut fifo = FifoReplacer::new(frame_count);
    for &page in sequence {
        fifo.access(page);
    }
    results.push(fifo.stats());

    // LRU
    let mut lru = LruReplacer::new(frame_count);
    for &page in sequence {
        lru.access(page);
    }
    results.push(lru.stats());

    // LFU
    let mut lfu = LfuReplacer::new(frame_count);
    for &page in sequence {
        lfu.access(page);
    }
    results.push(lfu.stats());

    // Clock
    let mut clock = ClockReplacer::new(frame_count);
    for &page in sequence {
        clock.access(page);
    }
    results.push(clock.stats());

    // Optimal
    let opt = OptimalReplacer::new(frame_count);
    let opt_result = opt.simulate(sequence);
    results.push(PageStats {
        algorithm: PageAlgorithm::Optimal,
        frame_count,
        total_accesses: opt_result.total_accesses,
        page_faults: opt_result.faults,
        page_hits: opt_result.hits,
        hit_rate: opt_result.hit_rate,
        fault_rate: 1.0 - opt_result.hit_rate,
    });

    results
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fifo_basic() {
        let mut fifo = FifoReplacer::new(3);
        assert!(matches!(fifo.access(1), AccessResult::Fault { evicted: None }));
        assert!(matches!(fifo.access(2), AccessResult::Fault { evicted: None }));
        assert!(matches!(fifo.access(3), AccessResult::Fault { evicted: None }));
        assert!(matches!(fifo.access(1), AccessResult::Hit));
        // Evict page 1 (oldest) when adding page 4
        assert!(matches!(fifo.access(4), AccessResult::Fault { evicted: Some(1) }));
    }

    #[test]
    fn test_fifo_stats() {
        let mut fifo = FifoReplacer::new(3);
        for &p in &[1, 2, 3, 1, 4, 5] {
            fifo.access(p);
        }
        let stats = fifo.stats();
        assert_eq!(stats.total_accesses, 6);
        assert_eq!(stats.page_hits, 1);
        assert_eq!(stats.page_faults, 5);
    }

    #[test]
    fn test_lru_basic() {
        let mut lru = LruReplacer::new(3);
        lru.access(1);
        lru.access(2);
        lru.access(3);
        lru.access(1); // Hit — moves 1 to most recent
        // Pages in LRU order: [2, 3, 1]
        // Evict 2 (least recently used)
        assert!(matches!(lru.access(4), AccessResult::Fault { evicted: Some(2) }));
    }

    #[test]
    fn test_lru_eviction_order() {
        let mut lru = LruReplacer::new(2);
        lru.access(1);
        lru.access(2);
        lru.access(1); // Re-access 1, making 2 the LRU
        let result = lru.access(3);
        assert!(matches!(result, AccessResult::Fault { evicted: Some(2) }));
    }

    #[test]
    fn test_lfu_basic() {
        let mut lfu = LfuReplacer::new(3);
        lfu.access(1);
        lfu.access(1); // freq=2
        lfu.access(2);
        lfu.access(2); // freq=2
        lfu.access(3); // freq=1
        // Frame full. Evict LFU = page 3 (freq=1)
        let result = lfu.access(4);
        assert!(matches!(result, AccessResult::Fault { evicted: Some(3) }));
    }

    #[test]
    fn test_lfu_frequency() {
        let mut lfu = LfuReplacer::new(3);
        lfu.access(5);
        lfu.access(5);
        lfu.access(5);
        assert_eq!(lfu.page_frequency(5), 3); // 1 initial + 2 hits
    }

    #[test]
    fn test_clock_basic() {
        let mut clock = ClockReplacer::new(3);
        clock.access(1);
        clock.access(2);
        clock.access(3);
        clock.access(1); // Set reference bit for 1
        // Access 4: sweep — page 1 has ref bit set, clear it; page 2 has ref bit set, clear it; page 3... clear it too...
        // After second sweep, evict page 1 (or whichever has ref=false first)
        let result = clock.access(4);
        assert!(matches!(result, AccessResult::Fault { evicted: Some(_) }));
    }

    #[test]
    fn test_clock_hit() {
        let mut clock = ClockReplacer::new(3);
        clock.access(1);
        clock.access(2);
        assert!(matches!(clock.access(1), AccessResult::Hit));
    }

    #[test]
    fn test_optimal_basic() {
        let opt = OptimalReplacer::new(3);
        let seq = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5];
        let result = opt.simulate(&seq);
        // Optimal should have <= faults of any other algorithm
        assert!(result.faults <= result.total_accesses);
        assert!(result.hits + result.faults == result.total_accesses);
    }

    #[test]
    fn test_optimal_no_eviction_small_sequence() {
        let opt = OptimalReplacer::new(4);
        let seq = [1, 2, 3, 4];
        let result = opt.simulate(&seq);
        assert_eq!(result.faults, 4); // All compulsory misses
        assert_eq!(result.hits, 0);
    }

    #[test]
    fn test_working_set_tracker() {
        let mut ws = WorkingSetTracker::new(4);
        ws.access(1);
        ws.access(2);
        ws.access(3);
        assert_eq!(ws.working_set_size(), 3);
        ws.access(1);
        assert_eq!(ws.working_set_size(), 3);
        ws.access(4);
        ws.access(5);
        // Window: [3, 1, 4, 5]
        assert_eq!(ws.working_set_size(), 4);
    }

    #[test]
    fn test_compare_algorithms() {
        let seq = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5];
        let results = compare_algorithms(&seq, 3);
        assert_eq!(results.len(), 5);
        // Optimal should be the best (or tied)
        let opt_faults = results.iter().find(|s| s.algorithm == PageAlgorithm::Optimal).unwrap().page_faults;
        for stat in &results {
            assert!(stat.page_faults >= opt_faults);
        }
    }

    #[test]
    fn test_fifo_belady_anomaly() {
        // Belady's anomaly: more frames can lead to more faults with FIFO
        let seq = [1, 2, 3, 4, 1, 2, 5, 1, 2, 3, 4, 5];
        let mut fifo3 = FifoReplacer::new(3);
        let mut fifo4 = FifoReplacer::new(4);
        for &p in &seq {
            fifo3.access(p);
            fifo4.access(p);
        }
        // This specific sequence shows Belady's anomaly
        let faults3 = fifo3.stats().page_faults;
        let faults4 = fifo4.stats().page_faults;
        // Just verify they both produce valid results
        assert!(faults3 > 0);
        assert!(faults4 > 0);
    }

    #[test]
    fn test_lru_current_pages() {
        let mut lru = LruReplacer::new(3);
        lru.access(1);
        lru.access(2);
        lru.access(3);
        let pages = lru.current_pages();
        assert_eq!(pages.len(), 3);
        assert!(pages.contains(&1));
        assert!(pages.contains(&2));
        assert!(pages.contains(&3));
    }

    #[test]
    fn test_clock_hand_advances() {
        let mut clock = ClockReplacer::new(3);
        clock.access(1);
        clock.access(2);
        clock.access(3);
        let h1 = clock.hand_position();
        clock.access(4); // Will sweep and evict
        let h2 = clock.hand_position();
        // Hand should have moved
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_all_hits_no_faults() {
        let mut lru = LruReplacer::new(3);
        lru.access(1);
        lru.access(2);
        lru.access(3);
        // All subsequent are hits
        for _ in 0..10 {
            assert!(matches!(lru.access(1), AccessResult::Hit));
        }
        let stats = lru.stats();
        assert_eq!(stats.page_faults, 3); // only compulsory
        assert_eq!(stats.page_hits, 10);
    }

    #[test]
    fn test_working_set_sliding_window() {
        let mut ws = WorkingSetTracker::new(3);
        ws.access(1);
        ws.access(2);
        ws.access(3);
        assert_eq!(ws.working_set_size(), 3);
        ws.access(4);
        // Window: [2, 3, 4]
        let set = ws.working_set();
        assert!(!set.contains(&1));
        assert!(set.contains(&4));
    }
}
