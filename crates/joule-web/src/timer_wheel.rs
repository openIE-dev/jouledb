//! Hierarchical timing wheel — multi-level wheel for efficient timer management.
//!
//! Replaces setTimeout/setInterval and timer libraries with a pure-Rust
//! hierarchical timing wheel. Supports millisecond, second, and minute
//! granularity with overflow lists for far-future timers.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

// ── Timer ID ──────────────────────────────────────────────────

/// Unique identifier for a timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TimerId(pub u64);

// ── Timer Entry ───────────────────────────────────────────────

/// A scheduled timer entry.
#[derive(Debug, Clone)]
pub struct TimerEntry {
    pub id: TimerId,
    /// Absolute tick at which the timer fires.
    pub expiry_tick: u64,
    /// Optional label for debugging.
    pub label: String,
    /// Whether this timer repeats, and at what interval in ticks.
    pub repeat_interval: Option<u64>,
}

// ── Expired Timer ─────────────────────────────────────────────

/// A timer that has expired and is ready for callback collection.
#[derive(Debug, Clone)]
pub struct ExpiredTimer {
    pub id: TimerId,
    pub scheduled_tick: u64,
    pub actual_tick: u64,
    pub label: String,
}

// ── Wheel Level ───────────────────────────────────────────────

/// A single level in the hierarchical wheel.
#[derive(Debug)]
struct WheelLevel {
    /// Number of slots in this level.
    slot_count: usize,
    /// Ticks per slot at this level.
    ticks_per_slot: u64,
    /// Current position in the wheel.
    current_slot: usize,
    /// Slots containing timer IDs.
    slots: Vec<Vec<TimerId>>,
}

impl WheelLevel {
    fn new(slot_count: usize, ticks_per_slot: u64) -> Self {
        let slots = (0..slot_count).map(|_| Vec::new()).collect();
        Self {
            slot_count,
            ticks_per_slot,
            current_slot: 0,
            slots,
        }
    }

    /// Total tick span covered by this level.
    fn span(&self) -> u64 {
        self.slot_count as u64 * self.ticks_per_slot
    }

    /// Insert a timer ID into the appropriate slot.
    /// Returns true if the timer fits in this level.
    /// `ticks_from_now` must be >= 1 (delay 0 is handled by the caller).
    fn insert(&mut self, id: TimerId, ticks_from_now: u64) -> bool {
        if ticks_from_now == 0 || ticks_from_now > self.span() {
            return false;
        }
        let slot_offset = (ticks_from_now / self.ticks_per_slot) as usize;
        let target_slot = (self.current_slot + slot_offset) % self.slot_count;
        self.slots[target_slot].push(id);
        true
    }

    /// Remove a timer ID from a specific slot.
    fn remove_from_slot(&mut self, slot: usize, id: TimerId) -> bool {
        if slot >= self.slot_count {
            return false;
        }
        let before = self.slots[slot].len();
        self.slots[slot].retain(|t| *t != id);
        self.slots[slot].len() < before
    }

    /// Advance the wheel by one slot: move pointer forward, then drain the new slot.
    fn advance(&mut self) -> Vec<TimerId> {
        self.current_slot = (self.current_slot + 1) % self.slot_count;
        std::mem::take(&mut self.slots[self.current_slot])
    }
}

// ── Wheel Statistics ──────────────────────────────────────────

/// Statistics about the timing wheel.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WheelStats {
    pub total_timers: u64,
    pub active_timers: u64,
    pub expired_timers: u64,
    pub cancelled_timers: u64,
    pub reset_timers: u64,
    pub overflow_timers: u64,
    pub current_tick: u64,
    pub total_ticks_advanced: u64,
}

// ── Hierarchical Timing Wheel ─────────────────────────────────

/// A three-level hierarchical timing wheel.
///
/// Level 0: millisecond-granularity (256 slots, 1 tick/slot)
/// Level 1: second-granularity (64 slots, 256 ticks/slot)
/// Level 2: minute-granularity (64 slots, 16384 ticks/slot)
///
/// Timers beyond the wheel span go into the overflow list.
#[derive(Debug)]
pub struct TimerWheel {
    levels: Vec<WheelLevel>,
    /// All known timer entries.
    timers: HashMap<TimerId, TimerEntry>,
    /// Overflow list for far-future timers, sorted by expiry_tick.
    overflow: BTreeMap<u64, Vec<TimerId>>,
    /// Current absolute tick.
    current_tick: u64,
    /// Next timer ID to assign.
    next_id: u64,
    /// Collected expired timers.
    expired_buffer: VecDeque<ExpiredTimer>,
    /// Statistics.
    stats: WheelStats,
}

impl TimerWheel {
    /// Create a new three-level timing wheel.
    pub fn new() -> Self {
        let levels = vec![
            WheelLevel::new(256, 1),     // L0: 256 ms
            WheelLevel::new(64, 256),    // L1: 16384 ms (~16 sec)
            WheelLevel::new(64, 16384),  // L2: 1048576 ms (~17 min)
        ];
        Self {
            levels,
            timers: HashMap::new(),
            overflow: BTreeMap::new(),
            current_tick: 0,
            next_id: 1,
            expired_buffer: VecDeque::new(),
            stats: WheelStats::default(),
        }
    }

    /// Create a timing wheel with custom level configurations.
    /// Each tuple is (slot_count, ticks_per_slot).
    pub fn with_levels(level_configs: &[(usize, u64)]) -> Self {
        let levels = level_configs
            .iter()
            .map(|&(slots, ticks)| WheelLevel::new(slots, ticks))
            .collect();
        Self {
            levels,
            timers: HashMap::new(),
            overflow: BTreeMap::new(),
            current_tick: 0,
            next_id: 1,
            expired_buffer: VecDeque::new(),
            stats: WheelStats::default(),
        }
    }

    /// Schedule a one-shot timer.
    pub fn schedule(&mut self, delay_ticks: u64, label: &str) -> TimerId {
        let id = TimerId(self.next_id);
        self.next_id += 1;
        let expiry_tick = self.current_tick + delay_ticks;

        let entry = TimerEntry {
            id,
            expiry_tick,
            label: label.to_string(),
            repeat_interval: None,
        };
        self.timers.insert(id, entry);
        self.insert_into_wheel(id, delay_ticks);
        self.stats.total_timers += 1;
        self.stats.active_timers += 1;
        id
    }

    /// Schedule a repeating timer.
    pub fn schedule_repeat(&mut self, delay_ticks: u64, interval: u64, label: &str) -> TimerId {
        let id = TimerId(self.next_id);
        self.next_id += 1;
        let expiry_tick = self.current_tick + delay_ticks;

        let entry = TimerEntry {
            id,
            expiry_tick,
            label: label.to_string(),
            repeat_interval: Some(interval),
        };
        self.timers.insert(id, entry);
        self.insert_into_wheel(id, delay_ticks);
        self.stats.total_timers += 1;
        self.stats.active_timers += 1;
        id
    }

    /// Cancel a timer by ID. Returns true if found and removed.
    pub fn cancel(&mut self, id: TimerId) -> bool {
        if self.timers.remove(&id).is_some() {
            // Remove from wheel levels or overflow.
            self.remove_from_all(id);
            self.stats.cancelled_timers += 1;
            self.stats.active_timers = self.stats.active_timers.saturating_sub(1);
            true
        } else {
            false
        }
    }

    /// Reset a timer to fire after a new delay from now.
    pub fn reset(&mut self, id: TimerId, new_delay_ticks: u64) -> bool {
        if let Some(entry) = self.timers.get_mut(&id) {
            let new_expiry = self.current_tick + new_delay_ticks;
            entry.expiry_tick = new_expiry;
            // Remove from current position and re-insert.
            self.remove_from_all(id);
            self.insert_into_wheel(id, new_delay_ticks);
            self.stats.reset_timers += 1;
            true
        } else {
            false
        }
    }

    /// Advance the wheel by one tick and collect expired timers.
    pub fn tick(&mut self) -> Vec<ExpiredTimer> {
        self.current_tick += 1;
        self.stats.total_ticks_advanced += 1;
        self.stats.current_tick = self.current_tick;

        // Advance level 0.
        let expired_ids = self.levels[0].advance();
        let mut result = Vec::new();

        for id in expired_ids {
            if let Some(entry) = self.timers.get(&id) {
                if entry.expiry_tick <= self.current_tick {
                    let expired = ExpiredTimer {
                        id,
                        scheduled_tick: entry.expiry_tick,
                        actual_tick: self.current_tick,
                        label: entry.label.clone(),
                    };
                    result.push(expired);
                } else {
                    // Timer was reset or belongs to a higher level cascade, re-insert.
                    let remaining = entry.expiry_tick.saturating_sub(self.current_tick);
                    self.insert_into_wheel(id, remaining);
                }
            }
        }

        // Handle cascading from higher levels.
        self.cascade_higher_levels();

        // Handle repeating timers: re-insert them.
        let mut to_reinsert = Vec::new();
        for expired in &result {
            if let Some(entry) = self.timers.get(&expired.id) {
                if let Some(interval) = entry.repeat_interval {
                    to_reinsert.push((expired.id, interval));
                }
            }
        }

        for (id, interval) in to_reinsert {
            if let Some(entry) = self.timers.get_mut(&id) {
                entry.expiry_tick = self.current_tick + interval;
                self.insert_into_wheel(id, interval);
            }
        }

        // Remove non-repeating expired timers from the registry.
        for expired in &result {
            if let Some(entry) = self.timers.get(&expired.id) {
                if entry.repeat_interval.is_none() {
                    self.timers.remove(&expired.id);
                    self.stats.active_timers = self.stats.active_timers.saturating_sub(1);
                }
            }
            self.stats.expired_timers += 1;
        }

        // Also try to promote overflow timers.
        self.promote_overflow();

        // Buffer expired timers.
        for e in &result {
            self.expired_buffer.push_back(e.clone());
        }

        result
    }

    /// Advance the wheel by multiple ticks, collecting all expired timers.
    pub fn advance(&mut self, ticks: u64) -> Vec<ExpiredTimer> {
        let mut all_expired = Vec::new();
        for _ in 0..ticks {
            let expired = self.tick();
            all_expired.extend(expired);
        }
        all_expired
    }

    /// Collect all buffered expired timers and clear the buffer.
    pub fn collect_expired(&mut self) -> Vec<ExpiredTimer> {
        self.expired_buffer.drain(..).collect()
    }

    /// Get the current tick.
    pub fn current_tick(&self) -> u64 {
        self.current_tick
    }

    /// Get the number of active timers.
    pub fn active_count(&self) -> usize {
        self.timers.len()
    }

    /// Get overflow count.
    pub fn overflow_count(&self) -> usize {
        self.overflow.values().map(|v| v.len()).sum()
    }

    /// Check if a timer exists.
    pub fn contains(&self, id: TimerId) -> bool {
        self.timers.contains_key(&id)
    }

    /// Get time remaining for a timer in ticks.
    pub fn remaining(&self, id: TimerId) -> Option<u64> {
        self.timers.get(&id).map(|e| {
            e.expiry_tick.saturating_sub(self.current_tick)
        })
    }

    /// Get wheel statistics.
    pub fn stats(&self) -> &WheelStats {
        &self.stats
    }

    /// Total span (in ticks) the wheel can handle without overflow.
    pub fn wheel_span(&self) -> u64 {
        self.levels.last().map_or(0, |l| {
            let mut span = 0u64;
            for level in &self.levels {
                span += level.span();
            }
            let _ = l; // use last for the total
            span
        })
    }

    /// Get a snapshot of all active timer IDs.
    pub fn active_timer_ids(&self) -> Vec<TimerId> {
        self.timers.keys().copied().collect()
    }

    // ── Internal helpers ──────────────────────────────────────

    fn insert_into_wheel(&mut self, id: TimerId, ticks_from_now: u64) {
        // delay=0 means "fire on the very next tick", so treat as delay=1.
        let effective_delay = if ticks_from_now == 0 { 1 } else { ticks_from_now };
        for level in &mut self.levels {
            if level.insert(id, effective_delay) {
                return;
            }
        }
        // Doesn't fit in any level — put in overflow.
        let expiry = self.current_tick + effective_delay;
        self.overflow.entry(expiry).or_default().push(id);
        self.stats.overflow_timers += 1;
    }

    fn remove_from_all(&mut self, id: TimerId) {
        for level in &mut self.levels {
            for slot in 0..level.slot_count {
                if level.remove_from_slot(slot, id) {
                    return;
                }
            }
        }
        // Also remove from overflow.
        let mut empty_keys = Vec::new();
        for (key, ids) in self.overflow.iter_mut() {
            ids.retain(|t| *t != id);
            if ids.is_empty() {
                empty_keys.push(*key);
            }
        }
        for key in empty_keys {
            self.overflow.remove(&key);
        }
    }

    fn cascade_higher_levels(&mut self) {
        // When level 0 wraps around, cascade from level 1, etc.
        for i in 1..self.levels.len() {
            let should_cascade = self.levels[i - 1].current_slot == 0;
            if should_cascade {
                let demoted_ids = self.levels[i].advance();
                // Re-insert demoted timers into lower levels.
                for id in demoted_ids {
                    if let Some(entry) = self.timers.get(&id) {
                        let remaining = entry.expiry_tick.saturating_sub(self.current_tick);
                        // Try to insert into level 0 or leave for next cascade.
                        if remaining == 0 {
                            // This timer should have already fired.
                            continue;
                        }
                        for level_idx in 0..i {
                            if self.levels[level_idx].insert(id, remaining) {
                                break;
                            }
                        }
                    }
                }
            } else {
                break;
            }
        }
    }

    fn promote_overflow(&mut self) {
        let max_span = self.wheel_span();
        let cutoff = self.current_tick + max_span;
        let mut to_promote = Vec::new();

        for (&tick, ids) in &self.overflow {
            if tick <= cutoff {
                for id in ids {
                    to_promote.push((*id, tick));
                }
            } else {
                break; // BTreeMap is sorted, no more within range.
            }
        }

        for (id, _tick) in &to_promote {
            if let Some(entry) = self.timers.get(id) {
                let remaining = entry.expiry_tick.saturating_sub(self.current_tick);
                for level in &mut self.levels {
                    if level.insert(*id, remaining) {
                        break;
                    }
                }
            }
        }

        // Remove promoted from overflow.
        let keys_to_remove: Vec<u64> = self.overflow.keys()
            .take_while(|&&k| k <= cutoff)
            .copied()
            .collect();
        for key in keys_to_remove {
            self.overflow.remove(&key);
        }
    }
}

impl Default for TimerWheel {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schedule_and_expire() {
        let mut wheel = TimerWheel::new();
        let id = wheel.schedule(5, "test");
        assert!(wheel.contains(id));
        assert_eq!(wheel.active_count(), 1);
        // Advance 4 ticks — not yet expired.
        let expired = wheel.advance(4);
        assert!(expired.is_empty());
        // Advance 1 more — should expire.
        let expired = wheel.advance(1);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, id);
    }

    #[test]
    fn test_cancel_timer() {
        let mut wheel = TimerWheel::new();
        let id = wheel.schedule(10, "cancel-me");
        assert!(wheel.cancel(id));
        assert!(!wheel.contains(id));
        let expired = wheel.advance(20);
        assert!(expired.is_empty());
    }

    #[test]
    fn test_cancel_nonexistent() {
        let mut wheel = TimerWheel::new();
        assert!(!wheel.cancel(TimerId(999)));
    }

    #[test]
    fn test_reset_timer() {
        let mut wheel = TimerWheel::new();
        let id = wheel.schedule(5, "reset-me");
        wheel.advance(3); // 3 ticks in
        assert!(wheel.reset(id, 10)); // Now fires at tick 13
        let expired = wheel.advance(2);
        assert!(expired.is_empty()); // Tick 5 passed, not expired
        let expired = wheel.advance(8);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, id);
    }

    #[test]
    fn test_reset_nonexistent() {
        let mut wheel = TimerWheel::new();
        assert!(!wheel.reset(TimerId(999), 10));
    }

    #[test]
    fn test_repeating_timer() {
        let mut wheel = TimerWheel::new();
        let id = wheel.schedule_repeat(3, 3, "repeat");
        // First expiry at tick 3.
        let expired = wheel.advance(3);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, id);
        // Second expiry at tick 6.
        let expired = wheel.advance(3);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, id);
        // Should still be active.
        assert!(wheel.contains(id));
    }

    #[test]
    fn test_multiple_timers_same_tick() {
        let mut wheel = TimerWheel::new();
        let id1 = wheel.schedule(5, "a");
        let id2 = wheel.schedule(5, "b");
        let expired = wheel.advance(5);
        assert_eq!(expired.len(), 2);
        let ids: Vec<_> = expired.iter().map(|e| e.id).collect();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_current_tick() {
        let mut wheel = TimerWheel::new();
        assert_eq!(wheel.current_tick(), 0);
        wheel.advance(10);
        assert_eq!(wheel.current_tick(), 10);
    }

    #[test]
    fn test_remaining_time() {
        let mut wheel = TimerWheel::new();
        let id = wheel.schedule(100, "long");
        assert_eq!(wheel.remaining(id), Some(100));
        wheel.advance(30);
        assert_eq!(wheel.remaining(id), Some(70));
    }

    #[test]
    fn test_remaining_nonexistent() {
        let wheel = TimerWheel::new();
        assert_eq!(wheel.remaining(TimerId(999)), None);
    }

    #[test]
    fn test_overflow_timer() {
        // Create a small wheel to force overflow.
        let mut wheel = TimerWheel::with_levels(&[(4, 1), (4, 4)]);
        // Total span: 4 + 16 = 20 ticks. Schedule at 100.
        let id = wheel.schedule(100, "overflow");
        assert!(wheel.contains(id));
        assert!(wheel.overflow_count() > 0);
    }

    #[test]
    fn test_stats_tracking() {
        let mut wheel = TimerWheel::new();
        wheel.schedule(5, "a");
        wheel.schedule(10, "b");
        assert_eq!(wheel.stats().total_timers, 2);
        assert_eq!(wheel.stats().active_timers, 2);
        wheel.advance(5);
        assert_eq!(wheel.stats().expired_timers, 1);
    }

    #[test]
    fn test_cancel_updates_stats() {
        let mut wheel = TimerWheel::new();
        let id = wheel.schedule(10, "x");
        wheel.cancel(id);
        assert_eq!(wheel.stats().cancelled_timers, 1);
        assert_eq!(wheel.stats().active_timers, 0);
    }

    #[test]
    fn test_collect_expired_buffer() {
        let mut wheel = TimerWheel::new();
        wheel.schedule(2, "buf");
        wheel.advance(2);
        let collected = wheel.collect_expired();
        assert_eq!(collected.len(), 1);
        // Buffer should be empty now.
        let collected2 = wheel.collect_expired();
        assert!(collected2.is_empty());
    }

    #[test]
    fn test_active_timer_ids() {
        let mut wheel = TimerWheel::new();
        let id1 = wheel.schedule(10, "a");
        let id2 = wheel.schedule(20, "b");
        let ids = wheel.active_timer_ids();
        assert!(ids.contains(&id1));
        assert!(ids.contains(&id2));
    }

    #[test]
    fn test_wheel_span() {
        let wheel = TimerWheel::new();
        // 256*1 + 64*256 + 64*16384 = 256 + 16384 + 1048576 = 1065216
        assert_eq!(wheel.wheel_span(), 1_065_216);
    }

    #[test]
    fn test_zero_delay_timer() {
        let mut wheel = TimerWheel::new();
        let _id = wheel.schedule(0, "immediate");
        // A 0-delay timer expires on the next tick.
        let expired = wheel.tick();
        // It's in slot 0 which passes on next advance.
        // The timer was placed in the current slot, so it fires immediately on tick.
        assert!(!expired.is_empty() || wheel.active_count() == 0);
    }

    #[test]
    fn test_large_batch_schedule() {
        let mut wheel = TimerWheel::new();
        let mut ids = Vec::new();
        for i in 1..=100 {
            ids.push(wheel.schedule(i, "batch"));
        }
        assert_eq!(wheel.active_count(), 100);
        // Advance through all.
        let expired = wheel.advance(100);
        assert_eq!(expired.len(), 100);
    }

    #[test]
    fn test_custom_levels() {
        let mut wheel = TimerWheel::with_levels(&[(8, 1), (8, 8)]);
        let id = wheel.schedule(3, "custom");
        let expired = wheel.advance(3);
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].id, id);
    }

    #[test]
    fn test_expired_timer_label() {
        let mut wheel = TimerWheel::new();
        wheel.schedule(1, "my-label");
        let expired = wheel.tick();
        assert_eq!(expired.len(), 1);
        assert_eq!(expired[0].label, "my-label");
    }

    #[test]
    fn test_default_construction() {
        let wheel = TimerWheel::default();
        assert_eq!(wheel.current_tick(), 0);
        assert_eq!(wheel.active_count(), 0);
    }
}
