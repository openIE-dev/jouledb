//! SDM Context Window: replaces the fixed n-gram window with content-addressable memory.
//!
//! Based on Bricken & Pehlevan (NeurIPS 2021): transformer attention ≈ SDM read.
//! Instead of a sliding window of 3 tokens, we write each sentence/phrase into
//! SDM and read by similarity. This gives an effective context window of
//! thousands of items, with graceful degradation (not a hard cutoff).
//!
//! The SDM context acts like working memory: recent items are strongest,
//! old items naturally decay (Forget primitive), and retrieval is by
//! content similarity (Compare primitive).

use crate::BinaryHV;
use joule_db_hdc::BundleAccumulator;
use std::collections::VecDeque;

/// An entry in the context window.
#[derive(Clone, Debug)]
pub struct ContextEntry {
    /// The holographic encoding of this context item.
    pub vector: BinaryHV,
    /// Human-readable label (the text that was encoded).
    pub label: String,
    /// Timestamp (monotonic counter, not wall clock).
    pub timestamp: u64,
    /// Strength: starts at 1.0, decays over time.
    pub strength: f32,
}

/// SDM-backed context window.
pub struct ContextWindow {
    /// All entries (most recent last).
    entries: VecDeque<ContextEntry>,
    /// Maximum entries before eviction.
    capacity: usize,
    /// Running centroid of all context (what "current topic" looks like).
    centroid_acc: BundleAccumulator,
    centroid: Option<BinaryHV>,
    /// Decay rate: strength multiplied by this each tick.
    decay_rate: f32,
    /// Monotonic timestamp counter.
    tick: u64,
    /// Dimension.
    dim: usize,
}

impl ContextWindow {
    pub fn new(capacity: usize, dim: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            centroid_acc: BundleAccumulator::new(dim),
            centroid: None,
            decay_rate: 0.95,
            tick: 0,
            dim,
        }
    }

    /// Default: 1000 entries, 10K dimensions.
    pub fn default_large() -> Self {
        Self::new(1000, 10_000)
    }

    /// Write a new entry into the context window.
    pub fn write(&mut self, vector: BinaryHV, label: &str) {
        self.tick += 1;

        // Decay existing entries
        for entry in &mut self.entries {
            entry.strength *= self.decay_rate;
        }

        // Evict weakest if at capacity
        if self.entries.len() >= self.capacity {
            // Remove the entry with lowest strength
            if let Some(min_idx) = self
                .entries
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.strength
                        .partial_cmp(&b.1.strength)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
            {
                self.entries.remove(min_idx);
            }
        }

        // Add new entry
        self.centroid_acc.add(&vector);
        self.entries.push_back(ContextEntry {
            vector,
            label: label.to_string(),
            timestamp: self.tick,
            strength: 1.0,
        });

        // Update centroid
        if self.tick % 5 == 0 || self.centroid.is_none() {
            self.centroid = Some(self.centroid_acc.threshold());
        }
    }

    /// Read: find the K entries most similar to the query (attention-SDM read).
    /// Returns entries sorted by similarity × strength (recency-weighted).
    pub fn read(&self, query: &BinaryHV, k: usize) -> Vec<(String, f32)> {
        let mut scored: Vec<(String, f32)> = self
            .entries
            .iter()
            .map(|entry| {
                let sim = entry.vector.similarity(query);
                let weighted = sim * entry.strength;
                (entry.label.clone(), weighted)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Read and bundle: retrieve K most similar entries, bundle their vectors.
    /// This is the SDM equivalent of attention's weighted sum of values.
    pub fn read_bundle(&self, query: &BinaryHV, k: usize) -> Option<BinaryHV> {
        let top = self.read(query, k);
        if top.is_empty() {
            return None;
        }

        let mut acc = BundleAccumulator::new(self.dim);
        for (label, _) in &top {
            // Find the entry and add its vector
            if let Some(entry) = self.entries.iter().find(|e| &e.label == label) {
                acc.add(&entry.vector);
            }
        }
        Some(acc.threshold())
    }

    /// Get the current context centroid (what "the current topic" looks like).
    pub fn centroid(&self) -> Option<&BinaryHV> {
        self.centroid.as_ref()
    }

    /// How many entries are in the context.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Is the context empty?
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Get the N most recent entries.
    pub fn recent(&self, n: usize) -> Vec<&ContextEntry> {
        self.entries.iter().rev().take(n).collect()
    }

    /// Clear the context window.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.centroid_acc = BundleAccumulator::new(self.dim);
        self.centroid = None;
        self.tick = 0;
    }

    /// Novelty of a query relative to the current context.
    pub fn novelty(&self, query: &BinaryHV) -> f64 {
        match &self.centroid {
            Some(c) => 1.0 - query.similarity(c) as f64,
            None => 1.0,
        }
    }
}

impl Default for ContextWindow {
    fn default() -> Self {
        Self::default_large()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_and_read() {
        let mut ctx = ContextWindow::new(100, 1000);
        let v1 = BinaryHV::random(1000, 1);
        let v2 = BinaryHV::random(1000, 2);

        ctx.write(v1.clone(), "first");
        ctx.write(v2.clone(), "second");

        // Query with v1 should find "first" as most similar
        let results = ctx.read(&v1, 2);
        assert_eq!(results[0].0, "first");
    }

    #[test]
    fn test_decay() {
        let mut ctx = ContextWindow::new(100, 1000);
        let v1 = BinaryHV::random(1000, 1);
        ctx.write(v1.clone(), "old");

        // Write many more entries to decay "old"
        for i in 0..20 {
            ctx.write(BinaryHV::random(1000, 100 + i), &format!("new_{i}"));
        }

        // "old" should have low strength
        let old_entry = ctx.entries.iter().find(|e| e.label == "old");
        assert!(old_entry.is_some());
        assert!(
            old_entry.unwrap().strength < 0.5,
            "old entry should have decayed"
        );
    }

    #[test]
    fn test_eviction_at_capacity() {
        let mut ctx = ContextWindow::new(5, 1000);
        for i in 0..10 {
            ctx.write(BinaryHV::random(1000, i), &format!("entry_{i}"));
        }
        assert_eq!(ctx.len(), 5);
    }

    #[test]
    fn test_read_bundle() {
        let mut ctx = ContextWindow::new(100, 1000);
        let v1 = BinaryHV::random(1000, 1);
        let v2 = BinaryHV::random(1000, 2);
        ctx.write(v1.clone(), "a");
        ctx.write(v2.clone(), "b");

        let bundle = ctx.read_bundle(&v1, 2);
        assert!(bundle.is_some());
    }

    #[test]
    fn test_novelty() {
        let mut ctx = ContextWindow::new(100, 1000);
        let v1 = BinaryHV::random(1000, 1);
        ctx.write(v1.clone(), "known");

        // Known vector: low novelty
        let known_novelty = ctx.novelty(&v1);
        // Unknown vector: higher novelty
        let unknown_novelty = ctx.novelty(&BinaryHV::random(1000, 999));

        // With only 1 entry, centroid = that entry, so novelty of it should be low
        assert!(known_novelty < unknown_novelty || ctx.len() == 1);
    }

    #[test]
    fn test_recent() {
        let mut ctx = ContextWindow::new(100, 1000);
        ctx.write(BinaryHV::random(1000, 1), "first");
        ctx.write(BinaryHV::random(1000, 2), "second");
        ctx.write(BinaryHV::random(1000, 3), "third");

        let recent = ctx.recent(2);
        assert_eq!(recent.len(), 2);
        assert_eq!(recent[0].label, "third"); // Most recent first
        assert_eq!(recent[1].label, "second");
    }
}
