//! Playlist management — ordered items, playback modes, shuffle, queue, history.

// ── Playback Mode ───────────────────────────────────────────────

/// Playback mode for the playlist.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackMode {
    Sequential,
    RepeatOne,
    RepeatAll,
    Shuffle,
}

// ── Playlist Item ───────────────────────────────────────────────

/// A single item in a playlist.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaylistItem {
    pub id: String,
    pub title: String,
    pub duration: f64,
    pub uri: String,
}

impl PlaylistItem {
    pub fn new(id: impl Into<String>, title: impl Into<String>, duration: f64, uri: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            duration,
            uri: uri.into(),
        }
    }
}

// ── Seeded RNG (xoshiro128++) ───────────────────────────────────

/// Minimal seeded PRNG for deterministic shuffle (xorshift64).
#[derive(Debug, Clone)]
struct SeededRng {
    state: u64,
}

impl SeededRng {
    fn new(seed: u64) -> Self {
        // Avoid zero state
        Self {
            state: if seed == 0 { 0x12345678_9ABCDEF0 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate a random index in [0, n).
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ── Playlist ────────────────────────────────────────────────────

/// A playlist with ordered items, playback modes, queue, and history.
#[derive(Debug, Clone)]
pub struct Playlist {
    items: Vec<PlaylistItem>,
    pub mode: PlaybackMode,
    current_index: Option<usize>,
    /// Sequential position saved before queue detour. Some(pos) = saved.
    saved_position: Option<Option<usize>>,
    /// Shuffle order (indices into `items`).
    shuffle_order: Vec<usize>,
    shuffle_pos: usize,
    rng: SeededRng,
    /// Queue of item indices to play next (overrides normal order).
    queue: Vec<usize>,
    /// History stack of previously played indices.
    history: Vec<usize>,
}

impl Playlist {
    /// Create an empty playlist.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            mode: PlaybackMode::Sequential,
            current_index: None,
            saved_position: None,
            shuffle_order: Vec::new(),
            shuffle_pos: 0,
            rng: SeededRng::new(42),
            queue: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Create a playlist with a specific shuffle seed.
    pub fn with_seed(seed: u64) -> Self {
        Self {
            rng: SeededRng::new(seed),
            ..Self::new()
        }
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Get the items slice.
    pub fn items(&self) -> &[PlaylistItem] {
        &self.items
    }

    /// Get the current item.
    pub fn current(&self) -> Option<&PlaylistItem> {
        self.current_index.and_then(|i| self.items.get(i))
    }

    /// Current index.
    pub fn current_index(&self) -> Option<usize> {
        self.current_index
    }

    /// Add an item to the end of the playlist.
    pub fn add(&mut self, item: PlaylistItem) {
        self.items.push(item);
        if self.mode == PlaybackMode::Shuffle {
            self.rebuild_shuffle();
        }
    }

    /// Remove an item by id. Returns true if found.
    pub fn remove(&mut self, id: &str) -> bool {
        let pos = self.items.iter().position(|i| i.id == id);
        let Some(idx) = pos else { return false };

        self.items.remove(idx);

        // Adjust current_index
        match self.current_index {
            Some(ci) if ci == idx => {
                if self.items.is_empty() {
                    self.current_index = None;
                } else if ci >= self.items.len() {
                    self.current_index = Some(self.items.len() - 1);
                }
            }
            Some(ci) if ci > idx => {
                self.current_index = Some(ci - 1);
            }
            _ => {}
        }

        // Remove from queue
        self.queue.retain(|qi| *qi != idx);
        for q in &mut self.queue {
            if *q > idx {
                *q -= 1;
            }
        }

        if self.mode == PlaybackMode::Shuffle {
            self.rebuild_shuffle();
        }

        true
    }

    /// Reorder: move item at `from` to `to`.
    pub fn reorder(&mut self, from: usize, to: usize) {
        if from >= self.items.len() || to >= self.items.len() || from == to {
            return;
        }
        let item = self.items.remove(from);
        self.items.insert(to, item);

        // Adjust current_index
        if let Some(ci) = self.current_index {
            if ci == from {
                self.current_index = Some(to);
            } else if from < ci && ci <= to {
                self.current_index = Some(ci - 1);
            } else if to <= ci && ci < from {
                self.current_index = Some(ci + 1);
            }
        }
    }

    /// Add an item to play next in the queue.
    pub fn add_next(&mut self, item: PlaylistItem) {
        self.items.push(item);
        let idx = self.items.len() - 1;
        self.queue.insert(0, idx);
    }

    /// Add an item to the end of the queue.
    pub fn add_to_queue(&mut self, item: PlaylistItem) {
        self.items.push(item);
        let idx = self.items.len() - 1;
        self.queue.push(idx);
    }

    /// Select a specific item by index.
    pub fn select(&mut self, index: usize) -> bool {
        if index >= self.items.len() {
            return false;
        }
        if let Some(ci) = self.current_index {
            self.history.push(ci);
        }
        self.current_index = Some(index);
        true
    }

    /// Navigate to the next item, respecting playback mode and queue.
    pub fn next(&mut self) -> Option<&PlaylistItem> {
        if self.items.is_empty() {
            return None;
        }

        // Push current to history
        if let Some(ci) = self.current_index {
            self.history.push(ci);
        }

        // Check queue first
        if !self.queue.is_empty() {
            let idx = self.queue.remove(0);
            if idx < self.items.len() {
                // Save sequential position so we can resume after queue
                if self.saved_position.is_none() {
                    self.saved_position = Some(self.current_index);
                }
                self.current_index = Some(idx);
                return self.current();
            }
        }

        // Restore sequential position after queue detour
        if let Some(saved) = self.saved_position.take() {
            self.current_index = saved;
        }

        match self.mode {
            PlaybackMode::Sequential => {
                let next = match self.current_index {
                    Some(ci) => ci + 1,
                    None => 0,
                };
                if next >= self.items.len() {
                    self.current_index = None;
                    return None;
                }
                self.current_index = Some(next);
            }
            PlaybackMode::RepeatOne => {
                // Stay on the same item, or start at 0
                if self.current_index.is_none() {
                    self.current_index = Some(0);
                }
            }
            PlaybackMode::RepeatAll => {
                let next = match self.current_index {
                    Some(ci) => (ci + 1) % self.items.len(),
                    None => 0,
                };
                self.current_index = Some(next);
            }
            PlaybackMode::Shuffle => {
                if self.shuffle_order.is_empty() {
                    self.rebuild_shuffle();
                }
                if self.shuffle_pos >= self.shuffle_order.len() {
                    self.shuffle_pos = 0;
                    self.rebuild_shuffle();
                }
                let idx = self.shuffle_order[self.shuffle_pos];
                self.shuffle_pos += 1;
                self.current_index = Some(idx);
            }
        }

        self.current()
    }

    /// Navigate to the previous item using history.
    pub fn previous(&mut self) -> Option<&PlaylistItem> {
        if let Some(prev) = self.history.pop() {
            if prev < self.items.len() {
                self.current_index = Some(prev);
                return self.current();
            }
        }

        // No history — in sequential/repeat modes, go back one
        match self.mode {
            PlaybackMode::Sequential | PlaybackMode::RepeatAll => {
                let prev = match self.current_index {
                    Some(0) if self.mode == PlaybackMode::RepeatAll => self.items.len().saturating_sub(1),
                    Some(ci) if ci > 0 => ci - 1,
                    _ => return None,
                };
                self.current_index = Some(prev);
                self.current()
            }
            PlaybackMode::RepeatOne => self.current(),
            PlaybackMode::Shuffle => None,
        }
    }

    /// Get the history stack.
    pub fn history(&self) -> &[usize] {
        &self.history
    }

    /// Total duration of all items.
    pub fn total_duration(&self) -> f64 {
        self.items.iter().map(|i| i.duration).sum()
    }

    /// Rebuild the shuffle order using Fisher-Yates.
    fn rebuild_shuffle(&mut self) {
        let n = self.items.len();
        self.shuffle_order = (0..n).collect();
        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = self.rng.next_usize(i + 1);
            self.shuffle_order.swap(i, j);
        }
        self.shuffle_pos = 0;
    }

    /// Number of items in the queue.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Clear the queue.
    pub fn clear_queue(&mut self) {
        self.queue.clear();
    }
}

impl Default for Playlist {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_playlist() -> Playlist {
        let mut pl = Playlist::new();
        pl.add(PlaylistItem::new("1", "Song A", 180.0, "a.mp3"));
        pl.add(PlaylistItem::new("2", "Song B", 240.0, "b.mp3"));
        pl.add(PlaylistItem::new("3", "Song C", 200.0, "c.mp3"));
        pl
    }

    #[test]
    fn add_and_len() {
        let pl = sample_playlist();
        assert_eq!(pl.len(), 3);
        assert!(!pl.is_empty());
    }

    #[test]
    fn sequential_next() {
        let mut pl = sample_playlist();
        pl.mode = PlaybackMode::Sequential;
        assert_eq!(pl.next().unwrap().id, "1");
        assert_eq!(pl.next().unwrap().id, "2");
        assert_eq!(pl.next().unwrap().id, "3");
        assert!(pl.next().is_none());
    }

    #[test]
    fn repeat_all() {
        let mut pl = sample_playlist();
        pl.mode = PlaybackMode::RepeatAll;
        assert_eq!(pl.next().unwrap().id, "1");
        assert_eq!(pl.next().unwrap().id, "2");
        assert_eq!(pl.next().unwrap().id, "3");
        assert_eq!(pl.next().unwrap().id, "1"); // wraps around
    }

    #[test]
    fn repeat_one() {
        let mut pl = sample_playlist();
        pl.mode = PlaybackMode::RepeatOne;
        pl.select(1);
        assert_eq!(pl.next().unwrap().id, "2"); // stays on same
        assert_eq!(pl.next().unwrap().id, "2");
    }

    #[test]
    fn shuffle_visits_all() {
        let mut pl = Playlist::with_seed(12345);
        for i in 0..10 {
            pl.add(PlaylistItem::new(format!("{i}"), format!("Song {i}"), 100.0, format!("{i}.mp3")));
        }
        pl.mode = PlaybackMode::Shuffle;
        let mut visited = std::collections::HashSet::new();
        for _ in 0..10 {
            let item = pl.next().unwrap();
            visited.insert(item.id.clone());
        }
        assert_eq!(visited.len(), 10);
    }

    #[test]
    fn shuffle_is_deterministic() {
        let mut pl1 = Playlist::with_seed(42);
        let mut pl2 = Playlist::with_seed(42);
        for i in 0..5 {
            let item = PlaylistItem::new(format!("{i}"), format!("S{i}"), 100.0, format!("{i}.mp3"));
            pl1.add(item.clone());
            pl2.add(item);
        }
        pl1.mode = PlaybackMode::Shuffle;
        pl2.mode = PlaybackMode::Shuffle;
        for _ in 0..5 {
            assert_eq!(pl1.next().unwrap().id, pl2.next().unwrap().id);
        }
    }

    #[test]
    fn queue_overrides_order() {
        let mut pl = sample_playlist();
        pl.mode = PlaybackMode::Sequential;
        pl.add_next(PlaylistItem::new("q1", "Queued", 100.0, "q.mp3"));
        // Queue has index 3 (the newly added item)
        let item = pl.next().unwrap();
        assert_eq!(item.id, "q1"); // queue item comes first
        let item = pl.next().unwrap();
        assert_eq!(item.id, "1"); // then sequential
    }

    #[test]
    fn previous_with_history() {
        let mut pl = sample_playlist();
        pl.mode = PlaybackMode::Sequential;
        pl.next(); // "1"
        pl.next(); // "2"
        pl.next(); // "3"
        let prev = pl.previous().unwrap();
        assert_eq!(prev.id, "2");
        let prev = pl.previous().unwrap();
        assert_eq!(prev.id, "1");
    }

    #[test]
    fn remove_item() {
        let mut pl = sample_playlist();
        pl.select(1); // current = "2"
        assert!(pl.remove("2"));
        assert_eq!(pl.len(), 2);
        // current should adjust
        assert!(pl.current_index().is_some());
    }

    #[test]
    fn reorder_items() {
        let mut pl = sample_playlist();
        pl.select(0);
        pl.reorder(0, 2);
        assert_eq!(pl.items()[0].id, "2");
        assert_eq!(pl.items()[1].id, "3");
        assert_eq!(pl.items()[2].id, "1");
        assert_eq!(pl.current_index(), Some(2));
    }

    #[test]
    fn total_duration() {
        let pl = sample_playlist();
        assert!((pl.total_duration() - 620.0).abs() < 1e-9);
    }

    #[test]
    fn select_item() {
        let mut pl = sample_playlist();
        assert!(pl.select(2));
        assert_eq!(pl.current().unwrap().id, "3");
        assert!(!pl.select(10)); // out of bounds
    }

    #[test]
    fn add_to_queue_end() {
        let mut pl = sample_playlist();
        pl.mode = PlaybackMode::Sequential;
        pl.add_next(PlaylistItem::new("q1", "Q1", 100.0, "q1.mp3"));
        pl.add_to_queue(PlaylistItem::new("q2", "Q2", 100.0, "q2.mp3"));
        assert_eq!(pl.queue_len(), 2);
        assert_eq!(pl.next().unwrap().id, "q1"); // add_next comes first
        assert_eq!(pl.next().unwrap().id, "q2"); // add_to_queue second
    }

    #[test]
    fn clear_queue() {
        let mut pl = sample_playlist();
        pl.add_next(PlaylistItem::new("q1", "Q1", 100.0, "q1.mp3"));
        pl.clear_queue();
        assert_eq!(pl.queue_len(), 0);
    }

    #[test]
    fn empty_playlist_next() {
        let mut pl = Playlist::new();
        assert!(pl.next().is_none());
        assert!(pl.previous().is_none());
    }
}
