//! Autocomplete / typeahead engine.
//!
//! Prefix trie with frequency, top-K suggestions, fuzzy prefix matching,
//! recent queries boost, category-scoped suggestions, and highlight matching
//! portions.

use std::collections::HashMap;

// ── TrieNode ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct AcTrieNode {
    children: HashMap<char, AcTrieNode>,
    /// If this node is the end of a word, its frequency.
    frequency: Option<u64>,
    /// Optional category for scoped suggestions.
    category: Option<String>,
}

impl AcTrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            frequency: None,
            category: None,
        }
    }
}

// ── Suggestion ──────────────────────────────────────────────────

/// A single autocomplete suggestion.
#[derive(Debug, Clone)]
pub struct AcSuggestion {
    /// The suggested text.
    pub text: String,
    /// Frequency / popularity score.
    pub frequency: u64,
    /// Category of the suggestion (if any).
    pub category: Option<String>,
    /// Edit distance from the query (0 for exact prefix matches).
    pub edit_distance: usize,
    /// The highlighted version: matching portion wrapped in tags.
    pub highlighted: String,
}

// ── HighlightConfig ─────────────────────────────────────────────

/// Configuration for highlighting the matching portion.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    pub open_tag: String,
    pub close_tag: String,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            open_tag: "<b>".to_string(),
            close_tag: "</b>".to_string(),
        }
    }
}

// ── AutocompleteEngine ──────────────────────────────────────────

/// Autocomplete engine backed by a prefix trie with frequency ranking.
#[derive(Debug, Clone)]
pub struct AutocompleteEngine {
    root: AcTrieNode,
    /// Total number of entries.
    entry_count: usize,
    /// Recently queried terms with timestamps (Unix epoch seconds).
    recent_queries: Vec<(String, u64)>,
    /// Maximum recent queries to keep.
    max_recent: usize,
    /// Highlight config.
    highlight: HighlightConfig,
}

impl AutocompleteEngine {
    /// Create a new engine.
    pub fn new() -> Self {
        Self {
            root: AcTrieNode::new(),
            entry_count: 0,
            recent_queries: Vec::new(),
            max_recent: 100,
            highlight: HighlightConfig::default(),
        }
    }

    /// Set the highlight tags.
    pub fn with_highlight(mut self, open: &str, close: &str) -> Self {
        self.highlight.open_tag = open.to_string();
        self.highlight.close_tag = close.to_string();
        self
    }

    /// Set max recent queries.
    pub fn with_max_recent(mut self, n: usize) -> Self {
        self.max_recent = n;
        self
    }

    /// Number of entries in the trie.
    pub fn entry_count(&self) -> usize {
        self.entry_count
    }

    /// Insert a term with a frequency.
    pub fn insert(&mut self, text: &str, frequency: u64) {
        self.insert_with_category(text, frequency, None);
    }

    /// Insert a term with frequency and category.
    pub fn insert_with_category(
        &mut self,
        text: &str,
        frequency: u64,
        category: Option<&str>,
    ) {
        let lower = text.to_lowercase();
        let mut node = &mut self.root;
        for ch in lower.chars() {
            node = node.children.entry(ch).or_insert_with(AcTrieNode::new);
        }
        if node.frequency.is_none() {
            self.entry_count += 1;
        }
        node.frequency = Some(frequency);
        node.category = category.map(|s| s.to_string());
    }

    /// Remove a term. Returns true if it existed.
    pub fn remove(&mut self, text: &str) -> bool {
        let lower = text.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        if Self::remove_recursive(&mut self.root, &chars, 0) {
            self.entry_count -= 1;
            true
        } else {
            false
        }
    }

    fn remove_recursive(node: &mut AcTrieNode, chars: &[char], depth: usize) -> bool {
        if depth == chars.len() {
            if node.frequency.is_some() {
                node.frequency = None;
                node.category = None;
                return true;
            }
            return false;
        }
        let ch = chars[depth];
        let child_exists = node.children.contains_key(&ch);
        if !child_exists {
            return false;
        }
        let removed = {
            let child = node.children.get_mut(&ch).unwrap();
            Self::remove_recursive(child, chars, depth + 1)
        };
        if removed {
            // Clean up leaf nodes.
            let child = node.children.get(&ch).unwrap();
            if child.children.is_empty() && child.frequency.is_none() {
                node.children.remove(&ch);
            }
        }
        removed
    }

    /// Update the frequency of a term.
    pub fn update_frequency(&mut self, text: &str, frequency: u64) -> bool {
        let lower = text.to_lowercase();
        let mut node = &mut self.root;
        for ch in lower.chars() {
            match node.children.get_mut(&ch) {
                Some(child) => node = child,
                None => return false,
            }
        }
        if node.frequency.is_some() {
            node.frequency = Some(frequency);
            true
        } else {
            false
        }
    }

    /// Record a recent query (for boosting).
    pub fn record_query(&mut self, query: &str, timestamp: u64) {
        self.recent_queries.push((query.to_lowercase(), timestamp));
        if self.recent_queries.len() > self.max_recent {
            self.recent_queries.remove(0);
        }
    }

    /// Navigate to the node at the end of a prefix.
    fn find_prefix_node(&self, prefix: &str) -> Option<&AcTrieNode> {
        let lower = prefix.to_lowercase();
        let mut node = &self.root;
        for ch in lower.chars() {
            node = node.children.get(&ch)?;
        }
        Some(node)
    }

    /// Collect all words under a node.
    fn collect_words(
        node: &AcTrieNode,
        prefix: &str,
        results: &mut Vec<(String, u64, Option<String>)>,
    ) {
        if let Some(freq) = node.frequency {
            results.push((prefix.to_string(), freq, node.category.clone()));
        }
        // Sort children by key for deterministic order.
        let mut keys: Vec<char> = node.children.keys().copied().collect();
        keys.sort();
        for ch in keys {
            let child = &node.children[&ch];
            let mut next = prefix.to_string();
            next.push(ch);
            Self::collect_words(child, &next, results);
        }
    }

    /// Highlight the matching prefix in a suggestion.
    fn highlight_match(&self, text: &str, prefix_len: usize) -> String {
        let chars: Vec<char> = text.chars().collect();
        let match_part: String = chars[..prefix_len.min(chars.len())].iter().collect();
        let rest: String = chars[prefix_len.min(chars.len())..].iter().collect();
        format!(
            "{}{}{}{}",
            self.highlight.open_tag, match_part, self.highlight.close_tag, rest
        )
    }

    /// Get top-K suggestions for a prefix.
    pub fn suggest(&self, prefix: &str, k: usize) -> Vec<AcSuggestion> {
        let lower_prefix = prefix.to_lowercase();
        let prefix_len = lower_prefix.chars().count();

        let node = match self.find_prefix_node(&lower_prefix) {
            Some(n) => n,
            None => return Vec::new(),
        };

        let mut words = Vec::new();
        Self::collect_words(node, &lower_prefix, &mut words);

        // Boost recent queries.
        let recent_set: HashMap<String, usize> = self
            .recent_queries
            .iter()
            .enumerate()
            .map(|(i, (q, _))| (q.clone(), i))
            .collect();

        words.sort_by(|a, b| {
            let a_recent = recent_set.get(&a.0).copied().unwrap_or(0);
            let b_recent = recent_set.get(&b.0).copied().unwrap_or(0);
            // Higher frequency first, then more recent first.
            b.1.cmp(&a.1).then_with(|| b_recent.cmp(&a_recent))
        });

        words
            .into_iter()
            .take(k)
            .map(|(text, freq, cat)| {
                let highlighted = self.highlight_match(&text, prefix_len);
                AcSuggestion {
                    text,
                    frequency: freq,
                    category: cat,
                    edit_distance: 0,
                    highlighted,
                }
            })
            .collect()
    }

    /// Suggest with category filter.
    pub fn suggest_in_category(
        &self,
        prefix: &str,
        category: &str,
        k: usize,
    ) -> Vec<AcSuggestion> {
        let all = self.suggest(prefix, usize::MAX);
        all.into_iter()
            .filter(|s| s.category.as_deref() == Some(category))
            .take(k)
            .collect()
    }

    /// Fuzzy prefix matching: find suggestions within an edit distance.
    pub fn fuzzy_suggest(&self, prefix: &str, k: usize, max_distance: usize) -> Vec<AcSuggestion> {
        let lower_prefix = prefix.to_lowercase();
        let prefix_len = lower_prefix.chars().count();

        // Collect all words in the trie.
        let mut all_words = Vec::new();
        Self::collect_words(&self.root, "", &mut all_words);

        let mut candidates: Vec<AcSuggestion> = Vec::new();

        for (word, freq, cat) in all_words {
            // Check prefix edit distance.
            let word_prefix: String = word.chars().take(prefix_len).collect();
            let ed = Self::edit_distance(&lower_prefix, &word_prefix);
            if ed <= max_distance {
                let highlighted = self.highlight_match(&word, prefix_len);
                candidates.push(AcSuggestion {
                    text: word,
                    frequency: freq,
                    category: cat,
                    edit_distance: ed,
                    highlighted,
                });
            }
        }

        // Sort: exact matches first, then by edit distance, then frequency.
        candidates.sort_by(|a, b| {
            a.edit_distance
                .cmp(&b.edit_distance)
                .then_with(|| b.frequency.cmp(&a.frequency))
        });
        candidates.truncate(k);
        candidates
    }

    /// Simple Levenshtein edit distance.
    fn edit_distance(a: &str, b: &str) -> usize {
        let a_chars: Vec<char> = a.chars().collect();
        let b_chars: Vec<char> = b.chars().collect();
        let n = a_chars.len();
        let m = b_chars.len();

        if n == 0 {
            return m;
        }
        if m == 0 {
            return n;
        }

        let mut prev: Vec<usize> = (0..=m).collect();
        let mut curr = vec![0usize; m + 1];

        for i in 1..=n {
            curr[0] = i;
            for j in 1..=m {
                let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
                curr[j] = (prev[j] + 1)
                    .min(curr[j - 1] + 1)
                    .min(prev[j - 1] + cost);
            }
            std::mem::swap(&mut prev, &mut curr);
        }
        prev[m]
    }

    /// Check if a term exists.
    pub fn contains(&self, text: &str) -> bool {
        let lower = text.to_lowercase();
        match self.find_prefix_node(&lower) {
            Some(node) => node.frequency.is_some(),
            None => false,
        }
    }

    /// Get the frequency of a term.
    pub fn get_frequency(&self, text: &str) -> Option<u64> {
        let lower = text.to_lowercase();
        self.find_prefix_node(&lower)
            .and_then(|node| node.frequency)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.root = AcTrieNode::new();
        self.entry_count = 0;
        self.recent_queries.clear();
    }
}

impl Default for AutocompleteEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_engine() -> AutocompleteEngine {
        let mut e = AutocompleteEngine::new();
        e.insert("hello", 100);
        e.insert("help", 80);
        e.insert("helicopter", 50);
        e.insert("hero", 60);
        e.insert("world", 90);
        e.insert("wonder", 40);
        e.insert("work", 70);
        e.insert("worker", 30);
        e
    }

    #[test]
    fn test_insert_and_count() {
        let e = build_engine();
        assert_eq!(e.entry_count(), 8);
    }

    #[test]
    fn test_contains() {
        let e = build_engine();
        assert!(e.contains("hello"));
        assert!(e.contains("Hello")); // case insensitive
        assert!(!e.contains("helloworld"));
    }

    #[test]
    fn test_get_frequency() {
        let e = build_engine();
        assert_eq!(e.get_frequency("hello"), Some(100));
        assert_eq!(e.get_frequency("nonexistent"), None);
    }

    #[test]
    fn test_suggest_prefix() {
        let e = build_engine();
        let sugs = e.suggest("hel", 10);
        assert!(!sugs.is_empty());
        // All suggestions should start with "hel"
        for s in &sugs {
            assert!(s.text.starts_with("hel"));
        }
    }

    #[test]
    fn test_suggest_sorted_by_frequency() {
        let e = build_engine();
        let sugs = e.suggest("hel", 10);
        // Should be sorted by frequency descending
        for window in sugs.windows(2) {
            assert!(window[0].frequency >= window[1].frequency);
        }
    }

    #[test]
    fn test_suggest_top_k() {
        let e = build_engine();
        let sugs = e.suggest("hel", 2);
        assert!(sugs.len() <= 2);
    }

    #[test]
    fn test_suggest_no_match() {
        let e = build_engine();
        let sugs = e.suggest("xyz", 10);
        assert!(sugs.is_empty());
    }

    #[test]
    fn test_remove() {
        let mut e = build_engine();
        assert!(e.remove("hello"));
        assert!(!e.contains("hello"));
        assert_eq!(e.entry_count(), 7);
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut e = build_engine();
        assert!(!e.remove("nonexistent"));
    }

    #[test]
    fn test_update_frequency() {
        let mut e = build_engine();
        assert!(e.update_frequency("hello", 200));
        assert_eq!(e.get_frequency("hello"), Some(200));
    }

    #[test]
    fn test_update_nonexistent() {
        let mut e = build_engine();
        assert!(!e.update_frequency("nonexistent", 50));
    }

    #[test]
    fn test_highlight_default() {
        let e = build_engine();
        let sugs = e.suggest("hel", 1);
        assert!(!sugs.is_empty());
        // Should contain <b>hel</b>
        assert!(sugs[0].highlighted.contains("<b>hel</b>"));
    }

    #[test]
    fn test_highlight_custom_tags() {
        let e = AutocompleteEngine::new()
            .with_highlight("<em>", "</em>");
        let mut eng = e;
        eng.insert("hello", 10);
        let sugs = eng.suggest("hel", 1);
        assert!(sugs[0].highlighted.contains("<em>hel</em>"));
    }

    #[test]
    fn test_category_suggestions() {
        let mut e = AutocompleteEngine::new();
        e.insert_with_category("apple", 100, Some("fruit"));
        e.insert_with_category("apricot", 80, Some("fruit"));
        e.insert_with_category("application", 90, Some("tech"));

        let fruit = e.suggest_in_category("ap", "fruit", 10);
        assert_eq!(fruit.len(), 2);
        for s in &fruit {
            assert_eq!(s.category.as_deref(), Some("fruit"));
        }

        let tech = e.suggest_in_category("ap", "tech", 10);
        assert_eq!(tech.len(), 1);
        assert_eq!(tech[0].text, "application");
    }

    #[test]
    fn test_fuzzy_suggest() {
        let e = build_engine();
        // "helo" is 1 edit from "helo" prefix of "hello"
        let sugs = e.fuzzy_suggest("helo", 10, 1);
        // Should find "hello" within edit distance 1
        assert!(sugs.iter().any(|s| s.text == "hello"));
    }

    #[test]
    fn test_recent_queries_boost() {
        let mut e = build_engine();
        e.record_query("help", 1000);
        let sugs = e.suggest("hel", 10);
        // "help" should be present (it's a valid prefix match)
        assert!(sugs.iter().any(|s| s.text == "help"));
    }

    #[test]
    fn test_clear() {
        let mut e = build_engine();
        e.clear();
        assert_eq!(e.entry_count(), 0);
        assert!(e.suggest("hel", 10).is_empty());
    }

    #[test]
    fn test_case_insensitive() {
        let mut e = AutocompleteEngine::new();
        e.insert("JavaScript", 100);
        assert!(e.contains("javascript"));
        let sugs = e.suggest("java", 10);
        assert_eq!(sugs.len(), 1);
    }

    #[test]
    fn test_default_trait() {
        let e = AutocompleteEngine::default();
        assert_eq!(e.entry_count(), 0);
    }

    #[test]
    fn test_edit_distance_internal() {
        assert_eq!(AutocompleteEngine::edit_distance("hello", "hello"), 0);
        assert_eq!(AutocompleteEngine::edit_distance("hello", "helo"), 1);
        assert_eq!(AutocompleteEngine::edit_distance("", "abc"), 3);
    }
}
