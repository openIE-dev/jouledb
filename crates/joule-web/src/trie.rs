//! Trie (prefix tree) — efficient string storage and retrieval.
//!
//! Supports insert, search, delete, prefix search (autocomplete), longest common
//! prefix, wildcard matching, word count, and serialization. Includes a compressed
//! (Patricia) trie variant.

use std::collections::HashMap;

// ── TrieNode ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TrieNode {
    children: HashMap<char, TrieNode>,
    is_end: bool,
}

impl TrieNode {
    fn new() -> Self {
        Self {
            children: HashMap::new(),
            is_end: false,
        }
    }
}

// ── Trie ────────────────────────────────────────────────────────────────────

/// A standard trie (prefix tree) for string keys.
#[derive(Debug, Clone)]
pub struct Trie {
    root: TrieNode,
    word_count: usize,
}

impl Trie {
    pub fn new() -> Self {
        Self {
            root: TrieNode::new(),
            word_count: 0,
        }
    }

    /// Number of words stored.
    pub fn word_count(&self) -> usize {
        self.word_count
    }

    pub fn is_empty(&self) -> bool {
        self.word_count == 0
    }

    /// Insert a word into the trie.
    pub fn insert(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        if !node.is_end {
            node.is_end = true;
            self.word_count += 1;
        }
    }

    /// Check if a word is in the trie.
    pub fn search(&self, word: &str) -> bool {
        self.find_node(word).is_some_and(|n| n.is_end)
    }

    /// Check if any word starts with the given prefix.
    pub fn starts_with(&self, prefix: &str) -> bool {
        self.find_node(prefix).is_some()
    }

    fn find_node(&self, prefix: &str) -> Option<&TrieNode> {
        let mut node = &self.root;
        for ch in prefix.chars() {
            node = node.children.get(&ch)?;
        }
        Some(node)
    }

    /// Delete a word from the trie. Returns true if the word existed.
    pub fn delete(&mut self, word: &str) -> bool {
        if Self::delete_recursive(&mut self.root, &word.chars().collect::<Vec<_>>(), 0) {
            self.word_count -= 1;
            true
        } else {
            false
        }
    }

    fn delete_recursive(node: &mut TrieNode, chars: &[char], depth: usize) -> bool {
        if depth == chars.len() {
            if !node.is_end {
                return false;
            }
            node.is_end = false;
            return true;
        }
        let ch = chars[depth];
        if let Some(child) = node.children.get_mut(&ch) {
            if Self::delete_recursive(child, chars, depth + 1) {
                // Remove child if it has no children and is not end of another word
                if child.children.is_empty() && !child.is_end {
                    node.children.remove(&ch);
                }
                return true;
            }
        }
        false
    }

    /// Return all words with the given prefix (autocomplete).
    pub fn prefix_search(&self, prefix: &str) -> Vec<String> {
        let mut results = Vec::new();
        if let Some(node) = self.find_node(prefix) {
            let mut current = prefix.to_string();
            Self::collect_words(node, &mut current, &mut results);
        }
        results
    }

    fn collect_words(node: &TrieNode, current: &mut String, results: &mut Vec<String>) {
        if node.is_end {
            results.push(current.clone());
        }
        let mut keys: Vec<_> = node.children.keys().copied().collect();
        keys.sort();
        for ch in keys {
            current.push(ch);
            Self::collect_words(&node.children[&ch], current, results);
            current.pop();
        }
    }

    /// Longest common prefix among all words in the trie.
    pub fn longest_common_prefix(&self) -> String {
        let mut prefix = String::new();
        let mut node = &self.root;
        loop {
            if node.is_end || node.children.len() != 1 {
                break;
            }
            let (&ch, child) = node.children.iter().next().unwrap();
            prefix.push(ch);
            node = child;
        }
        prefix
    }

    /// Wildcard matching where '.' matches any single character.
    pub fn wildcard_search(&self, pattern: &str) -> Vec<String> {
        let chars: Vec<char> = pattern.chars().collect();
        let mut results = Vec::new();
        let mut current = String::new();
        Self::wildcard_dfs(&self.root, &chars, 0, &mut current, &mut results);
        results
    }

    fn wildcard_dfs(
        node: &TrieNode,
        pattern: &[char],
        depth: usize,
        current: &mut String,
        results: &mut Vec<String>,
    ) {
        if depth == pattern.len() {
            if node.is_end {
                results.push(current.clone());
            }
            return;
        }
        let ch = pattern[depth];
        if ch == '.' {
            let mut keys: Vec<_> = node.children.keys().copied().collect();
            keys.sort();
            for k in keys {
                current.push(k);
                Self::wildcard_dfs(&node.children[&k], pattern, depth + 1, current, results);
                current.pop();
            }
        } else if let Some(child) = node.children.get(&ch) {
            current.push(ch);
            Self::wildcard_dfs(child, pattern, depth + 1, current, results);
            current.pop();
        }
    }

    /// Serialize the trie to a JSON-compatible string representation.
    pub fn serialize(&self) -> String {
        let words = self.prefix_search("");
        serde_json::to_string(&words).unwrap_or_default()
    }

    /// Deserialize from JSON array of strings.
    pub fn deserialize(data: &str) -> Option<Self> {
        let words: Vec<String> = serde_json::from_str(data).ok()?;
        let mut trie = Self::new();
        for w in &words {
            trie.insert(w);
        }
        Some(trie)
    }
}

impl Default for Trie {
    fn default() -> Self {
        Self::new()
    }
}

// ── CompressedTrie (Patricia Trie) ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct CompressedNode {
    label: String,
    children: HashMap<char, CompressedNode>,
    is_end: bool,
}

impl CompressedNode {
    fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            children: HashMap::new(),
            is_end: false,
        }
    }
}

/// Compressed (Patricia) trie — merges single-child chains into single edges.
#[derive(Debug, Clone)]
pub struct CompressedTrie {
    root: CompressedNode,
    word_count: usize,
}

impl CompressedTrie {
    pub fn new() -> Self {
        Self {
            root: CompressedNode::new(""),
            word_count: 0,
        }
    }

    pub fn word_count(&self) -> usize {
        self.word_count
    }

    /// Insert a word into the compressed trie.
    pub fn insert(&mut self, word: &str) {
        if Self::insert_at(&mut self.root, word) {
            self.word_count += 1;
        }
    }

    fn insert_at(node: &mut CompressedNode, remaining: &str) -> bool {
        if remaining.is_empty() {
            if node.is_end {
                return false;
            }
            node.is_end = true;
            return true;
        }

        let first = remaining.chars().next().unwrap();

        if let Some(child) = node.children.get_mut(&first) {
            // Find common prefix
            let common_len = child
                .label
                .chars()
                .zip(remaining.chars())
                .take_while(|(a, b)| a == b)
                .count();
            let child_label_len = child.label.chars().count();

            if common_len == child_label_len {
                // Full match on the edge — recurse into child
                let rest: String = remaining.chars().skip(common_len).collect();
                return Self::insert_at(child, &rest);
            }

            // Partial match — split the edge
            let common: String = child.label.chars().take(common_len).collect();
            let child_rest: String = child.label.chars().skip(common_len).collect();
            let new_rest: String = remaining.chars().skip(common_len).collect();

            let old_child = node.children.remove(&first).unwrap();
            let child_rest_first = child_rest.chars().next().unwrap();

            let mut split = CompressedNode::new(&common);
            let mut old_branch = CompressedNode::new(&child_rest);
            old_branch.children = old_child.children;
            old_branch.is_end = old_child.is_end;
            split.children.insert(child_rest_first, old_branch);

            if new_rest.is_empty() {
                split.is_end = true;
            } else {
                let new_first = new_rest.chars().next().unwrap();
                let mut new_node = CompressedNode::new(&new_rest);
                new_node.is_end = true;
                split.children.insert(new_first, new_node);
            }

            node.children.insert(first, split);
            return true;
        }

        // No matching child — create new edge
        let mut new_node = CompressedNode::new(remaining);
        new_node.is_end = true;
        node.children.insert(first, new_node);
        true
    }

    /// Search for a word in the compressed trie.
    pub fn search(&self, word: &str) -> bool {
        Self::search_at(&self.root, word)
    }

    fn search_at(node: &CompressedNode, remaining: &str) -> bool {
        if remaining.is_empty() {
            return node.is_end;
        }
        let first = remaining.chars().next().unwrap();
        if let Some(child) = node.children.get(&first) {
            let label_len = child.label.chars().count();
            let rem_chars: Vec<char> = remaining.chars().collect();
            if rem_chars.len() < label_len {
                return false;
            }
            let prefix: String = rem_chars.iter().take(label_len).collect();
            if prefix == child.label {
                let rest: String = rem_chars.iter().skip(label_len).collect();
                return Self::search_at(child, &rest);
            }
        }
        false
    }
}

impl Default for CompressedTrie {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_and_search() {
        let mut trie = Trie::new();
        trie.insert("hello");
        trie.insert("world");
        assert!(trie.search("hello"));
        assert!(trie.search("world"));
        assert!(!trie.search("hell"));
        assert!(!trie.search("worlds"));
    }

    #[test]
    fn test_starts_with() {
        let mut trie = Trie::new();
        trie.insert("apple");
        trie.insert("application");
        assert!(trie.starts_with("app"));
        assert!(trie.starts_with("apple"));
        assert!(!trie.starts_with("banana"));
    }

    #[test]
    fn test_delete() {
        let mut trie = Trie::new();
        trie.insert("hello");
        trie.insert("help");
        assert!(trie.delete("hello"));
        assert!(!trie.search("hello"));
        assert!(trie.search("help"));
        assert_eq!(trie.word_count(), 1);
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut trie = Trie::new();
        trie.insert("hello");
        assert!(!trie.delete("world"));
        assert_eq!(trie.word_count(), 1);
    }

    #[test]
    fn test_prefix_search() {
        let mut trie = Trie::new();
        trie.insert("car");
        trie.insert("card");
        trie.insert("care");
        trie.insert("careful");
        trie.insert("dog");
        let mut results = trie.prefix_search("car");
        results.sort();
        assert_eq!(results, vec!["car", "card", "care", "careful"]);
    }

    #[test]
    fn test_longest_common_prefix() {
        let mut trie = Trie::new();
        trie.insert("flower");
        trie.insert("flow");
        trie.insert("flight");
        assert_eq!(trie.longest_common_prefix(), "fl");
    }

    #[test]
    fn test_lcp_single_word() {
        let mut trie = Trie::new();
        trie.insert("hello");
        assert_eq!(trie.longest_common_prefix(), "hello");
    }

    #[test]
    fn test_wildcard_search() {
        let mut trie = Trie::new();
        trie.insert("bad");
        trie.insert("bed");
        trie.insert("bid");
        trie.insert("bud");
        let mut results = trie.wildcard_search("b.d");
        results.sort();
        assert_eq!(results, vec!["bad", "bed", "bid", "bud"]);
    }

    #[test]
    fn test_wildcard_no_match() {
        let mut trie = Trie::new();
        trie.insert("abc");
        let results = trie.wildcard_search("x.z");
        assert!(results.is_empty());
    }

    #[test]
    fn test_word_count() {
        let mut trie = Trie::new();
        assert_eq!(trie.word_count(), 0);
        assert!(trie.is_empty());
        trie.insert("a");
        trie.insert("b");
        trie.insert("a"); // duplicate
        assert_eq!(trie.word_count(), 2);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let mut trie = Trie::new();
        trie.insert("alpha");
        trie.insert("beta");
        trie.insert("gamma");
        let json = trie.serialize();
        let restored = Trie::deserialize(&json).unwrap();
        assert!(restored.search("alpha"));
        assert!(restored.search("beta"));
        assert!(restored.search("gamma"));
        assert_eq!(restored.word_count(), 3);
    }

    #[test]
    fn test_compressed_trie_basic() {
        let mut ct = CompressedTrie::new();
        ct.insert("test");
        ct.insert("testing");
        ct.insert("team");
        assert!(ct.search("test"));
        assert!(ct.search("testing"));
        assert!(ct.search("team"));
        assert!(!ct.search("tea"));
        assert!(!ct.search("tester"));
        assert_eq!(ct.word_count(), 3);
    }

    #[test]
    fn test_compressed_trie_sharing() {
        let mut ct = CompressedTrie::new();
        ct.insert("romane");
        ct.insert("romanus");
        ct.insert("romulus");
        assert!(ct.search("romane"));
        assert!(ct.search("romanus"));
        assert!(ct.search("romulus"));
        assert!(!ct.search("rom"));
    }
}
