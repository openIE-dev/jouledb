//! GIN (Generalized Inverted Index) -- maps extracted keys to sets of record IDs.
//!
//! Used by PostgreSQL for JSONB `@>`, array `&&`/`@>`, and full-text `@@` operators.
//! JouleDB's implementation supports all three use cases through a single structure.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// What kind of values this GIN index extracts
#[derive(Debug, Clone, PartialEq)]
pub enum GinStrategy {
    /// Extract JSON paths: {"a": {"b": 1}} -> ["a.b=1"]
    JsonbPathOps,
    /// Extract JSON keys and values: {"a": 1, "b": 2} -> ["a", "b", "1", "2"]
    JsonbOps,
    /// Extract array elements: [1, 2, 3] -> ["1", "2", "3"]
    ArrayOps,
    /// Extract text tokens (whitespace-split, lowercased, with optional stemming)
    TextSearchOps,
    /// Extract trigrams for LIKE/ILIKE optimization
    TrigramOps,
}

/// Configuration for GIN index
#[derive(Debug, Clone)]
pub struct GinConfig {
    /// Indexing strategy (text search ops vs trigram ops)
    pub strategy: GinStrategy,
    /// For TextSearchOps: apply simple stemming (strip -ing, -ed, -s suffixes)
    pub enable_stemming: bool,
    /// For TextSearchOps: stop words to skip
    pub stop_words: HashSet<String>,
    /// Whether to support fast updates (pending list) or immediate consistency
    pub fast_update: bool,
}

impl Default for GinConfig {
    fn default() -> Self {
        Self {
            strategy: GinStrategy::JsonbPathOps,
            enable_stemming: false,
            stop_words: HashSet::new(),
            fast_update: false,
        }
    }
}

/// A GIN inverted index
pub struct GinIndex {
    config: GinConfig,
    /// The inverted index: extracted_key -> set of record IDs
    postings: BTreeMap<String, BTreeSet<u64>>,
    /// Forward index: record_id -> set of keys (for deletion)
    forward: HashMap<u64, HashSet<String>>,
    /// Pending entries (fast_update mode) -- not yet merged into main postings
    pending: Vec<(u64, Vec<String>)>,
    /// Number of indexed records
    count: usize,
}

impl GinIndex {
    /// Create a new GIN index with the given configuration.
    pub fn new(config: GinConfig) -> Self {
        Self {
            config,
            postings: BTreeMap::new(),
            forward: HashMap::new(),
            pending: Vec::new(),
            count: 0,
        }
    }

    /// Insert a record's value into the index.
    /// The value is parsed and keys are extracted according to the configured strategy.
    pub fn insert(&mut self, id: u64, value: &serde_json::Value) {
        let keys = self.extract_keys(value);
        if keys.is_empty() {
            return;
        }

        if self.config.fast_update {
            self.pending.push((id, keys.clone()));
            // Still update forward index so removal works before flush
            let fwd = self.forward.entry(id).or_default();
            for k in &keys {
                fwd.insert(k.clone());
            }
        } else {
            let fwd = self.forward.entry(id).or_default();
            for k in &keys {
                self.postings.entry(k.clone()).or_default().insert(id);
                fwd.insert(k.clone());
            }
        }

        self.count += 1;
    }

    /// Remove a record from the index.
    pub fn remove(&mut self, id: u64) -> bool {
        let Some(keys) = self.forward.remove(&id) else {
            return false;
        };

        for k in &keys {
            if let Some(set) = self.postings.get_mut(k) {
                set.remove(&id);
                if set.is_empty() {
                    self.postings.remove(k);
                }
            }
        }

        // Also scrub pending list
        self.pending.retain(|(pid, _)| *pid != id);

        self.count -= 1;
        true
    }

    /// Search: find all record IDs that contain ALL of the given keys (AND).
    pub fn search_contains_all(&self, keys: &[String]) -> BTreeSet<u64> {
        self.flush_view(|postings| {
            let mut result: Option<BTreeSet<u64>> = None;
            for k in keys {
                let ids = postings.get(k).cloned().unwrap_or_default();
                result = Some(match result {
                    None => ids,
                    Some(acc) => acc.intersection(&ids).copied().collect(),
                });
            }
            result.unwrap_or_default()
        })
    }

    /// Search: find all record IDs that contain ANY of the given keys (OR).
    pub fn search_contains_any(&self, keys: &[String]) -> BTreeSet<u64> {
        self.flush_view(|postings| {
            let mut result = BTreeSet::new();
            for k in keys {
                if let Some(ids) = postings.get(k) {
                    result.extend(ids);
                }
            }
            result
        })
    }

    /// JSONB `@>` operator: does the indexed value contain the query value?
    pub fn search_jsonb_contains(&self, query: &serde_json::Value) -> BTreeSet<u64> {
        let mut keys = Vec::new();
        match self.config.strategy {
            GinStrategy::JsonbPathOps => {
                self.extract_jsonb_path_ops(query, "", &mut keys);
            }
            GinStrategy::JsonbOps => {
                self.extract_jsonb_ops(query, &mut keys);
            }
            _ => return BTreeSet::new(),
        }
        self.search_contains_all(&keys)
    }

    /// Array `&&` operator: does the indexed array overlap with the query array?
    pub fn search_array_overlap(&self, elements: &[serde_json::Value]) -> BTreeSet<u64> {
        let keys: Vec<String> = elements.iter().map(value_to_string).collect();
        self.search_contains_any(&keys)
    }

    /// Full-text search: find records matching all query terms.
    pub fn search_text(&self, query: &str) -> BTreeSet<u64> {
        let tokens = self.tokenize_text(query);
        if tokens.is_empty() {
            return BTreeSet::new();
        }
        self.search_contains_all(&tokens)
    }

    /// Trigram search: find records matching LIKE/ILIKE pattern.
    ///
    /// The pattern uses SQL LIKE syntax (`%` = any, `_` = single char).
    /// We extract trigrams from the non-wildcard portions and intersect.
    /// Pattern trigrams are extracted WITHOUT padding so they only match
    /// interior trigrams of indexed values.
    pub fn search_trigram(&self, pattern: &str) -> BTreeSet<u64> {
        let normalized = pattern.to_lowercase();
        // Strip leading/trailing % for trigram extraction
        let stripped = normalized.trim_matches('%');
        if stripped.len() < 3 {
            // Too short for trigrams: return all indexed records as candidates
            return self.forward.keys().copied().collect();
        }

        // Extract trigrams from the pattern WITHOUT padding — we only want
        // interior trigrams that must appear in any matching string.
        let chars: Vec<char> = stripped.chars().collect();
        let mut trigram_keys = Vec::new();
        let mut seen = HashSet::new();
        for window in chars.windows(3) {
            let tri: String = window.iter().collect();
            if seen.insert(tri.clone()) {
                trigram_keys.push(tri);
            }
        }

        if trigram_keys.is_empty() {
            return self.forward.keys().copied().collect();
        }

        self.search_contains_all(&trigram_keys)
    }

    /// Flush pending entries to main postings (fast_update mode).
    pub fn flush_pending(&mut self) {
        let pending = std::mem::take(&mut self.pending);
        for (id, keys) in pending {
            for k in keys {
                self.postings.entry(k).or_default().insert(id);
            }
        }
    }

    /// Return the number of indexed records.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Return true if the index contains no records.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    // ------------------------------------------------------------------
    // Key extraction
    // ------------------------------------------------------------------

    fn extract_keys(&self, value: &serde_json::Value) -> Vec<String> {
        let mut keys = Vec::new();
        match self.config.strategy {
            GinStrategy::JsonbPathOps => self.extract_jsonb_path_ops(value, "", &mut keys),
            GinStrategy::JsonbOps => self.extract_jsonb_ops(value, &mut keys),
            GinStrategy::ArrayOps => self.extract_array_ops(value, &mut keys),
            GinStrategy::TextSearchOps => self.extract_text_tokens(value, &mut keys),
            GinStrategy::TrigramOps => {
                if let Some(s) = value.as_str() {
                    self.extract_trigrams(s, &mut keys);
                } else {
                    let s = value.to_string();
                    self.extract_trigrams(&s, &mut keys);
                }
            }
        }
        keys
    }

    /// JsonbPathOps: recursively walk JSON, emit `"path=value"` for leaf values.
    fn extract_jsonb_path_ops(
        &self,
        value: &serde_json::Value,
        prefix: &str,
        keys: &mut Vec<String>,
    ) {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    let path = if prefix.is_empty() {
                        k.clone()
                    } else {
                        format!("{prefix}.{k}")
                    };
                    self.extract_jsonb_path_ops(v, &path, keys);
                }
            }
            serde_json::Value::Array(arr) => {
                for (i, v) in arr.iter().enumerate() {
                    let path = if prefix.is_empty() {
                        i.to_string()
                    } else {
                        format!("{prefix}.{i}")
                    };
                    self.extract_jsonb_path_ops(v, &path, keys);
                }
            }
            _ => {
                let val_str = value_to_string(value);
                if prefix.is_empty() {
                    keys.push(format!("={val_str}"));
                } else {
                    keys.push(format!("{prefix}={val_str}"));
                }
            }
        }
    }

    /// JsonbOps: emit all keys and leaf values as separate entries.
    fn extract_jsonb_ops(&self, value: &serde_json::Value, keys: &mut Vec<String>) {
        match value {
            serde_json::Value::Object(map) => {
                for (k, v) in map {
                    keys.push(k.clone());
                    self.extract_jsonb_ops(v, keys);
                }
            }
            serde_json::Value::Array(arr) => {
                for v in arr {
                    self.extract_jsonb_ops(v, keys);
                }
            }
            _ => {
                keys.push(value_to_string(value));
            }
        }
    }

    /// ArrayOps: emit string representation of each element.
    fn extract_array_ops(&self, value: &serde_json::Value, keys: &mut Vec<String>) {
        if let serde_json::Value::Array(arr) = value {
            for v in arr {
                keys.push(value_to_string(v));
            }
        }
    }

    /// TextSearchOps: lowercase, split on whitespace/punctuation, optional stemming.
    fn extract_text_tokens(&self, value: &serde_json::Value, keys: &mut Vec<String>) {
        let text = match value {
            serde_json::Value::String(s) => s.clone(),
            _ => value.to_string(),
        };
        let tokens = self.tokenize_text(&text);
        keys.extend(tokens);
    }

    /// Tokenize text: lowercase, split on non-alphanumeric, filter stop words, stem.
    fn tokenize_text(&self, text: &str) -> Vec<String> {
        let lower = text.to_lowercase();
        lower
            .split(|c: char| !c.is_alphanumeric())
            .filter(|w| !w.is_empty())
            .filter(|w| !self.config.stop_words.contains(*w))
            .map(|w| {
                if self.config.enable_stemming {
                    self.stem_word(w)
                } else {
                    w.to_string()
                }
            })
            .collect()
    }

    /// Extract trigrams from text (padded with spaces).
    fn extract_trigrams(&self, text: &str, keys: &mut Vec<String>) {
        let padded = format!("  {text} ");
        let lower = padded.to_lowercase();
        let chars: Vec<char> = lower.chars().collect();
        let mut seen = HashSet::new();
        for window in chars.windows(3) {
            let tri: String = window.iter().collect();
            if seen.insert(tri.clone()) {
                keys.push(tri);
            }
        }
    }

    /// Simple English stemmer: strip common suffixes.
    fn stem_word(&self, word: &str) -> String {
        if word.len() <= 3 {
            return word.to_string();
        }
        // Order matters: try longest suffixes first
        if let Some(base) = word.strip_suffix("ying") {
            return format!("{base}y");
        }
        if let Some(base) = word.strip_suffix("ies") {
            return format!("{base}y");
        }
        if let Some(base) = word.strip_suffix("ness") {
            return base.to_string();
        }
        if let Some(base) = word.strip_suffix("ment") {
            if base.len() >= 3 {
                return base.to_string();
            }
        }
        if let Some(base) = word.strip_suffix("tion") {
            if base.len() >= 2 {
                return format!("{base}te");
            }
        }
        if let Some(base) = word.strip_suffix("ing") {
            if base.len() >= 3 {
                return base.to_string();
            }
        }
        if let Some(base) = word.strip_suffix("ting") {
            if base.len() >= 2 {
                return format!("{base}te");
            }
        }
        if let Some(base) = word.strip_suffix("ed") {
            if base.len() >= 3 {
                return base.to_string();
            }
        }
        if let Some(base) = word.strip_suffix("es") {
            if base.len() >= 3 {
                return base.to_string();
            }
        }
        if let Some(base) = word.strip_suffix("s") {
            if base.len() >= 3 {
                return base.to_string();
            }
        }
        word.to_string()
    }

    // ------------------------------------------------------------------
    // Internal helpers
    // ------------------------------------------------------------------

    /// Execute a search function against a merged view of postings + pending.
    fn flush_view<F>(&self, f: F) -> BTreeSet<u64>
    where
        F: FnOnce(&BTreeMap<String, BTreeSet<u64>>) -> BTreeSet<u64>,
    {
        if self.pending.is_empty() {
            return f(&self.postings);
        }
        // Build a temporary merged view
        let mut merged = self.postings.clone();
        for (id, keys) in &self.pending {
            for k in keys {
                merged.entry(k.clone()).or_default().insert(*id);
            }
        }
        f(&merged)
    }
}

/// Convert a serde_json::Value to its string representation for indexing.
fn value_to_string(v: &serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "null".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => s.clone(),
        // For complex values, use JSON repr
        other => other.to_string(),
    }
}

// ======================================================================
// Tests
// ======================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn default_stop_words() -> HashSet<String> {
        ["the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "of"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    // ------------------------------------------------------------------
    // JSONB containment (PathOps)
    // ------------------------------------------------------------------

    #[test]
    fn jsonb_path_ops_basic_containment() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbPathOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!({"name": "alice", "age": 30}));
        idx.insert(2, &json!({"name": "bob", "age": 25}));
        idx.insert(3, &json!({"name": "alice", "age": 25}));

        // @> {"name": "alice"}
        let result = idx.search_jsonb_contains(&json!({"name": "alice"}));
        assert_eq!(result, BTreeSet::from([1, 3]));

        // @> {"age": 25}
        let result = idx.search_jsonb_contains(&json!({"age": 25}));
        assert_eq!(result, BTreeSet::from([2, 3]));

        // @> {"name": "alice", "age": 30} — must match both
        let result = idx.search_jsonb_contains(&json!({"name": "alice", "age": 30}));
        assert_eq!(result, BTreeSet::from([1]));
    }

    #[test]
    fn jsonb_path_ops_nested() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbPathOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!({"user": {"role": "admin", "active": true}}));
        idx.insert(2, &json!({"user": {"role": "viewer", "active": true}}));
        idx.insert(3, &json!({"user": {"role": "admin", "active": false}}));

        let result = idx.search_jsonb_contains(&json!({"user": {"role": "admin"}}));
        assert_eq!(result, BTreeSet::from([1, 3]));

        let result =
            idx.search_jsonb_contains(&json!({"user": {"role": "admin", "active": true}}));
        assert_eq!(result, BTreeSet::from([1]));
    }

    #[test]
    fn jsonb_path_ops_with_arrays() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbPathOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!({"tags": ["rust", "db"]}));
        idx.insert(2, &json!({"tags": ["python", "ml"]}));

        // Search for tag at index 0 = "rust"
        let keys = vec!["tags.0=rust".to_string()];
        let result = idx.search_contains_all(&keys);
        assert_eq!(result, BTreeSet::from([1]));
    }

    // ------------------------------------------------------------------
    // JSONB containment (JsonbOps)
    // ------------------------------------------------------------------

    #[test]
    fn jsonb_ops_key_value_extraction() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!({"color": "red", "size": "large"}));
        idx.insert(2, &json!({"color": "blue", "size": "small"}));
        idx.insert(3, &json!({"color": "red", "size": "small"}));

        // Search for any record with key "color" and value "red"
        let keys = vec!["red".to_string()];
        let result = idx.search_contains_all(&keys);
        assert_eq!(result, BTreeSet::from([1, 3]));

        // AND search: "red" AND "small"
        let keys = vec!["red".to_string(), "small".to_string()];
        let result = idx.search_contains_all(&keys);
        assert_eq!(result, BTreeSet::from([3]));
    }

    #[test]
    fn jsonb_ops_contains_query() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!({"a": 1, "b": 2, "c": 3}));
        idx.insert(2, &json!({"a": 1, "d": 4}));

        // @> {"a": 1} — both keys "a" and value "1" must be present
        let result = idx.search_jsonb_contains(&json!({"a": 1}));
        assert_eq!(result, BTreeSet::from([1, 2]));
    }

    // ------------------------------------------------------------------
    // Array overlap
    // ------------------------------------------------------------------

    #[test]
    fn array_overlap_basic() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!([1, 2, 3]));
        idx.insert(2, &json!([3, 4, 5]));
        idx.insert(3, &json!([5, 6, 7]));

        // && [3] — records containing 3
        let result = idx.search_array_overlap(&[json!(3)]);
        assert_eq!(result, BTreeSet::from([1, 2]));

        // && [5, 1] — records containing 5 OR 1
        let result = idx.search_array_overlap(&[json!(5), json!(1)]);
        assert_eq!(result, BTreeSet::from([1, 2, 3]));

        // && [99] — no match
        let result = idx.search_array_overlap(&[json!(99)]);
        assert!(result.is_empty());
    }

    #[test]
    fn array_contains_all() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!(["rust", "python", "go"]));
        idx.insert(2, &json!(["rust", "java"]));
        idx.insert(3, &json!(["python", "go"]));

        // @> ["rust", "python"] — must contain both
        let keys = vec!["rust".to_string(), "python".to_string()];
        let result = idx.search_contains_all(&keys);
        assert_eq!(result, BTreeSet::from([1]));
    }

    #[test]
    fn array_with_mixed_types() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!([1, "two", true, null]));

        let result = idx.search_array_overlap(&[json!("two")]);
        assert_eq!(result, BTreeSet::from([1]));

        let result = idx.search_array_overlap(&[json!(true)]);
        assert_eq!(result, BTreeSet::from([1]));
    }

    // ------------------------------------------------------------------
    // Full-text search
    // ------------------------------------------------------------------

    #[test]
    fn text_search_basic() {
        let config = GinConfig {
            strategy: GinStrategy::TextSearchOps,
            stop_words: default_stop_words(),
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!("The quick brown fox jumps over the lazy dog"));
        idx.insert(2, &json!("A fast brown car drives on the road"));
        idx.insert(3, &json!("Quick brown foxes are amazing"));

        let result = idx.search_text("brown");
        assert_eq!(result, BTreeSet::from([1, 2, 3]));

        let result = idx.search_text("quick brown");
        assert_eq!(result, BTreeSet::from([1, 3]));

        let result = idx.search_text("drives");
        assert_eq!(result, BTreeSet::from([2]));
    }

    #[test]
    fn text_search_with_stemming() {
        let config = GinConfig {
            strategy: GinStrategy::TextSearchOps,
            enable_stemming: true,
            stop_words: default_stop_words(),
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!("The foxes are running quickly"));
        idx.insert(2, &json!("A fox jumped over fences"));
        idx.insert(3, &json!("Dogs and cats played together"));

        // "foxes" stems to "fox", "running" to "runn" (simple stemmer)
        // "fox" matches "fox" from record 2 and "fox" from stemmed "foxes" in record 1
        let result = idx.search_text("fox");
        assert_eq!(result, BTreeSet::from([1, 2]));
    }

    #[test]
    fn text_search_stop_words_filtered() {
        let config = GinConfig {
            strategy: GinStrategy::TextSearchOps,
            stop_words: default_stop_words(),
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!("the cat in the hat"));

        // "the" and "in" are stop words; only "cat" and "hat" indexed
        let result = idx.search_text("cat");
        assert_eq!(result, BTreeSet::from([1]));

        // Searching for only stop words yields empty (no tokens to match)
        let result = idx.search_text("the");
        assert!(result.is_empty());
    }

    #[test]
    fn text_search_empty_query() {
        let config = GinConfig {
            strategy: GinStrategy::TextSearchOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);
        idx.insert(1, &json!("hello world"));

        let result = idx.search_text("");
        assert!(result.is_empty());
    }

    // ------------------------------------------------------------------
    // Trigram search
    // ------------------------------------------------------------------

    #[test]
    fn trigram_basic() {
        let config = GinConfig {
            strategy: GinStrategy::TrigramOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!("hello world"));
        idx.insert(2, &json!("help wanted"));
        idx.insert(3, &json!("goodbye moon"));

        // LIKE '%hell%' — trigrams of "hell": "hel", "ell"
        let result = idx.search_trigram("%hell%");
        assert!(result.contains(&1)); // "hello" contains "hel" and "ell"
        assert!(!result.contains(&2)); // "help" has "hel" but not "ell"
        assert!(!result.contains(&3)); // "goodbye" has neither

        // LIKE '%hel%' — trigram "hel" only; matches both hello and help
        let result = idx.search_trigram("%hel%");
        assert!(result.contains(&1));
        assert!(result.contains(&2));
    }

    #[test]
    fn trigram_short_pattern_returns_all() {
        let config = GinConfig {
            strategy: GinStrategy::TrigramOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!("abc"));
        idx.insert(2, &json!("def"));

        // Pattern too short for trigrams: return all as candidates
        let result = idx.search_trigram("%ab%");
        assert_eq!(result, BTreeSet::from([1, 2]));
    }

    #[test]
    fn trigram_case_insensitive() {
        let config = GinConfig {
            strategy: GinStrategy::TrigramOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!("Hello World"));
        idx.insert(2, &json!("HELLO there"));

        // Trigrams are lowercased, so "hello" matches both
        let result = idx.search_trigram("%hello%");
        assert_eq!(result, BTreeSet::from([1, 2]));
    }

    // ------------------------------------------------------------------
    // Insert + remove correctness
    // ------------------------------------------------------------------

    #[test]
    fn insert_and_remove() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!([10, 20]));
        idx.insert(2, &json!([20, 30]));
        assert_eq!(idx.len(), 2);

        let result = idx.search_array_overlap(&[json!(20)]);
        assert_eq!(result, BTreeSet::from([1, 2]));

        assert!(idx.remove(1));
        assert_eq!(idx.len(), 1);

        let result = idx.search_array_overlap(&[json!(20)]);
        assert_eq!(result, BTreeSet::from([2]));

        // 10 was only in record 1; should be gone now
        let result = idx.search_array_overlap(&[json!(10)]);
        assert!(result.is_empty());
    }

    #[test]
    fn remove_nonexistent_returns_false() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);
        assert!(!idx.remove(999));
    }

    #[test]
    fn double_remove() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);
        idx.insert(1, &json!([1]));
        assert!(idx.remove(1));
        assert!(!idx.remove(1));
    }

    // ------------------------------------------------------------------
    // Empty index behavior
    // ------------------------------------------------------------------

    #[test]
    fn empty_index_searches() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbPathOps,
            ..GinConfig::default()
        };
        let idx = GinIndex::new(config);

        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        assert!(idx.search_contains_all(&["foo".into()]).is_empty());
        assert!(idx.search_contains_any(&["foo".into()]).is_empty());
        assert!(idx.search_jsonb_contains(&json!({"a": 1})).is_empty());
        assert!(idx.search_text("hello").is_empty());
    }

    #[test]
    fn empty_keys_search() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);
        idx.insert(1, &json!([1, 2, 3]));

        // Empty key lists
        let result = idx.search_contains_all(&[]);
        assert!(result.is_empty());
        let result = idx.search_contains_any(&[]);
        assert!(result.is_empty());
    }

    // ------------------------------------------------------------------
    // Fast update mode
    // ------------------------------------------------------------------

    #[test]
    fn fast_update_pending_search() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            fast_update: true,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!([10, 20]));
        idx.insert(2, &json!([20, 30]));

        // Pending entries should still be searchable (via flush_view)
        let result = idx.search_array_overlap(&[json!(20)]);
        assert_eq!(result, BTreeSet::from([1, 2]));

        // Postings should be empty (not flushed yet)
        assert!(idx.postings.is_empty());
    }

    #[test]
    fn fast_update_flush() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            fast_update: true,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!([10, 20]));
        idx.insert(2, &json!([20, 30]));

        assert!(!idx.pending.is_empty());
        idx.flush_pending();
        assert!(idx.pending.is_empty());

        // After flush, postings should contain the data
        assert!(idx.postings.contains_key("20"));

        let result = idx.search_array_overlap(&[json!(20)]);
        assert_eq!(result, BTreeSet::from([1, 2]));
    }

    #[test]
    fn fast_update_remove_before_flush() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            fast_update: true,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!([10]));
        idx.insert(2, &json!([20]));
        assert!(idx.remove(1));

        // Pending should no longer contain record 1
        let pending_ids: Vec<u64> = idx.pending.iter().map(|(id, _)| *id).collect();
        assert!(!pending_ids.contains(&1));

        idx.flush_pending();
        let result = idx.search_array_overlap(&[json!(10)]);
        assert!(result.is_empty());
    }

    // ------------------------------------------------------------------
    // Stemming unit tests
    // ------------------------------------------------------------------

    #[test]
    fn stemmer_suffixes() {
        let config = GinConfig::default();
        let idx = GinIndex::new(config);

        assert_eq!(idx.stem_word("foxes"), "fox");
        assert_eq!(idx.stem_word("running"), "runn");
        assert_eq!(idx.stem_word("jumped"), "jump");
        assert_eq!(idx.stem_word("cats"), "cat");
        assert_eq!(idx.stem_word("happiness"), "happi");
        // Short words unchanged
        assert_eq!(idx.stem_word("the"), "the");
        assert_eq!(idx.stem_word("is"), "is");
    }

    // ------------------------------------------------------------------
    // Edge cases
    // ------------------------------------------------------------------

    #[test]
    fn jsonb_null_values() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbPathOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!({"key": null}));
        let result = idx.search_jsonb_contains(&json!({"key": null}));
        assert_eq!(result, BTreeSet::from([1]));
    }

    #[test]
    fn jsonb_boolean_values() {
        let config = GinConfig {
            strategy: GinStrategy::JsonbPathOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        idx.insert(1, &json!({"active": true}));
        idx.insert(2, &json!({"active": false}));

        let result = idx.search_jsonb_contains(&json!({"active": true}));
        assert_eq!(result, BTreeSet::from([1]));
    }

    #[test]
    fn large_document_count() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let mut idx = GinIndex::new(config);

        for i in 0..1000u64 {
            idx.insert(i, &json!([i, i + 1000]));
        }
        assert_eq!(idx.len(), 1000);

        // Record 500 has elements [500, 1500]
        let result = idx.search_array_overlap(&[json!(500)]);
        assert!(result.contains(&500));
        // Record 499 does not have 500
        assert!(!result.contains(&499));
    }

    #[test]
    fn contains_all_empty_returns_none() {
        let config = GinConfig {
            strategy: GinStrategy::ArrayOps,
            ..GinConfig::default()
        };
        let idx = GinIndex::new(config);
        let result = idx.search_contains_all(&[]);
        assert!(result.is_empty());
    }
}
