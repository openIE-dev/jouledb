//! Client-side fuzzy search with trigram indexing and weighted fields.
//!
//! Replaces Fuse.js / Lunr.js. Supports exact, substring, character-order,
//! and Levenshtein-based fuzzy matching with configurable thresholds.

use std::collections::HashMap;

// ── Configuration ───────────────────────────────────────────────

/// Search engine configuration.
#[derive(Debug, Clone)]
pub struct SearchConfig {
    /// 0.0 = exact only, 1.0 = match anything. Default 0.4.
    pub threshold: f64,
    pub case_sensitive: bool,
    pub max_results: usize,
    /// Field name to weight (default 1.0 if absent).
    pub field_weights: HashMap<String, f64>,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            threshold: 0.4,
            case_sensitive: false,
            max_results: 20,
            field_weights: HashMap::new(),
        }
    }
}

// ── Results ─────────────────────────────────────────────────────

/// A single search result with score and match info.
pub struct SearchResult<'a, T> {
    pub item: &'a T,
    pub score: f64,
    pub matches: Vec<MatchInfo>,
}

/// Where a match occurred within a field.
#[derive(Debug, Clone)]
pub struct MatchInfo {
    pub field: String,
    /// (start, end) byte indices of matched substrings.
    pub indices: Vec<(usize, usize)>,
}

// ── Scoring ─────────────────────────────────────────────────────

/// Compute a fuzzy match score. 0.0 = perfect, 1.0 = no match.
pub fn fuzzy_score(pattern: &str, text: &str) -> f64 {
    if pattern.is_empty() {
        return 0.0;
    }
    if text.is_empty() {
        return 1.0;
    }

    let p = pattern.to_lowercase();
    let t = text.to_lowercase();

    // Exact match.
    if p == t {
        return 0.0;
    }

    // Contains as substring.
    if t.contains(&p) {
        return 0.1;
    }

    // Character-order match (all chars present in order).
    if chars_in_order(&p, &t) {
        let gap_score = char_order_gap_score(&p, &t);
        let score = 0.2 + 0.3 * gap_score; // range ~0.2..0.5
        return score.min(0.9);
    }

    // Fall back to normalized Levenshtein.
    let nlev = normalized_levenshtein(&p, &t);
    nlev.min(1.0)
}

/// Check if all characters of `pattern` appear in `text` in order.
fn chars_in_order(pattern: &str, text: &str) -> bool {
    let mut text_iter = text.chars();
    for pc in pattern.chars() {
        let mut found = false;
        for tc in text_iter.by_ref() {
            if tc == pc {
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

/// Score the gaps when characters match in order (0.0 = tight, 1.0 = spread out).
fn char_order_gap_score(pattern: &str, text: &str) -> f64 {
    let mut total_gap: usize = 0;
    let mut text_pos = 0;
    let text_chars: Vec<char> = text.chars().collect();

    for pc in pattern.chars() {
        while text_pos < text_chars.len() {
            if text_chars[text_pos] == pc {
                break;
            }
            total_gap += 1;
            text_pos += 1;
        }
        text_pos += 1;
    }
    if text_chars.is_empty() {
        return 1.0;
    }
    (total_gap as f64) / (text_chars.len() as f64)
}

/// Standard Levenshtein edit distance.
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] { 0 } else { 1 };
            curr[j] = (prev[j] + 1)
                .min(curr[j - 1] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }

    prev[n]
}

/// Normalized Levenshtein distance: 0.0 = identical, 1.0 = completely different.
pub fn normalized_levenshtein(a: &str, b: &str) -> f64 {
    let max_len = a.chars().count().max(b.chars().count());
    if max_len == 0 {
        return 0.0;
    }
    levenshtein_distance(a, b) as f64 / max_len as f64
}

// ── Tokenization ────────────────────────────────────────────────

/// Split text into lowercase tokens on whitespace.
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace().map(|w| w.to_lowercase()).collect()
}

/// Generate trigrams (sliding window of 3 characters).
pub fn trigrams(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() < 3 {
        if chars.is_empty() {
            return Vec::new();
        }
        return vec![chars.iter().collect()];
    }
    chars.windows(3).map(|w| w.iter().collect()).collect()
}

// ── Search Index ────────────────────────────────────────────────

/// Indexed search collection.
pub struct SearchIndex<T> {
    items: Vec<T>,
    fields: Vec<(String, Box<dyn Fn(&T) -> String>)>,
    config: SearchConfig,
    /// trigram -> [(item_idx, field_name)]
    inverted_index: HashMap<String, Vec<(usize, String)>>,
}

impl<T> SearchIndex<T> {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            items: Vec::new(),
            fields: Vec::new(),
            config,
            inverted_index: HashMap::new(),
        }
    }

    /// Register a searchable field with an accessor function.
    pub fn add_field(&mut self, name: &str, accessor: impl Fn(&T) -> String + 'static) {
        self.fields.push((name.to_string(), Box::new(accessor)));
    }

    /// Add a single item and index it.
    pub fn add(&mut self, item: T) {
        let idx = self.items.len();
        self.items.push(item);
        self.index_item(idx);
    }

    /// Add multiple items.
    pub fn add_all(&mut self, items: Vec<T>) {
        let start = self.items.len();
        self.items.extend(items);
        for idx in start..self.items.len() {
            self.index_item(idx);
        }
    }

    fn index_item(&mut self, idx: usize) {
        let item = &self.items[idx];
        for (field_name, accessor) in &self.fields {
            let text = accessor(item).to_lowercase();
            for tri in trigrams(&text) {
                self.inverted_index
                    .entry(tri)
                    .or_default()
                    .push((idx, field_name.clone()));
            }
        }
    }

    /// Rebuild the entire inverted index.
    pub fn build_index(&mut self) {
        self.inverted_index.clear();
        for idx in 0..self.items.len() {
            self.index_item(idx);
        }
    }

    /// Search for items matching the query.
    pub fn search(&self, query: &str) -> Vec<SearchResult<'_, T>> {
        if query.is_empty() {
            return Vec::new();
        }

        let q = if self.config.case_sensitive {
            query.to_string()
        } else {
            query.to_lowercase()
        };

        // Use trigrams to find candidate items.
        let query_trigrams = trigrams(&q);
        let mut candidate_counts: HashMap<usize, usize> = HashMap::new();

        if !query_trigrams.is_empty() {
            for tri in &query_trigrams {
                if let Some(entries) = self.inverted_index.get(tri) {
                    for (idx, _) in entries {
                        *candidate_counts.entry(*idx).or_insert(0) += 1;
                    }
                }
            }
        }

        // If no trigram matches (short query), scan all items.
        let candidates: Vec<usize> = if candidate_counts.is_empty() {
            (0..self.items.len()).collect()
        } else {
            candidate_counts.into_keys().collect()
        };

        let mut results = Vec::new();

        for idx in candidates {
            let item = &self.items[idx];
            let mut best_score = f64::MAX;
            let mut all_matches = Vec::new();

            for (field_name, accessor) in &self.fields {
                let text = accessor(item);
                let field_text = if self.config.case_sensitive {
                    text.clone()
                } else {
                    text.to_lowercase()
                };

                let score = fuzzy_score(&q, &field_text);
                let weight = self.config.field_weights.get(field_name).copied().unwrap_or(1.0);
                let weighted = score / weight.max(0.001);

                if weighted < best_score {
                    best_score = weighted;
                }

                // Collect match indices for substring matches.
                if score <= self.config.threshold {
                    let mut indices = Vec::new();
                    if let Some(start) = field_text.find(&q) {
                        indices.push((start, start + q.len()));
                    }
                    all_matches.push(MatchInfo {
                        field: field_name.clone(),
                        indices,
                    });
                }
            }

            if best_score <= self.config.threshold {
                results.push(SearchResult {
                    item,
                    score: best_score,
                    matches: all_matches,
                });
            }
        }

        results.sort_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(self.config.max_results);
        results
    }

    /// Remove an item by index.
    pub fn remove_at(&mut self, index: usize) -> T {
        let item = self.items.remove(index);
        self.build_index(); // re-index after removal
        item
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn clear(&mut self) {
        self.items.clear();
        self.inverted_index.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_score_zero() {
        assert!((fuzzy_score("hello", "hello") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn no_match_above_threshold_filtered() {
        let score = fuzzy_score("xyz", "abcdefg");
        assert!(score > 0.4);
    }

    #[test]
    fn fuzzy_matches_character_order() {
        let score = fuzzy_score("hlo", "hello");
        assert!(score > 0.0);
        assert!(score < 0.6);
    }

    #[test]
    fn levenshtein_kitten_sitting() {
        assert_eq!(levenshtein_distance("kitten", "sitting"), 3);
    }

    #[test]
    fn weighted_fields_rank_correctly() {
        let mut config = SearchConfig::default();
        config.field_weights.insert("title".into(), 2.0);
        config.threshold = 0.5;

        let mut index: SearchIndex<(String, String)> = SearchIndex::new(config);
        index.add_field("title", |item| item.0.clone());
        index.add_field("body", |item| item.1.clone());
        index.add(("rust guide".into(), "other stuff".into()));
        index.add(("other stuff".into(), "rust guide".into()));

        let results = index.search("rust");
        assert!(!results.is_empty());
        // Item with "rust" in title (higher weight) should rank first.
        assert!(results[0].item.0.contains("rust"));
    }

    #[test]
    fn case_insensitive() {
        let score = fuzzy_score("HELLO", "hello");
        assert!((score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn max_results_limits() {
        let mut config = SearchConfig::default();
        config.max_results = 2;
        config.threshold = 1.0;

        let mut index: SearchIndex<String> = SearchIndex::new(config);
        index.add_field("text", |s| s.clone());
        for i in 0..10 {
            index.add(format!("item {i}"));
        }

        let results = index.search("item");
        assert!(results.len() <= 2);
    }

    #[test]
    fn search_returns_sorted_by_score() {
        let mut config = SearchConfig::default();
        config.threshold = 0.8;

        let mut index: SearchIndex<String> = SearchIndex::new(config);
        index.add_field("text", |s| s.clone());
        index.add("completely different".into());
        index.add("hello world".into());
        index.add("hello".into());

        let results = index.search("hello");
        assert!(results.len() >= 2);
        // Best match first.
        for i in 1..results.len() {
            assert!(results[i - 1].score <= results[i].score + f64::EPSILON);
        }
    }

    #[test]
    fn match_indices_correct() {
        let mut config = SearchConfig::default();
        config.threshold = 0.5;

        let mut index: SearchIndex<String> = SearchIndex::new(config);
        index.add_field("text", |s| s.clone());
        index.add("hello world".into());

        let results = index.search("hello");
        assert!(!results.is_empty());
        let m = &results[0].matches;
        assert!(!m.is_empty());
        assert!(!m[0].indices.is_empty());
        assert_eq!(m[0].indices[0], (0, 5));
    }

    #[test]
    fn trigram_generation() {
        let tris = trigrams("hello");
        assert_eq!(tris, vec!["hel", "ell", "llo"]);
    }

    #[test]
    fn empty_query_returns_empty() {
        let mut index: SearchIndex<String> = SearchIndex::new(SearchConfig::default());
        index.add_field("text", |s| s.clone());
        index.add("something".into());
        assert!(index.search("").is_empty());
    }

    #[test]
    fn add_search_lifecycle() {
        let mut config = SearchConfig::default();
        config.threshold = 0.5;
        let mut index: SearchIndex<String> = SearchIndex::new(config);
        index.add_field("text", |s| s.clone());
        assert_eq!(index.len(), 0);

        index.add("alpha".into());
        index.add("beta".into());
        assert_eq!(index.len(), 2);

        let results = index.search("alpha");
        assert!(!results.is_empty());

        index.clear();
        assert_eq!(index.len(), 0);
    }

    #[test]
    fn contains_match_low_score() {
        let score = fuzzy_score("ell", "hello");
        assert!((score - 0.1).abs() < f64::EPSILON);
    }
}
