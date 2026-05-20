//! Inverted index for full-text search.
//!
//! Tokenizer (whitespace, punctuation strip, lowercase), posting lists with
//! positions, TF-IDF scoring, boolean queries (AND/OR/NOT), phrase queries,
//! document add/remove, index serialization.

use std::collections::{BTreeMap, HashMap, HashSet};

// ── Errors ──────────────────────────────────────────────────────

/// Errors arising from index operations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum IndexError {
    #[error("document {0} not found")]
    DocumentNotFound(String),
    #[error("document {0} already exists")]
    DuplicateDocument(String),
    #[error("empty query")]
    EmptyQuery,
    #[error("serialization error: {0}")]
    SerializationError(String),
}

// ── Tokenizer ───────────────────────────────────────────────────

/// Tokenize text: split on whitespace, strip punctuation, lowercase.
/// Returns (token, position) pairs.
pub fn tokenize(text: &str) -> Vec<(String, usize)> {
    let mut tokens = Vec::new();
    let mut position = 0;
    for word in text.split_whitespace() {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .collect();
        if !cleaned.is_empty() {
            tokens.push((cleaned.to_lowercase(), position));
            position += 1;
        }
    }
    tokens
}

// ── Posting ─────────────────────────────────────────────────────

/// A posting: a document reference plus positions within that document.
#[derive(Debug, Clone)]
pub struct Posting {
    pub doc_id: String,
    /// The positions at which the term appears (0-indexed word positions).
    pub positions: Vec<usize>,
}

// ── BooleanOp ───────────────────────────────────────────────────

/// Boolean query operators.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BooleanOp {
    And,
    Or,
    Not,
}

/// A single clause in a boolean query.
#[derive(Debug, Clone)]
pub struct BooleanClause {
    pub term: String,
    pub op: BooleanOp,
}

// ── ScoredResult ────────────────────────────────────────────────

/// A search result with TF-IDF score.
#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub doc_id: String,
    pub score: f64,
}

// ── IndexStats ──────────────────────────────────────────────────

/// Statistics about the index.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub num_documents: usize,
    pub num_terms: usize,
    pub total_tokens: usize,
}

// ── Serialized form ─────────────────────────────────────────────

/// Serializable representation of the index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SerializedIndex {
    /// term -> Vec<(doc_id, positions)>
    pub postings: BTreeMap<String, Vec<(String, Vec<usize>)>>,
    /// doc_id -> token count
    pub doc_lengths: BTreeMap<String, usize>,
}

// ── InvertedIndex ───────────────────────────────────────────────

/// Full-text inverted index with positional postings.
#[derive(Debug, Clone)]
pub struct InvertedIndex {
    /// term -> doc_id -> positions
    postings: HashMap<String, HashMap<String, Vec<usize>>>,
    /// doc_id -> number of tokens
    doc_lengths: HashMap<String, usize>,
}

impl InvertedIndex {
    /// Create a new empty index.
    pub fn new() -> Self {
        Self {
            postings: HashMap::new(),
            doc_lengths: HashMap::new(),
        }
    }

    /// Number of documents in the index.
    pub fn num_documents(&self) -> usize {
        self.doc_lengths.len()
    }

    /// Number of distinct terms.
    pub fn num_terms(&self) -> usize {
        self.postings.len()
    }

    /// Get statistics about the index.
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            num_documents: self.doc_lengths.len(),
            num_terms: self.postings.len(),
            total_tokens: self.doc_lengths.values().sum(),
        }
    }

    /// Add a document to the index.
    pub fn add_document(&mut self, doc_id: &str, text: &str) -> Result<(), IndexError> {
        if self.doc_lengths.contains_key(doc_id) {
            return Err(IndexError::DuplicateDocument(doc_id.to_string()));
        }

        let tokens = tokenize(text);
        self.doc_lengths.insert(doc_id.to_string(), tokens.len());

        for (term, pos) in &tokens {
            self.postings
                .entry(term.clone())
                .or_default()
                .entry(doc_id.to_string())
                .or_default()
                .push(*pos);
        }

        Ok(())
    }

    /// Remove a document from the index.
    pub fn remove_document(&mut self, doc_id: &str) -> Result<(), IndexError> {
        if !self.doc_lengths.contains_key(doc_id) {
            return Err(IndexError::DocumentNotFound(doc_id.to_string()));
        }

        self.doc_lengths.remove(doc_id);

        // Remove from all posting lists.
        let mut empty_terms = Vec::new();
        for (term, docs) in &mut self.postings {
            docs.remove(doc_id);
            if docs.is_empty() {
                empty_terms.push(term.clone());
            }
        }
        for term in empty_terms {
            self.postings.remove(&term);
        }

        Ok(())
    }

    /// Check if the index contains a document.
    pub fn contains_document(&self, doc_id: &str) -> bool {
        self.doc_lengths.contains_key(doc_id)
    }

    /// Get posting list for a term.
    pub fn get_postings(&self, term: &str) -> Vec<Posting> {
        let normalized = term.to_lowercase();
        match self.postings.get(&normalized) {
            Some(docs) => {
                let mut result: Vec<Posting> = docs
                    .iter()
                    .map(|(doc_id, positions)| Posting {
                        doc_id: doc_id.clone(),
                        positions: positions.clone(),
                    })
                    .collect();
                result.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));
                result
            }
            None => Vec::new(),
        }
    }

    /// Document frequency: how many documents contain the term.
    pub fn document_frequency(&self, term: &str) -> usize {
        let normalized = term.to_lowercase();
        self.postings
            .get(&normalized)
            .map_or(0, |docs| docs.len())
    }

    /// Term frequency: how many times a term appears in a specific document.
    pub fn term_frequency(&self, term: &str, doc_id: &str) -> usize {
        let normalized = term.to_lowercase();
        self.postings
            .get(&normalized)
            .and_then(|docs| docs.get(doc_id))
            .map_or(0, |positions| positions.len())
    }

    /// Compute TF-IDF score for a term in a document.
    pub fn tf_idf(&self, term: &str, doc_id: &str) -> f64 {
        let tf = self.term_frequency(term, doc_id);
        if tf == 0 {
            return 0.0;
        }
        let df = self.document_frequency(term);
        let n = self.num_documents();
        if df == 0 || n == 0 {
            return 0.0;
        }
        let tf_normalized = 1.0 + (tf as f64).ln();
        let idf = (n as f64 / df as f64).ln();
        tf_normalized * idf
    }

    /// Search for a single term and return scored results.
    pub fn search_term(&self, term: &str) -> Vec<ScoredResult> {
        let normalized = term.to_lowercase();
        let docs = match self.postings.get(&normalized) {
            Some(d) => d,
            None => return Vec::new(),
        };

        let mut results: Vec<ScoredResult> = docs
            .keys()
            .map(|doc_id| {
                let score = self.tf_idf(&normalized, doc_id);
                ScoredResult {
                    doc_id: doc_id.clone(),
                    score,
                }
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Multi-term search: tokenize the query and sum TF-IDF for each term.
    pub fn search(&self, query: &str) -> Result<Vec<ScoredResult>, IndexError> {
        let tokens = tokenize(query);
        if tokens.is_empty() {
            return Err(IndexError::EmptyQuery);
        }

        let mut scores: HashMap<String, f64> = HashMap::new();
        for (term, _pos) in &tokens {
            if let Some(docs) = self.postings.get(term) {
                for doc_id in docs.keys() {
                    let s = self.tf_idf(term, doc_id);
                    *scores.entry(doc_id.clone()).or_insert(0.0) += s;
                }
            }
        }

        let mut results: Vec<ScoredResult> = scores
            .into_iter()
            .map(|(doc_id, score)| ScoredResult { doc_id, score })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }

    /// Boolean query: AND, OR, NOT across terms.
    pub fn boolean_search(&self, clauses: &[BooleanClause]) -> Result<Vec<String>, IndexError> {
        if clauses.is_empty() {
            return Err(IndexError::EmptyQuery);
        }

        let all_docs: HashSet<String> = self.doc_lengths.keys().cloned().collect();
        let mut result: Option<HashSet<String>> = None;

        for clause in clauses {
            let normalized = clause.term.to_lowercase();
            let term_docs: HashSet<String> = self
                .postings
                .get(&normalized)
                .map(|docs| docs.keys().cloned().collect())
                .unwrap_or_default();

            let clause_set = match clause.op {
                BooleanOp::Not => all_docs.difference(&term_docs).cloned().collect(),
                _ => term_docs,
            };

            result = Some(match result {
                None => clause_set,
                Some(prev) => match clause.op {
                    BooleanOp::And => prev.intersection(&clause_set).cloned().collect(),
                    BooleanOp::Or => prev.union(&clause_set).cloned().collect(),
                    BooleanOp::Not => prev.intersection(&clause_set).cloned().collect(),
                },
            });
        }

        let mut docs: Vec<String> = result.unwrap_or_default().into_iter().collect();
        docs.sort();
        Ok(docs)
    }

    /// Phrase query: find documents where all terms appear consecutively.
    pub fn phrase_search(&self, phrase: &str) -> Result<Vec<String>, IndexError> {
        let tokens = tokenize(phrase);
        if tokens.is_empty() {
            return Err(IndexError::EmptyQuery);
        }

        let terms: Vec<String> = tokens.into_iter().map(|(t, _)| t).collect();

        // Start with documents containing the first term.
        let first_docs = match self.postings.get(&terms[0]) {
            Some(d) => d,
            None => return Ok(Vec::new()),
        };

        let mut matches = Vec::new();

        for (doc_id, first_positions) in first_docs {
            // Check each starting position.
            'outer: for &start_pos in first_positions {
                for (offset, term) in terms.iter().enumerate().skip(1) {
                    let expected_pos = start_pos + offset;
                    let has_at_pos = self
                        .postings
                        .get(term)
                        .and_then(|docs| docs.get(doc_id))
                        .map_or(false, |positions| positions.contains(&expected_pos));
                    if !has_at_pos {
                        continue 'outer;
                    }
                }
                // All terms matched consecutively.
                matches.push(doc_id.clone());
                break;
            }
        }

        matches.sort();
        matches.dedup();
        Ok(matches)
    }

    /// Serialize the index to a portable form.
    pub fn serialize(&self) -> SerializedIndex {
        let mut postings = BTreeMap::new();
        for (term, docs) in &self.postings {
            let mut entries: Vec<(String, Vec<usize>)> = docs
                .iter()
                .map(|(doc_id, positions)| (doc_id.clone(), positions.clone()))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            postings.insert(term.clone(), entries);
        }

        let doc_lengths: BTreeMap<String, usize> = self.doc_lengths.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        SerializedIndex {
            postings,
            doc_lengths,
        }
    }

    /// Deserialize from a portable form.
    pub fn deserialize(serialized: &SerializedIndex) -> Self {
        let mut postings: HashMap<String, HashMap<String, Vec<usize>>> = HashMap::new();
        for (term, entries) in &serialized.postings {
            let mut docs = HashMap::new();
            for (doc_id, positions) in entries {
                docs.insert(doc_id.clone(), positions.clone());
            }
            postings.insert(term.clone(), docs);
        }

        let doc_lengths: HashMap<String, usize> = serialized
            .doc_lengths
            .iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();

        Self {
            postings,
            doc_lengths,
        }
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, IndexError> {
        let serialized = self.serialize();
        serde_json::to_string(&serialized)
            .map_err(|e| IndexError::SerializationError(e.to_string()))
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, IndexError> {
        let serialized: SerializedIndex = serde_json::from_str(json)
            .map_err(|e| IndexError::SerializationError(e.to_string()))?;
        Ok(Self::deserialize(&serialized))
    }

    /// Average document length across the corpus.
    pub fn avg_doc_length(&self) -> f64 {
        if self.doc_lengths.is_empty() {
            return 0.0;
        }
        let total: usize = self.doc_lengths.values().sum();
        total as f64 / self.doc_lengths.len() as f64
    }

    /// Get the length (token count) of a document.
    pub fn doc_length(&self, doc_id: &str) -> Option<usize> {
        self.doc_lengths.get(doc_id).copied()
    }

    /// Get all document IDs.
    pub fn document_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.doc_lengths.keys().cloned().collect();
        ids.sort();
        ids
    }

    /// Get all indexed terms.
    pub fn terms(&self) -> Vec<String> {
        let mut terms: Vec<String> = self.postings.keys().cloned().collect();
        terms.sort();
        terms
    }
}

impl Default for InvertedIndex {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_index() -> InvertedIndex {
        let mut idx = InvertedIndex::new();
        idx.add_document("d1", "the quick brown fox jumps over the lazy dog").unwrap();
        idx.add_document("d2", "the quick brown cat sits on the mat").unwrap();
        idx.add_document("d3", "a fox and a dog are friends").unwrap();
        idx
    }

    #[test]
    fn test_tokenize_basic() {
        let tokens = tokenize("Hello, World! foo-bar");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], ("hello".to_string(), 0));
        assert_eq!(tokens[1], ("world".to_string(), 1));
        assert_eq!(tokens[2], ("foo-bar".to_string(), 2));
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = tokenize("");
        assert!(tokens.is_empty());
    }

    #[test]
    fn test_add_document() {
        let idx = build_index();
        assert_eq!(idx.num_documents(), 3);
        assert!(idx.contains_document("d1"));
    }

    #[test]
    fn test_duplicate_document() {
        let mut idx = InvertedIndex::new();
        idx.add_document("d1", "hello").unwrap();
        let err = idx.add_document("d1", "world").unwrap_err();
        assert!(matches!(err, IndexError::DuplicateDocument(_)));
    }

    #[test]
    fn test_remove_document() {
        let mut idx = build_index();
        idx.remove_document("d1").unwrap();
        assert_eq!(idx.num_documents(), 2);
        assert!(!idx.contains_document("d1"));
    }

    #[test]
    fn test_remove_nonexistent() {
        let mut idx = build_index();
        let err = idx.remove_document("d99").unwrap_err();
        assert!(matches!(err, IndexError::DocumentNotFound(_)));
    }

    #[test]
    fn test_document_frequency() {
        let idx = build_index();
        // "the" appears in d1 and d2
        assert_eq!(idx.document_frequency("the"), 2);
        // "fox" appears in d1 and d3
        assert_eq!(idx.document_frequency("fox"), 2);
        // "nonexistent" appears nowhere
        assert_eq!(idx.document_frequency("nonexistent"), 0);
    }

    #[test]
    fn test_term_frequency() {
        let idx = build_index();
        // "the" appears 2 times in d1 ("the quick... the lazy")
        assert_eq!(idx.term_frequency("the", "d1"), 2);
        // "fox" appears once in d1
        assert_eq!(idx.term_frequency("fox", "d1"), 1);
    }

    #[test]
    fn test_tf_idf_positive() {
        let idx = build_index();
        let score = idx.tf_idf("fox", "d1");
        assert!(score > 0.0);
    }

    #[test]
    fn test_tf_idf_zero_for_missing() {
        let idx = build_index();
        assert_eq!(idx.tf_idf("nonexistent", "d1"), 0.0);
        assert_eq!(idx.tf_idf("fox", "d99"), 0.0);
    }

    #[test]
    fn test_search_term() {
        let idx = build_index();
        let results = idx.search_term("fox");
        assert_eq!(results.len(), 2);
        assert!(results.iter().any(|r| r.doc_id == "d1"));
        assert!(results.iter().any(|r| r.doc_id == "d3"));
    }

    #[test]
    fn test_multi_term_search() {
        let idx = build_index();
        let results = idx.search("quick brown").unwrap();
        assert!(!results.is_empty());
        // d1 and d2 both have "quick" and "brown"
        assert!(results.iter().any(|r| r.doc_id == "d1"));
        assert!(results.iter().any(|r| r.doc_id == "d2"));
    }

    #[test]
    fn test_search_empty_query() {
        let idx = build_index();
        let err = idx.search("").unwrap_err();
        assert!(matches!(err, IndexError::EmptyQuery));
    }

    #[test]
    fn test_boolean_and() {
        let idx = build_index();
        let clauses = vec![
            BooleanClause { term: "fox".to_string(), op: BooleanOp::And },
            BooleanClause { term: "dog".to_string(), op: BooleanOp::And },
        ];
        let results = idx.boolean_search(&clauses).unwrap();
        // d1 has both fox and dog, d3 has both fox and dog
        assert!(results.contains(&"d1".to_string()));
        assert!(results.contains(&"d3".to_string()));
        assert!(!results.contains(&"d2".to_string()));
    }

    #[test]
    fn test_boolean_or() {
        let idx = build_index();
        let clauses = vec![
            BooleanClause { term: "cat".to_string(), op: BooleanOp::Or },
            BooleanClause { term: "fox".to_string(), op: BooleanOp::Or },
        ];
        let results = idx.boolean_search(&clauses).unwrap();
        // cat is in d2, fox in d1 and d3
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_boolean_not() {
        let idx = build_index();
        let clauses = vec![
            BooleanClause { term: "fox".to_string(), op: BooleanOp::And },
            BooleanClause { term: "cat".to_string(), op: BooleanOp::Not },
        ];
        let results = idx.boolean_search(&clauses).unwrap();
        // fox in d1, d3; exclude docs with cat (d2) — but d1,d3 don't have cat anyway
        assert!(results.contains(&"d1".to_string()));
        assert!(results.contains(&"d3".to_string()));
    }

    #[test]
    fn test_phrase_search() {
        let idx = build_index();
        let results = idx.phrase_search("quick brown fox").unwrap();
        assert_eq!(results, vec!["d1"]);
    }

    #[test]
    fn test_phrase_search_no_match() {
        let idx = build_index();
        let results = idx.phrase_search("brown quick").unwrap();
        // "brown quick" is not in consecutive order in any doc
        assert!(results.is_empty());
    }

    #[test]
    fn test_postings_retrieval() {
        let idx = build_index();
        let postings = idx.get_postings("fox");
        assert_eq!(postings.len(), 2);
        let d1_posting = postings.iter().find(|p| p.doc_id == "d1").unwrap();
        assert!(d1_posting.positions.contains(&3)); // "the quick brown fox" -> position 3
    }

    #[test]
    fn test_serialization_roundtrip() {
        let idx = build_index();
        let json = idx.to_json().unwrap();
        let idx2 = InvertedIndex::from_json(&json).unwrap();
        assert_eq!(idx2.num_documents(), idx.num_documents());
        assert_eq!(idx2.num_terms(), idx.num_terms());
        // Verify a specific posting survived the roundtrip
        let postings = idx2.get_postings("fox");
        assert_eq!(postings.len(), 2);
    }

    #[test]
    fn test_stats() {
        let idx = build_index();
        let stats = idx.stats();
        assert_eq!(stats.num_documents, 3);
        assert!(stats.num_terms > 0);
        assert!(stats.total_tokens > 0);
    }

    #[test]
    fn test_avg_doc_length() {
        let idx = build_index();
        let avg = idx.avg_doc_length();
        // d1=9 tokens, d2=8 tokens, d3=7 tokens => avg = 8.0
        assert!((avg - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_document_ids_sorted() {
        let idx = build_index();
        let ids = idx.document_ids();
        assert_eq!(ids, vec!["d1", "d2", "d3"]);
    }

    #[test]
    fn test_terms_sorted() {
        let idx = build_index();
        let terms = idx.terms();
        // Verify sorted
        for window in terms.windows(2) {
            assert!(window[0] <= window[1]);
        }
    }

    #[test]
    fn test_default_constructor() {
        let idx = InvertedIndex::default();
        assert_eq!(idx.num_documents(), 0);
        assert_eq!(idx.num_terms(), 0);
    }

    #[test]
    fn test_remove_cleans_postings() {
        let mut idx = InvertedIndex::new();
        idx.add_document("d1", "unique word").unwrap();
        idx.add_document("d2", "other text").unwrap();
        idx.remove_document("d1").unwrap();
        // "unique" and "word" should be gone since only d1 had them
        assert_eq!(idx.document_frequency("unique"), 0);
        assert_eq!(idx.document_frequency("word"), 0);
    }
}
