//! # Full-Text Search
//!
//! Provides full-text indexing and search capabilities.
//!
//! ## Features
//!
//! - Inverted index for efficient text search
//! - Tokenization with configurable analyzers
//! - Boolean queries (AND, OR, NOT)
//! - Phrase search
//! - Fuzzy matching
//! - Field boosting and scoring (BM25)
//!
//! ## Example
//!
//! ```rust,ignore
//! use joule_db_features::fulltext::{FullTextIndex, FullTextConfig, SearchQuery};
//!
//! let config = FullTextConfig::default()
//!     .with_stemming(true)
//!     .with_stopwords(true);
//!
//! let mut index = FullTextIndex::new(config);
//! index.add_document("doc1", "The quick brown fox jumps over the lazy dog");
//!
//! let results = index.search(SearchQuery::term("fox"));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for full-text index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextConfig {
    /// Enable stemming (reduces words to root form).
    pub stemming: bool,
    /// Enable stopword removal.
    pub stopwords: bool,
    /// Minimum token length.
    pub min_token_length: usize,
    /// Maximum token length.
    pub max_token_length: usize,
    /// Custom stopwords list.
    pub custom_stopwords: HashSet<String>,
    /// BM25 k1 parameter (term frequency saturation).
    pub bm25_k1: f32,
    /// BM25 b parameter (document length normalization).
    pub bm25_b: f32,
}

impl FullTextConfig {
    /// Create a new config with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable stemming.
    pub fn with_stemming(mut self, enabled: bool) -> Self {
        self.stemming = enabled;
        self
    }

    /// Enable or disable stopword removal.
    pub fn with_stopwords(mut self, enabled: bool) -> Self {
        self.stopwords = enabled;
        self
    }

    /// Set minimum token length.
    pub fn with_min_token_length(mut self, len: usize) -> Self {
        self.min_token_length = len;
        self
    }

    /// Add custom stopwords.
    pub fn with_custom_stopwords(mut self, words: Vec<String>) -> Self {
        self.custom_stopwords = words.into_iter().collect();
        self
    }

    /// Set BM25 parameters.
    pub fn with_bm25_params(mut self, k1: f32, b: f32) -> Self {
        self.bm25_k1 = k1;
        self.bm25_b = b;
        self
    }
}

impl Default for FullTextConfig {
    fn default() -> Self {
        Self {
            stemming: true,
            stopwords: true,
            min_token_length: 2,
            max_token_length: 100,
            custom_stopwords: HashSet::new(),
            bm25_k1: 1.2,
            bm25_b: 0.75,
        }
    }
}

/// A search query.
#[derive(Debug, Clone)]
pub enum SearchQuery {
    /// Match a single term.
    Term(String),
    /// Match a phrase (terms in order).
    Phrase(Vec<String>),
    /// Match all sub-queries (AND).
    And(Vec<SearchQuery>),
    /// Match any sub-query (OR).
    Or(Vec<SearchQuery>),
    /// Exclude matches (NOT).
    Not(Box<SearchQuery>),
    /// Fuzzy match with edit distance.
    Fuzzy { term: String, distance: usize },
    /// Prefix match.
    Prefix(String),
    /// Wildcard match.
    Wildcard(String),
    /// Match all documents.
    MatchAll,
    /// Boost query score.
    Boost { query: Box<SearchQuery>, boost: f32 },
}

impl SearchQuery {
    /// Create a term query.
    pub fn term(term: impl Into<String>) -> Self {
        SearchQuery::Term(term.into())
    }

    /// Create a phrase query.
    pub fn phrase(terms: Vec<impl Into<String>>) -> Self {
        SearchQuery::Phrase(terms.into_iter().map(|t| t.into()).collect())
    }

    /// Create an AND query.
    pub fn and(queries: Vec<SearchQuery>) -> Self {
        SearchQuery::And(queries)
    }

    /// Create an OR query.
    pub fn or(queries: Vec<SearchQuery>) -> Self {
        SearchQuery::Or(queries)
    }

    /// Create a NOT query.
    pub fn not(query: SearchQuery) -> Self {
        SearchQuery::Not(Box::new(query))
    }

    /// Create a fuzzy query.
    pub fn fuzzy(term: impl Into<String>, distance: usize) -> Self {
        SearchQuery::Fuzzy {
            term: term.into(),
            distance,
        }
    }

    /// Create a prefix query.
    pub fn prefix(prefix: impl Into<String>) -> Self {
        SearchQuery::Prefix(prefix.into())
    }

    /// Boost the query score.
    pub fn boost(self, boost: f32) -> Self {
        SearchQuery::Boost {
            query: Box::new(self),
            boost,
        }
    }

    /// Parse a simple query string.
    /// Supports: term, "phrase", term1 AND term2, term1 OR term2, -term (NOT)
    pub fn parse(query: &str) -> Self {
        let query = query.trim();

        // Check for phrase query
        if query.starts_with('"') && query.ends_with('"') && query.len() > 2 {
            let inner = &query[1..query.len() - 1];
            let terms: Vec<String> = inner.split_whitespace().map(|s| s.to_lowercase()).collect();
            return SearchQuery::Phrase(terms);
        }

        // Check for boolean operators
        if query.contains(" AND ") {
            let parts: Vec<SearchQuery> = query
                .split(" AND ")
                .map(|p| SearchQuery::parse(p.trim()))
                .collect();
            return SearchQuery::And(parts);
        }

        if query.contains(" OR ") {
            let parts: Vec<SearchQuery> = query
                .split(" OR ")
                .map(|p| SearchQuery::parse(p.trim()))
                .collect();
            return SearchQuery::Or(parts);
        }

        // Check for NOT prefix
        if query.starts_with('-') || query.starts_with("NOT ") {
            let term = if query.starts_with('-') {
                &query[1..]
            } else {
                &query[4..]
            };
            return SearchQuery::Not(Box::new(SearchQuery::parse(term.trim())));
        }

        // Check for prefix query
        if query.ends_with('*') {
            return SearchQuery::Prefix(query[..query.len() - 1].to_lowercase());
        }

        // Check for fuzzy query
        if query.ends_with('~') {
            return SearchQuery::Fuzzy {
                term: query[..query.len() - 1].to_lowercase(),
                distance: 2,
            };
        }

        // Simple term query
        SearchQuery::Term(query.to_lowercase())
    }
}

/// A search hit (matched document).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    /// Document ID.
    pub id: String,
    /// Relevance score.
    pub score: f32,
    /// Matched terms with positions.
    pub highlights: Vec<Highlight>,
}

/// A highlighted match within a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Highlight {
    /// The matched term.
    pub term: String,
    /// Positions where the term appears.
    pub positions: Vec<usize>,
}

/// Posting list entry (document + positions).
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Posting {
    doc_id: usize,
    positions: Vec<usize>,
    term_frequency: f32,
}

/// Stored document metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct DocumentMeta {
    id: String,
    length: usize, // Number of tokens
    field_lengths: HashMap<String, usize>,
}

/// Full-text search index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullTextIndex {
    config: FullTextConfig,
    /// Inverted index: term -> posting list.
    inverted_index: HashMap<String, Vec<Posting>>,
    /// Document metadata.
    documents: Vec<DocumentMeta>,
    /// ID to index mapping.
    id_to_index: HashMap<String, usize>,
    /// Original document content (for highlighting).
    content: HashMap<usize, String>,
    /// Total tokens across all documents.
    total_tokens: usize,
    /// Average document length.
    avg_doc_length: f32,
}

impl FullTextIndex {
    /// Create a new full-text index.
    pub fn new(config: FullTextConfig) -> Self {
        Self {
            config,
            inverted_index: HashMap::new(),
            documents: Vec::new(),
            id_to_index: HashMap::new(),
            content: HashMap::new(),
            total_tokens: 0,
            avg_doc_length: 0.0,
        }
    }

    /// Create an index with default configuration.
    pub fn default_index() -> Self {
        Self::new(FullTextConfig::default())
    }

    /// Get the number of documents.
    pub fn len(&self) -> usize {
        self.documents.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.documents.is_empty()
    }

    /// Get the configuration.
    pub fn config(&self) -> &FullTextConfig {
        &self.config
    }

    /// Add a document to the index.
    pub fn add_document(
        &mut self,
        id: impl Into<String>,
        content: &str,
    ) -> Result<(), FullTextError> {
        let id = id.into();

        if self.id_to_index.contains_key(&id) {
            return Err(FullTextError::DuplicateDocument(id));
        }

        let doc_index = self.documents.len();
        let tokens = self.tokenize(content);
        let doc_length = tokens.len();

        // Build position map
        let mut term_positions: HashMap<String, Vec<usize>> = HashMap::new();
        for (pos, token) in tokens.iter().enumerate() {
            term_positions.entry(token.clone()).or_default().push(pos);
        }

        // Add to inverted index
        for (term, positions) in term_positions {
            let tf = positions.len() as f32 / doc_length as f32;
            let posting = Posting {
                doc_id: doc_index,
                positions,
                term_frequency: tf,
            };
            self.inverted_index.entry(term).or_default().push(posting);
        }

        // Store document metadata
        self.documents.push(DocumentMeta {
            id: id.clone(),
            length: doc_length,
            field_lengths: HashMap::new(),
        });

        self.id_to_index.insert(id, doc_index);
        self.content.insert(doc_index, content.to_string());

        // Update statistics
        self.total_tokens += doc_length;
        self.avg_doc_length = self.total_tokens as f32 / self.documents.len() as f32;

        Ok(())
    }

    /// Tokenize text into terms.
    fn tokenize(&self, text: &str) -> Vec<String> {
        let mut tokens = Vec::new();

        // Split on non-alphanumeric characters
        for word in text.split(|c: char| !c.is_alphanumeric()) {
            let word = word.to_lowercase();

            // Skip if too short or too long
            if word.len() < self.config.min_token_length
                || word.len() > self.config.max_token_length
            {
                continue;
            }

            // Skip stopwords
            if self.config.stopwords && self.is_stopword(&word) {
                continue;
            }

            // Apply stemming
            let token = if self.config.stemming {
                self.stem(&word)
            } else {
                word
            };

            tokens.push(token);
        }

        tokens
    }

    /// Check if a word is a stopword.
    fn is_stopword(&self, word: &str) -> bool {
        if self.config.custom_stopwords.contains(word) {
            return true;
        }

        // Default English stopwords
        const STOPWORDS: &[&str] = &[
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he", "in",
            "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with",
            "the", "this", "but", "they", "have", "had", "what", "when", "where", "who", "which",
            "why", "how",
        ];

        STOPWORDS.contains(&word)
    }

    /// Simple Porter-like stemmer.
    fn stem(&self, word: &str) -> String {
        let mut result = word.to_string();

        // Very simplified stemming rules
        if result.ends_with("ing") && result.len() > 5 {
            result = result[..result.len() - 3].to_string();
            if result.ends_with("nn") || result.ends_with("pp") || result.ends_with("tt") {
                result.pop();
            }
        } else if result.ends_with("ed") && result.len() > 4 {
            result = result[..result.len() - 2].to_string();
        } else if result.ends_with("ly") && result.len() > 4 {
            result = result[..result.len() - 2].to_string();
        } else if result.ends_with("ness") && result.len() > 6 {
            result = result[..result.len() - 4].to_string();
        } else if result.ends_with("ment") && result.len() > 6 {
            result = result[..result.len() - 4].to_string();
        } else if result.ends_with("ies") && result.len() > 4 {
            result = result[..result.len() - 3].to_string() + "y";
        } else if result.ends_with("es") && result.len() > 4 {
            result = result[..result.len() - 2].to_string();
        } else if result.ends_with('s') && result.len() > 3 && !result.ends_with("ss") {
            result = result[..result.len() - 1].to_string();
        }

        result
    }

    /// Search the index.
    pub fn search(&self, query: SearchQuery) -> Vec<SearchHit> {
        self.search_with_limit(query, 100)
    }

    /// Search with a result limit.
    pub fn search_with_limit(&self, query: SearchQuery, limit: usize) -> Vec<SearchHit> {
        let matching_docs = self.execute_query(&query);

        let mut hits: Vec<SearchHit> = matching_docs
            .into_iter()
            .map(|(doc_idx, score, highlights)| {
                let doc = &self.documents[doc_idx];
                SearchHit {
                    id: doc.id.clone(),
                    score,
                    highlights,
                }
            })
            .collect();

        // Sort by score descending
        hits.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        hits.truncate(limit);

        hits
    }

    /// Execute a query and return matching documents with scores.
    fn execute_query(&self, query: &SearchQuery) -> Vec<(usize, f32, Vec<Highlight>)> {
        match query {
            SearchQuery::Term(term) => self.search_term(term),
            SearchQuery::Phrase(terms) => self.search_phrase(terms),
            SearchQuery::And(queries) => self.search_and(queries),
            SearchQuery::Or(queries) => self.search_or(queries),
            SearchQuery::Not(inner) => self.search_not(inner),
            SearchQuery::Fuzzy { term, distance } => self.search_fuzzy(term, *distance),
            SearchQuery::Prefix(prefix) => self.search_prefix(prefix),
            SearchQuery::Wildcard(pattern) => self.search_wildcard(pattern),
            SearchQuery::MatchAll => self.search_match_all(),
            SearchQuery::Boost { query, boost } => {
                let mut results = self.execute_query(query);
                for (_, score, _) in &mut results {
                    *score *= boost;
                }
                results
            }
        }
    }

    /// Search for a single term.
    fn search_term(&self, term: &str) -> Vec<(usize, f32, Vec<Highlight>)> {
        let normalized = if self.config.stemming {
            self.stem(&term.to_lowercase())
        } else {
            term.to_lowercase()
        };

        let Some(postings) = self.inverted_index.get(&normalized) else {
            return Vec::new();
        };

        let idf = self.calculate_idf(postings.len());

        postings
            .iter()
            .map(|posting| {
                let doc = &self.documents[posting.doc_id];
                let score = self.calculate_bm25(posting.term_frequency, doc.length, idf);
                let highlight = Highlight {
                    term: term.to_string(),
                    positions: posting.positions.clone(),
                };
                (posting.doc_id, score, vec![highlight])
            })
            .collect()
    }

    /// Search for a phrase.
    fn search_phrase(&self, terms: &[String]) -> Vec<(usize, f32, Vec<Highlight>)> {
        if terms.is_empty() {
            return Vec::new();
        }

        let normalized: Vec<String> = terms
            .iter()
            .map(|t| {
                if self.config.stemming {
                    self.stem(&t.to_lowercase())
                } else {
                    t.to_lowercase()
                }
            })
            .collect();

        // Get postings for all terms
        let mut term_postings: Vec<&Vec<Posting>> = Vec::new();
        for term in &normalized {
            match self.inverted_index.get(term) {
                Some(postings) => term_postings.push(postings),
                None => return Vec::new(),
            }
        }

        // Find documents containing all terms
        let doc_ids: HashSet<usize> = term_postings[0].iter().map(|p| p.doc_id).collect();
        let mut common_docs: HashSet<usize> = doc_ids;

        for postings in &term_postings[1..] {
            let doc_ids: HashSet<usize> = postings.iter().map(|p| p.doc_id).collect();
            common_docs = common_docs.intersection(&doc_ids).copied().collect();
        }

        // Check phrase positions in each common document
        let mut results = Vec::new();

        for doc_id in common_docs {
            // Get positions for each term in this document
            let positions: Vec<&Vec<usize>> = term_postings
                .iter()
                .filter_map(|postings| {
                    postings
                        .iter()
                        .find(|p| p.doc_id == doc_id)
                        .map(|p| &p.positions)
                })
                .collect();

            if positions.len() != normalized.len() {
                continue;
            }

            // Check if terms appear consecutively
            let mut phrase_positions = Vec::new();
            for &start_pos in positions[0] {
                let mut matches = true;
                for (i, term_positions) in positions.iter().enumerate().skip(1) {
                    if !term_positions.contains(&(start_pos + i)) {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    phrase_positions.push(start_pos);
                }
            }

            if !phrase_positions.is_empty() {
                let doc = &self.documents[doc_id];
                let score = phrase_positions.len() as f32 / doc.length as f32 * 10.0;
                let highlight = Highlight {
                    term: terms.join(" "),
                    positions: phrase_positions,
                };
                results.push((doc_id, score, vec![highlight]));
            }
        }

        results
    }

    /// AND query - intersection of results.
    fn search_and(&self, queries: &[SearchQuery]) -> Vec<(usize, f32, Vec<Highlight>)> {
        if queries.is_empty() {
            return Vec::new();
        }

        let mut results_map: HashMap<usize, (f32, Vec<Highlight>)> = HashMap::new();

        // Execute first query
        for (doc_id, score, highlights) in self.execute_query(&queries[0]) {
            results_map.insert(doc_id, (score, highlights));
        }

        // Intersect with remaining queries
        for query in &queries[1..] {
            let query_results = self.execute_query(query);
            let query_doc_ids: HashSet<usize> =
                query_results.iter().map(|(id, _, _)| *id).collect();

            // Remove documents not in this query's results
            results_map.retain(|id, _| query_doc_ids.contains(id));

            // Add scores and highlights
            for (doc_id, score, highlights) in query_results {
                if let Some((existing_score, existing_highlights)) = results_map.get_mut(&doc_id) {
                    *existing_score += score;
                    existing_highlights.extend(highlights);
                }
            }
        }

        results_map
            .into_iter()
            .map(|(id, (score, highlights))| (id, score, highlights))
            .collect()
    }

    /// OR query - union of results.
    fn search_or(&self, queries: &[SearchQuery]) -> Vec<(usize, f32, Vec<Highlight>)> {
        let mut results_map: HashMap<usize, (f32, Vec<Highlight>)> = HashMap::new();

        for query in queries {
            for (doc_id, score, highlights) in self.execute_query(query) {
                results_map
                    .entry(doc_id)
                    .and_modify(|(s, h)| {
                        *s = s.max(score);
                        h.extend(highlights.clone());
                    })
                    .or_insert((score, highlights));
            }
        }

        results_map
            .into_iter()
            .map(|(id, (score, highlights))| (id, score, highlights))
            .collect()
    }

    /// NOT query - exclude matches.
    fn search_not(&self, query: &SearchQuery) -> Vec<(usize, f32, Vec<Highlight>)> {
        let excluded: HashSet<usize> = self
            .execute_query(query)
            .into_iter()
            .map(|(id, _, _)| id)
            .collect();

        // Return all documents NOT in excluded set
        self.documents
            .iter()
            .enumerate()
            .filter(|(idx, _)| !excluded.contains(idx))
            .map(|(idx, _)| (idx, 1.0, Vec::new()))
            .collect()
    }

    /// Fuzzy search with edit distance.
    fn search_fuzzy(&self, term: &str, max_distance: usize) -> Vec<(usize, f32, Vec<Highlight>)> {
        let normalized = term.to_lowercase();
        let mut matching_terms = Vec::new();

        // Find all terms within edit distance
        for index_term in self.inverted_index.keys() {
            if levenshtein_distance(&normalized, index_term) <= max_distance {
                matching_terms.push(index_term.clone());
            }
        }

        // Search for all matching terms (OR)
        let queries: Vec<SearchQuery> = matching_terms
            .iter()
            .map(|t| SearchQuery::Term(t.clone()))
            .collect();

        if queries.is_empty() {
            Vec::new()
        } else {
            self.search_or(&queries)
        }
    }

    /// Prefix search.
    fn search_prefix(&self, prefix: &str) -> Vec<(usize, f32, Vec<Highlight>)> {
        let normalized = prefix.to_lowercase();
        let matching_terms: Vec<String> = self
            .inverted_index
            .keys()
            .filter(|term| term.starts_with(&normalized))
            .cloned()
            .collect();

        let queries: Vec<SearchQuery> = matching_terms
            .iter()
            .map(|t| SearchQuery::Term(t.clone()))
            .collect();

        if queries.is_empty() {
            Vec::new()
        } else {
            self.search_or(&queries)
        }
    }

    /// Wildcard search (simple * matching).
    fn search_wildcard(&self, pattern: &str) -> Vec<(usize, f32, Vec<Highlight>)> {
        let normalized = pattern.to_lowercase();
        let parts: Vec<&str> = normalized.split('*').collect();

        let matching_terms: Vec<String> = self
            .inverted_index
            .keys()
            .filter(|term| {
                if parts.len() == 1 {
                    return **term == normalized;
                }

                let mut pos = 0;
                for (i, part) in parts.iter().enumerate() {
                    if part.is_empty() {
                        continue;
                    }

                    match term[pos..].find(part) {
                        Some(idx) => {
                            if i == 0 && idx != 0 {
                                return false;
                            }
                            pos += idx + part.len();
                        }
                        None => return false,
                    }
                }

                // If pattern doesn't end with *, term must end at pos
                if !normalized.ends_with('*') && pos != term.len() {
                    return false;
                }

                true
            })
            .cloned()
            .collect();

        let queries: Vec<SearchQuery> = matching_terms
            .iter()
            .map(|t| SearchQuery::Term(t.clone()))
            .collect();

        if queries.is_empty() {
            Vec::new()
        } else {
            self.search_or(&queries)
        }
    }

    /// Match all documents.
    fn search_match_all(&self) -> Vec<(usize, f32, Vec<Highlight>)> {
        self.documents
            .iter()
            .enumerate()
            .map(|(idx, _)| (idx, 1.0, Vec::new()))
            .collect()
    }

    /// Calculate IDF (Inverse Document Frequency).
    fn calculate_idf(&self, doc_frequency: usize) -> f32 {
        let n = self.documents.len() as f32;
        let df = doc_frequency as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Calculate BM25 score for a term in a document.
    fn calculate_bm25(&self, tf: f32, doc_length: usize, idf: f32) -> f32 {
        let k1 = self.config.bm25_k1;
        let b = self.config.bm25_b;
        let dl = doc_length as f32;
        let avgdl = self.avg_doc_length;

        idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))
    }

    /// Check if a document exists.
    pub fn contains(&self, id: &str) -> bool {
        self.id_to_index.contains_key(id)
    }

    /// Remove a document from the index.
    pub fn remove(&mut self, id: &str) -> bool {
        let Some(&doc_idx) = self.id_to_index.get(id) else {
            return false;
        };

        // Remove from inverted index
        for postings in self.inverted_index.values_mut() {
            postings.retain(|p| p.doc_id != doc_idx);
        }

        // Remove empty posting lists
        self.inverted_index
            .retain(|_, postings| !postings.is_empty());

        // Note: We don't actually remove the document from the documents vector
        // to avoid reindexing all document IDs. In a production system, you'd
        // want to periodically compact the index.

        self.id_to_index.remove(id);
        self.content.remove(&doc_idx);

        true
    }

    /// Get all unique terms in the index.
    pub fn terms(&self) -> impl Iterator<Item = &str> {
        self.inverted_index.keys().map(|s| s.as_str())
    }

    /// Get document frequency for a term.
    pub fn document_frequency(&self, term: &str) -> usize {
        self.inverted_index.get(term).map(|p| p.len()).unwrap_or(0)
    }

    /// Get the original content of a document.
    pub fn get_content(&self, id: &str) -> Option<&str> {
        self.id_to_index
            .get(id)
            .and_then(|&idx| self.content.get(&idx))
            .map(|s| s.as_str())
    }

    /// Clear the index.
    pub fn clear(&mut self) {
        self.inverted_index.clear();
        self.documents.clear();
        self.id_to_index.clear();
        self.content.clear();
        self.total_tokens = 0;
        self.avg_doc_length = 0.0;
    }

    /// Get index statistics.
    pub fn stats(&self) -> IndexStats {
        IndexStats {
            document_count: self.documents.len(),
            term_count: self.inverted_index.len(),
            total_tokens: self.total_tokens,
            avg_document_length: self.avg_doc_length,
        }
    }
}

/// Index statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexStats {
    pub document_count: usize,
    pub term_count: usize,
    pub total_tokens: usize,
    pub avg_document_length: f32,
}

/// Full-text search errors.
#[derive(Debug, Clone, thiserror::Error)]
pub enum FullTextError {
    #[error("Duplicate document ID: {0}")]
    DuplicateDocument(String),

    #[error("Document not found: {0}")]
    DocumentNotFound(String),

    #[error("Invalid query: {0}")]
    InvalidQuery(String),
}

/// Calculate Levenshtein edit distance between two strings.
fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let m = a_chars.len();
    let n = b_chars.len();

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut dp = vec![vec![0usize; n + 1]; m + 1];

    for i in 0..=m {
        dp[i][0] = i;
    }
    for j in 0..=n {
        dp[0][j] = j;
    }

    for i in 1..=m {
        for j in 1..=n {
            let cost = if a_chars[i - 1] == b_chars[j - 1] {
                0
            } else {
                1
            };
            dp[i][j] = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
        }
    }

    dp[m][n]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let mut index = FullTextIndex::default_index();

        index
            .add_document("doc1", "The quick brown fox jumps over the lazy dog")
            .unwrap();
        index
            .add_document("doc2", "A quick brown dog runs in the park")
            .unwrap();
        index
            .add_document("doc3", "The lazy cat sleeps all day")
            .unwrap();

        let results = index.search(SearchQuery::term("quick"));
        assert_eq!(results.len(), 2);

        let results = index.search(SearchQuery::term("cat"));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc3");
    }

    #[test]
    fn test_phrase_search() {
        let mut index = FullTextIndex::new(FullTextConfig::default().with_stemming(false));

        index.add_document("doc1", "the quick brown fox").unwrap();
        index.add_document("doc2", "quick brown dog").unwrap();
        index.add_document("doc3", "the brown quick fox").unwrap(); // Different order

        let results = index.search(SearchQuery::phrase(vec!["quick", "brown"]));

        // Should match doc1 and doc2, but not doc3 (different order)
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_and_query() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "quick fox").unwrap();
        index.add_document("doc2", "quick dog").unwrap();
        index.add_document("doc3", "slow fox").unwrap();

        let results = index.search(SearchQuery::and(vec![
            SearchQuery::term("quick"),
            SearchQuery::term("fox"),
        ]));

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_or_query() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "quick fox").unwrap();
        index.add_document("doc2", "slow dog").unwrap();
        index.add_document("doc3", "lazy cat").unwrap();

        let results = index.search(SearchQuery::or(vec![
            SearchQuery::term("fox"),
            SearchQuery::term("cat"),
        ]));

        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_not_query() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "quick fox").unwrap();
        index.add_document("doc2", "slow dog").unwrap();
        index.add_document("doc3", "lazy cat").unwrap();

        let results = index.search(SearchQuery::not(SearchQuery::term("fox")));

        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.id != "doc1"));
    }

    #[test]
    fn test_prefix_search() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "programming is fun").unwrap();
        index.add_document("doc2", "programs run fast").unwrap();
        index.add_document("doc3", "cats are cute").unwrap();

        let results = index.search(SearchQuery::prefix("prog"));
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_fuzzy_search() {
        let mut index = FullTextIndex::new(FullTextConfig::default().with_stemming(false));

        index.add_document("doc1", "hello world").unwrap();
        index.add_document("doc2", "hallo there").unwrap();
        index.add_document("doc3", "goodbye world").unwrap();

        let results = index.search(SearchQuery::fuzzy("hello", 2));

        // Should match "hello" and "hallo" (1 edit distance)
        assert!(results.len() >= 2);
    }

    #[test]
    fn test_query_parsing() {
        // Term query
        let q = SearchQuery::parse("fox");
        assert!(matches!(q, SearchQuery::Term(_)));

        // Phrase query
        let q = SearchQuery::parse("\"quick brown\"");
        assert!(matches!(q, SearchQuery::Phrase(_)));

        // AND query
        let q = SearchQuery::parse("quick AND fox");
        assert!(matches!(q, SearchQuery::And(_)));

        // OR query
        let q = SearchQuery::parse("quick OR slow");
        assert!(matches!(q, SearchQuery::Or(_)));

        // NOT query
        let q = SearchQuery::parse("-fox");
        assert!(matches!(q, SearchQuery::Not(_)));

        // Prefix query
        let q = SearchQuery::parse("prog*");
        assert!(matches!(q, SearchQuery::Prefix(_)));
    }

    #[test]
    fn test_stemming() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "running runners run").unwrap();
        index.add_document("doc2", "jumping high").unwrap();

        // Should match because of stemming
        let results = index.search(SearchQuery::term("runs"));
        assert!(!results.is_empty());
    }

    #[test]
    fn test_stopwords() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "the quick brown fox").unwrap();

        // "the" is a stopword and shouldn't be indexed
        let results = index.search(SearchQuery::term("the"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_duplicate_document() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "hello world").unwrap();
        let result = index.add_document("doc1", "another content");

        assert!(matches!(result, Err(FullTextError::DuplicateDocument(_))));
    }

    #[test]
    fn test_levenshtein() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("hello", "helo"), 1);
        assert_eq!(levenshtein_distance("cat", "dog"), 3);
        assert_eq!(levenshtein_distance("", "abc"), 3);
    }

    #[test]
    fn test_index_stats() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "hello world").unwrap();
        index.add_document("doc2", "foo bar baz").unwrap();

        let stats = index.stats();
        assert_eq!(stats.document_count, 2);
        assert!(stats.term_count > 0);
    }

    #[test]
    fn test_boost() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "important fox").unwrap();
        index.add_document("doc2", "regular fox").unwrap();

        let results = index.search(SearchQuery::or(vec![
            SearchQuery::term("important").boost(10.0),
            SearchQuery::term("fox"),
        ]));

        // Document with "important" should score higher
        assert_eq!(results[0].id, "doc1");
    }

    #[test]
    fn test_remove_document() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "hello world").unwrap();
        index.add_document("doc2", "goodbye world").unwrap();

        assert!(index.contains("doc1"));
        assert!(index.remove("doc1"));
        assert!(!index.contains("doc1"));

        let results = index.search(SearchQuery::term("hello"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_get_content() {
        let mut index = FullTextIndex::default_index();

        let content = "hello world";
        index.add_document("doc1", content).unwrap();

        assert_eq!(index.get_content("doc1"), Some(content));
        assert_eq!(index.get_content("nonexistent"), None);
    }

    #[test]
    fn test_clear() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "hello").unwrap();
        index.add_document("doc2", "world").unwrap();

        index.clear();

        assert!(index.is_empty());
        assert_eq!(index.stats().document_count, 0);
    }

    #[test]
    fn test_match_all() {
        let mut index = FullTextIndex::default_index();

        index.add_document("doc1", "hello").unwrap();
        index.add_document("doc2", "world").unwrap();
        index.add_document("doc3", "foo").unwrap();

        let results = index.search(SearchQuery::MatchAll);
        assert_eq!(results.len(), 3);
    }
}
