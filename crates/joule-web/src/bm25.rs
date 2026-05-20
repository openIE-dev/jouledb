//! BM25 ranking algorithm.
//!
//! Document frequency tracking, configurable k1/b parameters, field-length
//! normalization, multi-field scoring with field weights, batch scoring,
//! and relevance feedback.

use std::collections::HashMap;

// ── Configuration ───────────────────────────────────────────────

/// BM25 tuning parameters.
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Term saturation parameter. Higher values slow saturation. Default 1.2.
    pub k1: f64,
    /// Length normalization parameter (0.0 = none, 1.0 = full). Default 0.75.
    pub b: f64,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self { k1: 1.2, b: 0.75 }
    }
}

// ── Document ────────────────────────────────────────────────────

/// A document registered in the scorer with per-field token counts.
#[derive(Debug, Clone)]
struct DocumentEntry {
    /// field_name -> Vec<token>
    field_tokens: HashMap<String, Vec<String>>,
}

// ── FieldWeight ─────────────────────────────────────────────────

/// Weight assigned to a field for scoring purposes.
#[derive(Debug, Clone)]
pub struct FieldWeight {
    pub field: String,
    pub weight: f64,
}

// ── ScoredDoc ───────────────────────────────────────────────────

/// A scored document result.
#[derive(Debug, Clone)]
pub struct Bm25Result {
    pub doc_id: String,
    pub score: f64,
}

// ── RelevanceFeedback ───────────────────────────────────────────

/// Relevance feedback: expand queries with terms from known-relevant docs.
#[derive(Debug, Clone)]
pub struct RelevanceFeedback {
    /// doc_ids the user has marked relevant.
    pub relevant_doc_ids: Vec<String>,
    /// Number of expansion terms to extract.
    pub num_expansion_terms: usize,
}

// ── Bm25Scorer ──────────────────────────────────────────────────

/// BM25 scoring engine.
#[derive(Debug, Clone)]
pub struct Bm25Scorer {
    config: Bm25Config,
    documents: HashMap<String, DocumentEntry>,
    /// term -> set of doc_ids containing it (per-field flattened).
    doc_frequency: HashMap<String, Vec<String>>,
    /// field -> sum of all doc lengths in that field.
    total_field_lengths: HashMap<String, usize>,
    /// field -> count of docs having that field.
    field_doc_counts: HashMap<String, usize>,
}

impl Bm25Scorer {
    /// Create a new scorer with default configuration.
    pub fn new() -> Self {
        Self::with_config(Bm25Config::default())
    }

    /// Create a scorer with custom configuration.
    pub fn with_config(config: Bm25Config) -> Self {
        Self {
            config,
            documents: HashMap::new(),
            doc_frequency: HashMap::new(),
            total_field_lengths: HashMap::new(),
            field_doc_counts: HashMap::new(),
        }
    }

    /// Number of indexed documents.
    pub fn num_documents(&self) -> usize {
        self.documents.len()
    }

    /// Tokenize text: split on whitespace, lowercase, strip non-alphanumeric.
    fn tokenize(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| {
                w.chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect::<String>()
                    .to_lowercase()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }

    /// Add a document with a single default field.
    pub fn add_document(&mut self, doc_id: &str, text: &str) {
        let mut fields = HashMap::new();
        fields.insert("_default".to_string(), text.to_string());
        self.add_document_fields(doc_id, &fields);
    }

    /// Add a document with multiple named fields.
    pub fn add_document_fields(&mut self, doc_id: &str, fields: &HashMap<String, String>) {
        let mut field_tokens = HashMap::new();
        let mut all_terms_for_doc: Vec<String> = Vec::new();

        for (field_name, text) in fields {
            let tokens = Self::tokenize(text);

            *self.total_field_lengths.entry(field_name.clone()).or_insert(0) += tokens.len();
            *self.field_doc_counts.entry(field_name.clone()).or_insert(0) += 1;

            all_terms_for_doc.extend(tokens.clone());
            field_tokens.insert(field_name.clone(), tokens);
        }

        // Update document frequency (deduplicate terms per document).
        let mut seen_terms: Vec<String> = all_terms_for_doc.clone();
        seen_terms.sort();
        seen_terms.dedup();
        for term in &seen_terms {
            self.doc_frequency
                .entry(term.clone())
                .or_default()
                .push(doc_id.to_string());
        }

        self.documents.insert(
            doc_id.to_string(),
            DocumentEntry { field_tokens },
        );
    }

    /// Remove a document.
    pub fn remove_document(&mut self, doc_id: &str) -> bool {
        let entry = match self.documents.remove(doc_id) {
            Some(e) => e,
            None => return false,
        };

        for (field_name, tokens) in &entry.field_tokens {
            if let Some(total) = self.total_field_lengths.get_mut(field_name) {
                *total = total.saturating_sub(tokens.len());
            }
            if let Some(count) = self.field_doc_counts.get_mut(field_name) {
                *count = count.saturating_sub(1);
            }
        }

        // Remove from document frequency lists.
        for docs in self.doc_frequency.values_mut() {
            docs.retain(|d| d != doc_id);
        }
        self.doc_frequency.retain(|_, docs| !docs.is_empty());

        true
    }

    /// Average field length for a given field.
    pub fn avg_field_length(&self, field: &str) -> f64 {
        let total = *self.total_field_lengths.get(field).unwrap_or(&0);
        let count = *self.field_doc_counts.get(field).unwrap_or(&0);
        if count == 0 {
            0.0
        } else {
            total as f64 / count as f64
        }
    }

    /// Document frequency: how many documents contain a term.
    pub fn document_frequency(&self, term: &str) -> usize {
        let normalized = term.to_lowercase();
        self.doc_frequency.get(&normalized).map_or(0, |v| v.len())
    }

    /// IDF component using the standard BM25 formula.
    fn idf(&self, term: &str) -> f64 {
        let n = self.documents.len() as f64;
        let df = self.document_frequency(term) as f64;
        if df == 0.0 {
            return 0.0;
        }
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// Score a single document against a query for a specific field.
    fn score_field(&self, doc_id: &str, query_terms: &[String], field: &str) -> f64 {
        let entry = match self.documents.get(doc_id) {
            Some(e) => e,
            None => return 0.0,
        };

        let tokens = match entry.field_tokens.get(field) {
            Some(t) => t,
            None => return 0.0,
        };

        let dl = tokens.len() as f64;
        let avgdl = self.avg_field_length(field);
        if avgdl == 0.0 {
            return 0.0;
        }

        let k1 = self.config.k1;
        let b = self.config.b;

        let mut score = 0.0;
        for qt in query_terms {
            let tf = tokens.iter().filter(|t| *t == qt).count() as f64;
            if tf == 0.0 {
                continue;
            }
            let idf = self.idf(qt);
            let numerator = tf * (k1 + 1.0);
            let denominator = tf + k1 * (1.0 - b + b * (dl / avgdl));
            score += idf * (numerator / denominator);
        }
        score
    }

    /// Score a single document against a query using all fields with uniform weight.
    pub fn score(&self, doc_id: &str, query: &str) -> f64 {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() {
            return 0.0;
        }

        let entry = match self.documents.get(doc_id) {
            Some(e) => e,
            None => return 0.0,
        };

        let fields: Vec<String> = entry.field_tokens.keys().cloned().collect();
        let mut total = 0.0;
        for field in &fields {
            total += self.score_field(doc_id, &query_terms, field);
        }
        total
    }

    /// Score a document with field weights.
    pub fn score_weighted(&self, doc_id: &str, query: &str, weights: &[FieldWeight]) -> f64 {
        let query_terms = Self::tokenize(query);
        if query_terms.is_empty() {
            return 0.0;
        }

        let mut total = 0.0;
        for fw in weights {
            total += self.score_field(doc_id, &query_terms, &fw.field) * fw.weight;
        }
        total
    }

    /// Search the corpus with a query, returning scored results sorted by relevance.
    pub fn search(&self, query: &str) -> Vec<Bm25Result> {
        let doc_ids: Vec<String> = self.documents.keys().cloned().collect();
        let mut results: Vec<Bm25Result> = doc_ids
            .iter()
            .map(|doc_id| Bm25Result {
                doc_id: doc_id.clone(),
                score: self.score(doc_id, query),
            })
            .filter(|r| r.score > 0.0)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Search with field weights.
    pub fn search_weighted(&self, query: &str, weights: &[FieldWeight]) -> Vec<Bm25Result> {
        let doc_ids: Vec<String> = self.documents.keys().cloned().collect();
        let mut results: Vec<Bm25Result> = doc_ids
            .iter()
            .map(|doc_id| Bm25Result {
                doc_id: doc_id.clone(),
                score: self.score_weighted(doc_id, query, weights),
            })
            .filter(|r| r.score > 0.0)
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Batch score: score multiple queries against the corpus.
    pub fn batch_search(&self, queries: &[&str]) -> Vec<Vec<Bm25Result>> {
        queries.iter().map(|q| self.search(q)).collect()
    }

    /// Relevance feedback: extract top expansion terms from relevant documents.
    pub fn expansion_terms(&self, feedback: &RelevanceFeedback) -> Vec<String> {
        // Collect term frequencies across relevant docs.
        let mut term_freq: HashMap<String, usize> = HashMap::new();
        for doc_id in &feedback.relevant_doc_ids {
            if let Some(entry) = self.documents.get(doc_id) {
                for tokens in entry.field_tokens.values() {
                    for token in tokens {
                        *term_freq.entry(token.clone()).or_insert(0) += 1;
                    }
                }
            }
        }

        // Sort by frequency descending.
        let mut sorted: Vec<(String, usize)> = term_freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        sorted
            .into_iter()
            .take(feedback.num_expansion_terms)
            .map(|(term, _)| term)
            .collect()
    }

    /// Search with relevance feedback: expand the query with terms from
    /// known-relevant documents then re-score.
    pub fn search_with_feedback(
        &self,
        query: &str,
        feedback: &RelevanceFeedback,
    ) -> Vec<Bm25Result> {
        let expansion = self.expansion_terms(feedback);
        let expanded_query = format!("{} {}", query, expansion.join(" "));
        self.search(&expanded_query)
    }

    /// Get the BM25 config.
    pub fn config(&self) -> &Bm25Config {
        &self.config
    }
}

impl Default for Bm25Scorer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn build_scorer() -> Bm25Scorer {
        let mut s = Bm25Scorer::new();
        s.add_document("d1", "the quick brown fox jumps over the lazy dog");
        s.add_document("d2", "the quick brown cat sits on the mat");
        s.add_document("d3", "a fox and a dog are friends");
        s.add_document("d4", "dogs are loyal animals that love humans");
        s
    }

    #[test]
    fn test_new_scorer() {
        let s = Bm25Scorer::new();
        assert_eq!(s.num_documents(), 0);
    }

    #[test]
    fn test_default_config() {
        let c = Bm25Config::default();
        assert!((c.k1 - 1.2).abs() < 0.001);
        assert!((c.b - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_add_document() {
        let s = build_scorer();
        assert_eq!(s.num_documents(), 4);
    }

    #[test]
    fn test_remove_document() {
        let mut s = build_scorer();
        assert!(s.remove_document("d1"));
        assert_eq!(s.num_documents(), 3);
        assert!(!s.remove_document("d1")); // already removed
    }

    #[test]
    fn test_document_frequency() {
        let s = build_scorer();
        // "the" appears in d1, d2 (at least)
        assert!(s.document_frequency("the") >= 2);
        // "fox" appears in d1, d3
        assert_eq!(s.document_frequency("fox"), 2);
    }

    #[test]
    fn test_score_positive() {
        let s = build_scorer();
        let score = s.score("d1", "fox");
        assert!(score > 0.0, "score should be positive for matching doc");
    }

    #[test]
    fn test_score_zero_no_match() {
        let s = build_scorer();
        let score = s.score("d4", "fox");
        assert!(
            score == 0.0 || score.abs() < 0.001,
            "score should be ~0 for non-matching doc"
        );
    }

    #[test]
    fn test_search_ranking() {
        let s = build_scorer();
        let results = s.search("fox dog");
        assert!(!results.is_empty());
        // d1 has both "fox" and "dog" so should be top or near-top
        let d1_pos = results.iter().position(|r| r.doc_id == "d1");
        assert!(d1_pos.is_some());
    }

    #[test]
    fn test_search_no_results() {
        let s = build_scorer();
        let results = s.search("xylophone");
        assert!(results.is_empty());
    }

    #[test]
    fn test_multi_field() {
        let mut s = Bm25Scorer::new();
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "rust programming".to_string());
        fields.insert("body".to_string(), "a book about the rust programming language".to_string());
        s.add_document_fields("d1", &fields);

        let score = s.score("d1", "rust");
        assert!(score > 0.0);
    }

    #[test]
    fn test_weighted_scoring() {
        let mut s = Bm25Scorer::new();
        let mut fields = HashMap::new();
        fields.insert("title".to_string(), "rust programming".to_string());
        fields.insert("body".to_string(), "a book about the rust programming language".to_string());
        s.add_document_fields("d1", &fields);

        let weights = vec![
            FieldWeight { field: "title".to_string(), weight: 3.0 },
            FieldWeight { field: "body".to_string(), weight: 1.0 },
        ];
        let weighted = s.score_weighted("d1", "rust", &weights);
        let uniform = s.score("d1", "rust");
        // Weighted with title=3x should differ from uniform
        assert!((weighted - uniform).abs() > 0.001);
    }

    #[test]
    fn test_batch_search() {
        let s = build_scorer();
        let results = s.batch_search(&["fox", "cat"]);
        assert_eq!(results.len(), 2);
        // First query should find fox documents
        assert!(!results[0].is_empty());
    }

    #[test]
    fn test_idf_rare_term_higher() {
        let s = build_scorer();
        // "cat" appears in 1 doc, "the" in 2+ docs -> cat should have higher IDF
        let idf_cat = s.idf("cat");
        let idf_the = s.idf("the");
        assert!(idf_cat > idf_the, "rarer term should have higher IDF");
    }

    #[test]
    fn test_avg_field_length() {
        let s = build_scorer();
        let avg = s.avg_field_length("_default");
        assert!(avg > 0.0);
    }

    #[test]
    fn test_expansion_terms() {
        let s = build_scorer();
        let feedback = RelevanceFeedback {
            relevant_doc_ids: vec!["d1".to_string()],
            num_expansion_terms: 3,
        };
        let terms = s.expansion_terms(&feedback);
        assert!(!terms.is_empty());
        assert!(terms.len() <= 3);
    }

    #[test]
    fn test_search_with_feedback() {
        let s = build_scorer();
        let feedback = RelevanceFeedback {
            relevant_doc_ids: vec!["d1".to_string()],
            num_expansion_terms: 3,
        };
        let results = s.search_with_feedback("fox", &feedback);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_custom_config() {
        let config = Bm25Config { k1: 2.0, b: 0.5 };
        let mut s = Bm25Scorer::with_config(config);
        s.add_document("d1", "hello world");
        assert!((s.config().k1 - 2.0).abs() < 0.001);
        assert!((s.config().b - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_empty_query_score() {
        let s = build_scorer();
        let score = s.score("d1", "");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_nonexistent_doc_score() {
        let s = build_scorer();
        let score = s.score("d999", "fox");
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_search_weighted() {
        let mut s = Bm25Scorer::new();
        let mut f1 = HashMap::new();
        f1.insert("title".to_string(), "rust web framework".to_string());
        f1.insert("body".to_string(), "build web apps with rust".to_string());
        s.add_document_fields("d1", &f1);

        let mut f2 = HashMap::new();
        f2.insert("title".to_string(), "python data science".to_string());
        f2.insert("body".to_string(), "rust is mentioned once".to_string());
        s.add_document_fields("d2", &f2);

        let weights = vec![
            FieldWeight { field: "title".to_string(), weight: 5.0 },
            FieldWeight { field: "body".to_string(), weight: 1.0 },
        ];
        let results = s.search_weighted("rust", &weights);
        assert!(!results.is_empty());
        // d1 has "rust" in title, should rank higher with title weight=5
        assert_eq!(results[0].doc_id, "d1");
    }

    #[test]
    fn test_default_trait() {
        let s = Bm25Scorer::default();
        assert_eq!(s.num_documents(), 0);
    }
}
