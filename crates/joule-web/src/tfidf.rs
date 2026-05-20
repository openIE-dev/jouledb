//! TF-IDF computation: term frequency, inverse document frequency, TF-IDF
//! matrix, document similarity, top terms, and feature extraction.

use std::collections::{HashMap, HashSet};

// ── Term frequency variants ──────────────────────────────────────

/// How to compute term frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TfMode {
    /// Raw count of term in document.
    Raw,
    /// 1 + log(count) if count > 0, else 0.
    Log,
    /// 1 if term present, 0 otherwise.
    Boolean,
    /// count / max_count_in_doc.
    Augmented,
}

/// How to compute inverse document frequency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdfMode {
    /// log(N / df) where df = # docs containing term.
    Standard,
    /// log(1 + N / df).
    Smooth,
    /// log((N - df) / df) (probabilistic).
    Probabilistic,
}

// ── Helper: tokenize for TF-IDF ─────────────────────────────────

fn tokenize_simple(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ── Term frequency ───────────────────────────────────────────────

/// Compute raw term counts for a document.
pub fn term_counts(text: &str) -> HashMap<String, usize> {
    let mut counts = HashMap::new();
    for token in tokenize_simple(text) {
        *counts.entry(token).or_insert(0) += 1;
    }
    counts
}

/// Compute term frequency for a document with the given mode.
pub fn compute_tf(text: &str, mode: TfMode) -> HashMap<String, f64> {
    let counts = term_counts(text);
    let max_count = counts.values().copied().max().unwrap_or(1) as f64;

    counts
        .into_iter()
        .map(|(term, count)| {
            let tf = match mode {
                TfMode::Raw => count as f64,
                TfMode::Log => {
                    if count > 0 {
                        1.0 + (count as f64).ln()
                    } else {
                        0.0
                    }
                }
                TfMode::Boolean => {
                    if count > 0 { 1.0 } else { 0.0 }
                }
                TfMode::Augmented => 0.5 + 0.5 * (count as f64 / max_count),
            };
            (term, tf)
        })
        .collect()
}

// ── Inverse document frequency ───────────────────────────────────

/// Compute IDF from a collection of documents.
pub fn compute_idf(documents: &[&str], mode: IdfMode) -> HashMap<String, f64> {
    let n = documents.len() as f64;
    let mut doc_freq: HashMap<String, usize> = HashMap::new();

    for doc in documents {
        let unique_terms: HashSet<String> = tokenize_simple(doc).into_iter().collect();
        for term in unique_terms {
            *doc_freq.entry(term).or_insert(0) += 1;
        }
    }

    doc_freq
        .into_iter()
        .map(|(term, df)| {
            let idf = match mode {
                IdfMode::Standard => {
                    if df > 0 {
                        (n / df as f64).ln()
                    } else {
                        0.0
                    }
                }
                IdfMode::Smooth => (1.0 + n / df as f64).ln(),
                IdfMode::Probabilistic => {
                    let df_f = df as f64;
                    if df_f < n {
                        ((n - df_f) / df_f).ln().max(0.0)
                    } else {
                        0.0
                    }
                }
            };
            (term, idf)
        })
        .collect()
}

// ── TF-IDF matrix ────────────────────────────────────────────────

/// A TF-IDF matrix over a corpus.
#[derive(Debug, Clone)]
pub struct TfidfMatrix {
    /// Vocabulary (sorted for deterministic ordering).
    pub vocabulary: Vec<String>,
    /// term → index in vocabulary.
    pub term_index: HashMap<String, usize>,
    /// IDF values for each term in vocabulary order.
    pub idf: Vec<f64>,
    /// Per-document TF-IDF vectors (row = document, column = term).
    pub vectors: Vec<Vec<f64>>,
}

impl TfidfMatrix {
    /// Build a TF-IDF matrix from a corpus.
    pub fn build(documents: &[&str], tf_mode: TfMode, idf_mode: IdfMode) -> Self {
        let idf_map = compute_idf(documents, idf_mode);

        // Build sorted vocabulary.
        let mut vocabulary: Vec<String> = idf_map.keys().cloned().collect();
        vocabulary.sort();
        let term_index: HashMap<String, usize> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        let idf: Vec<f64> = vocabulary.iter().map(|t| idf_map[t]).collect();

        let vectors: Vec<Vec<f64>> = documents
            .iter()
            .map(|doc| {
                let tf = compute_tf(doc, tf_mode);
                let mut vec = vec![0.0; vocabulary.len()];
                for (term, tf_val) in &tf {
                    if let Some(idx) = term_index.get(term) {
                        vec[*idx] = tf_val * idf[*idx];
                    }
                }
                vec
            })
            .collect();

        Self {
            vocabulary,
            term_index,
            idf,
            vectors,
        }
    }

    /// Number of documents.
    pub fn num_docs(&self) -> usize {
        self.vectors.len()
    }

    /// Number of terms (features).
    pub fn num_terms(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get the TF-IDF vector for a specific document.
    pub fn doc_vector(&self, doc_idx: usize) -> Option<&[f64]> {
        self.vectors.get(doc_idx).map(|v| v.as_slice())
    }

    /// Get TF-IDF value for a specific document and term.
    pub fn get(&self, doc_idx: usize, term: &str) -> f64 {
        if let Some(term_idx) = self.term_index.get(term) {
            if let Some(vec) = self.vectors.get(doc_idx) {
                return vec[*term_idx];
            }
        }
        0.0
    }

    /// Top N terms for a document by TF-IDF score.
    pub fn top_terms(&self, doc_idx: usize, n: usize) -> Vec<(String, f64)> {
        let vec = match self.vectors.get(doc_idx) {
            Some(v) => v,
            None => return Vec::new(),
        };

        let mut scored: Vec<(String, f64)> = self
            .vocabulary
            .iter()
            .zip(vec.iter())
            .filter(|(_, score)| **score > 0.0)
            .map(|(term, score)| (term.clone(), *score))
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(n);
        scored
    }

    /// Transform a new document into a TF-IDF vector using this matrix's IDF.
    pub fn transform(&self, text: &str, tf_mode: TfMode) -> Vec<f64> {
        let tf = compute_tf(text, tf_mode);
        let mut vec = vec![0.0; self.vocabulary.len()];
        for (term, tf_val) in &tf {
            if let Some(idx) = self.term_index.get(term) {
                vec[*idx] = tf_val * self.idf[*idx];
            }
        }
        vec
    }
}

// ── Cosine similarity ────────────────────────────────────────────

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute cosine similarity between two documents in a TF-IDF matrix.
pub fn document_similarity(matrix: &TfidfMatrix, doc_a: usize, doc_b: usize) -> f64 {
    match (matrix.doc_vector(doc_a), matrix.doc_vector(doc_b)) {
        (Some(a), Some(b)) => cosine_similarity(a, b),
        _ => 0.0,
    }
}

/// Compute pairwise similarity matrix for all documents.
pub fn pairwise_similarity(matrix: &TfidfMatrix) -> Vec<Vec<f64>> {
    let n = matrix.num_docs();
    let mut sim = vec![vec![0.0; n]; n];
    for i in 0..n {
        sim[i][i] = 1.0;
        for j in (i + 1)..n {
            let s = document_similarity(matrix, i, j);
            sim[i][j] = s;
            sim[j][i] = s;
        }
    }
    sim
}

// ── Feature extraction ───────────────────────────────────────────

/// Extract the top N features (terms) across the entire corpus by average TF-IDF.
pub fn top_features(matrix: &TfidfMatrix, n: usize) -> Vec<(String, f64)> {
    let num_docs = matrix.num_docs() as f64;
    if num_docs == 0.0 {
        return Vec::new();
    }

    let mut avg_scores: Vec<(String, f64)> = matrix
        .vocabulary
        .iter()
        .enumerate()
        .map(|(idx, term)| {
            let sum: f64 = matrix.vectors.iter().map(|v| v[idx]).sum();
            (term.clone(), sum / num_docs)
        })
        .filter(|(_, avg)| *avg > 0.0)
        .collect();

    avg_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    avg_scores.truncate(n);
    avg_scores
}

/// Extract keywords from a single document (shorthand for top_terms on a
/// freshly built single-doc matrix).
pub fn extract_keywords(text: &str, corpus: &[&str], n: usize) -> Vec<(String, f64)> {
    let idf_map = compute_idf(corpus, IdfMode::Smooth);
    let tf = compute_tf(text, TfMode::Log);

    let mut scored: Vec<(String, f64)> = tf
        .into_iter()
        .map(|(term, tf_val)| {
            let idf_val = idf_map.get(&term).copied().unwrap_or(0.0);
            (term, tf_val * idf_val)
        })
        .filter(|(_, score)| *score > 0.0)
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_term_counts() {
        let counts = term_counts("the cat sat on the mat");
        assert_eq!(counts["the"], 2);
        assert_eq!(counts["cat"], 1);
    }

    #[test]
    fn test_tf_raw() {
        let tf = compute_tf("hello hello world", TfMode::Raw);
        assert_eq!(tf["hello"], 2.0);
        assert_eq!(tf["world"], 1.0);
    }

    #[test]
    fn test_tf_boolean() {
        let tf = compute_tf("hello hello world", TfMode::Boolean);
        assert_eq!(tf["hello"], 1.0);
        assert_eq!(tf["world"], 1.0);
    }

    #[test]
    fn test_tf_log() {
        let tf = compute_tf("hello hello world", TfMode::Log);
        // 1 + ln(2) ≈ 1.693
        assert!((tf["hello"] - (1.0 + 2.0_f64.ln())).abs() < 0.001);
        assert!((tf["world"] - 1.0).abs() < 0.001); // 1 + ln(1)
    }

    #[test]
    fn test_tf_augmented() {
        let tf = compute_tf("hello hello world", TfMode::Augmented);
        // hello: 0.5 + 0.5*(2/2) = 1.0
        assert!((tf["hello"] - 1.0).abs() < 0.001);
        // world: 0.5 + 0.5*(1/2) = 0.75
        assert!((tf["world"] - 0.75).abs() < 0.001);
    }

    #[test]
    fn test_idf_standard() {
        let docs = &["cat dog", "cat fish", "dog bird"];
        let idf = compute_idf(docs, IdfMode::Standard);
        // cat appears in 2/3 docs → ln(3/2) ≈ 0.405
        assert!((idf["cat"] - (3.0_f64 / 2.0).ln()).abs() < 0.001);
        // bird appears in 1/3 docs → ln(3/1) ≈ 1.099
        assert!((idf["bird"] - 3.0_f64.ln()).abs() < 0.001);
    }

    #[test]
    fn test_idf_smooth() {
        let docs = &["cat dog", "cat fish"];
        let idf = compute_idf(docs, IdfMode::Smooth);
        // cat: ln(1 + 2/2) = ln(2) ≈ 0.693
        assert!((idf["cat"] - 2.0_f64.ln()).abs() < 0.001);
    }

    #[test]
    fn test_tfidf_matrix_build() {
        let docs = &["cat sat mat", "dog ran fast", "cat dog friend"];
        let matrix = TfidfMatrix::build(docs, TfMode::Raw, IdfMode::Standard);
        assert_eq!(matrix.num_docs(), 3);
        assert!(matrix.num_terms() > 0);
    }

    #[test]
    fn test_tfidf_matrix_get() {
        let docs = &["hello world", "hello rust"];
        let matrix = TfidfMatrix::build(docs, TfMode::Raw, IdfMode::Standard);
        // "world" only in doc 0, "rust" only in doc 1 → nonzero
        assert!(matrix.get(0, "world") > 0.0);
        assert!(matrix.get(1, "rust") > 0.0);
        // "hello" in both docs with standard IDF → ln(2/2) = 0
        assert!((matrix.get(0, "hello")).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.001);
    }

    #[test]
    fn test_document_similarity() {
        let docs = &["cat dog pet", "cat dog animal", "math science physics"];
        let matrix = TfidfMatrix::build(docs, TfMode::Raw, IdfMode::Smooth);
        let sim_01 = document_similarity(&matrix, 0, 1);
        let sim_02 = document_similarity(&matrix, 0, 2);
        // docs 0 and 1 share more terms than 0 and 2
        assert!(sim_01 > sim_02);
    }

    #[test]
    fn test_top_terms() {
        let docs = &["alpha beta", "gamma delta", "alpha gamma epsilon"];
        let matrix = TfidfMatrix::build(docs, TfMode::Raw, IdfMode::Smooth);
        let top = matrix.top_terms(2, 3);
        assert!(!top.is_empty());
        // All top terms should have positive scores
        for (_, score) in &top {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_transform() {
        let docs = &["hello world", "hello rust"];
        let matrix = TfidfMatrix::build(docs, TfMode::Raw, IdfMode::Standard);
        let vec = matrix.transform("hello world", TfMode::Raw);
        assert_eq!(vec.len(), matrix.num_terms());
    }

    #[test]
    fn test_pairwise_similarity() {
        let docs = &["cat", "dog", "cat dog"];
        let matrix = TfidfMatrix::build(docs, TfMode::Raw, IdfMode::Smooth);
        let sim = pairwise_similarity(&matrix);
        assert_eq!(sim.len(), 3);
        // Diagonal should be 1.0
        for i in 0..3 {
            assert!((sim[i][i] - 1.0).abs() < 0.001);
        }
        // Symmetric
        assert!((sim[0][1] - sim[1][0]).abs() < 0.001);
    }

    #[test]
    fn test_extract_keywords() {
        let corpus = &[
            "machine learning algorithms",
            "deep learning neural networks",
            "cooking recipes dinner",
        ];
        let keywords = extract_keywords("machine learning is powerful algorithms", corpus, 3);
        assert!(!keywords.is_empty());
        assert!(keywords.len() <= 3);
    }

    #[test]
    fn test_top_features() {
        let docs = &["unique rare special", "common common common", "unique common"];
        let matrix = TfidfMatrix::build(docs, TfMode::Raw, IdfMode::Smooth);
        let features = top_features(&matrix, 5);
        assert!(!features.is_empty());
    }

    #[test]
    fn test_empty_document() {
        let tf = compute_tf("", TfMode::Raw);
        assert!(tf.is_empty());
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }
}
