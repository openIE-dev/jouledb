// search_index.rs — Inverted search index: term->doc_ids mapping,
// TF-IDF scoring, boolean queries (AND/OR/NOT), phrase search
// with position tracking, faceted counts, field boosting.

use std::collections::HashMap;

/// A posting: a document that contains a given term, with positions.
#[derive(Debug, Clone)]
struct Posting {
    doc_id: u64,
    positions: Vec<u32>,
    field: String,
}

/// The inverted index.
#[derive(Debug, Clone, Default)]
pub struct SearchIndex {
    /// term -> list of postings
    postings: HashMap<String, Vec<Posting>>,
    /// doc_id -> total token count (for TF normalization)
    doc_lengths: HashMap<u64, u32>,
    /// doc_id -> facet values (facet_name -> value)
    doc_facets: HashMap<u64, HashMap<String, String>>,
    /// field -> boost factor
    field_boosts: HashMap<String, f64>,
    /// Total number of indexed documents.
    doc_count: u64,
}

impl SearchIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a boost factor for a field.
    pub fn set_field_boost(&mut self, field: &str, boost: f64) {
        self.field_boosts.insert(field.to_string(), boost);
    }

    fn boost_for(&self, field: &str) -> f64 {
        self.field_boosts.get(field).copied().unwrap_or(1.0)
    }

    /// Index a document. `tokens` are (term, position) pairs from a single field.
    pub fn index_document(
        &mut self,
        doc_id: u64,
        field: &str,
        tokens: &[(String, u32)],
    ) {
        // Build per-term postings.
        let mut term_positions: HashMap<String, Vec<u32>> = HashMap::new();
        for (term, pos) in tokens {
            term_positions
                .entry(term.to_lowercase())
                .or_default()
                .push(*pos);
        }

        for (term, positions) in term_positions {
            let list = self.postings.entry(term).or_default();
            list.push(Posting {
                doc_id,
                positions,
                field: field.to_string(),
            });
        }

        let is_new = !self.doc_lengths.contains_key(&doc_id);
        let count = self.doc_lengths.entry(doc_id).or_insert(0);
        *count += tokens.len() as u32;

        if is_new {
            self.doc_count += 1;
        }
    }

    /// Convenience: index raw text (whitespace tokenizer).
    pub fn index_text(&mut self, doc_id: u64, field: &str, text: &str) {
        let tokens: Vec<(String, u32)> = text
            .split_whitespace()
            .enumerate()
            .map(|(i, w)| (w.to_string(), i as u32))
            .collect();
        self.index_document(doc_id, field, &tokens);
    }

    /// Set a facet value for a document.
    pub fn set_facet(&mut self, doc_id: u64, facet: &str, value: &str) {
        self.doc_facets
            .entry(doc_id)
            .or_default()
            .insert(facet.to_string(), value.to_string());
    }

    pub fn total_documents(&self) -> u64 {
        self.doc_count
    }

    /// Term frequency: how many times `term` appears in doc `doc_id`.
    pub fn term_frequency(&self, term: &str, doc_id: u64) -> u32 {
        let lc = term.to_lowercase();
        match self.postings.get(&lc) {
            None => 0,
            Some(list) => list
                .iter()
                .filter(|p| p.doc_id == doc_id)
                .map(|p| p.positions.len() as u32)
                .sum(),
        }
    }

    /// Document frequency: how many unique documents contain `term`.
    pub fn document_frequency(&self, term: &str) -> u64 {
        let lc = term.to_lowercase();
        match self.postings.get(&lc) {
            None => 0,
            Some(list) => {
                let mut seen = Vec::new();
                for p in list {
                    if !seen.contains(&p.doc_id) {
                        seen.push(p.doc_id);
                    }
                }
                seen.len() as u64
            }
        }
    }

    /// IDF: log(N / df) where N = total docs.
    pub fn idf(&self, term: &str) -> f64 {
        let df = self.document_frequency(term);
        if df == 0 || self.doc_count == 0 {
            return 0.0;
        }
        (1.0 + self.doc_count as f64 / df as f64).ln()
    }

    /// TF-IDF score for a term in a document.
    pub fn tfidf(&self, term: &str, doc_id: u64) -> f64 {
        let tf = self.term_frequency(term, doc_id) as f64;
        let doc_len = self.doc_lengths.get(&doc_id).copied().unwrap_or(1) as f64;
        let normalized_tf = tf / doc_len;
        normalized_tf * self.idf(term)
    }

    /// TF-IDF with field boosting.
    pub fn tfidf_boosted(&self, term: &str, doc_id: u64) -> f64 {
        let lc = term.to_lowercase();
        let base = self.tfidf(term, doc_id);
        // Find the max boost across fields containing this term for this doc.
        let max_boost = self
            .postings
            .get(&lc)
            .map(|list| {
                list.iter()
                    .filter(|p| p.doc_id == doc_id)
                    .map(|p| self.boost_for(&p.field))
                    .fold(1.0f64, f64::max)
            })
            .unwrap_or(1.0);
        base * max_boost
    }

    /// Docs containing the given term.
    pub fn search_term(&self, term: &str) -> Vec<u64> {
        let lc = term.to_lowercase();
        match self.postings.get(&lc) {
            None => Vec::new(),
            Some(list) => {
                let mut docs: Vec<u64> = list.iter().map(|p| p.doc_id).collect();
                docs.sort();
                docs.dedup();
                docs
            }
        }
    }

    // ---- Boolean queries ----

    /// AND: docs that contain ALL terms.
    pub fn search_and(&self, terms: &[&str]) -> Vec<u64> {
        if terms.is_empty() {
            return Vec::new();
        }
        let mut result = self.search_term(terms[0]);
        for term in &terms[1..] {
            let set = self.search_term(term);
            result.retain(|id| set.contains(id));
        }
        result
    }

    /// OR: docs that contain ANY term.
    pub fn search_or(&self, terms: &[&str]) -> Vec<u64> {
        let mut result = Vec::new();
        for term in terms {
            for id in self.search_term(term) {
                if !result.contains(&id) {
                    result.push(id);
                }
            }
        }
        result.sort();
        result
    }

    /// NOT: docs containing `include` but NOT `exclude`.
    pub fn search_not(&self, include: &str, exclude: &str) -> Vec<u64> {
        let inc = self.search_term(include);
        let exc = self.search_term(exclude);
        inc.into_iter().filter(|id| !exc.contains(id)).collect()
    }

    // ---- Phrase search ----

    /// Search for an exact phrase (consecutive positions in the same field).
    pub fn search_phrase(&self, phrase_terms: &[&str]) -> Vec<u64> {
        if phrase_terms.is_empty() {
            return Vec::new();
        }
        if phrase_terms.len() == 1 {
            return self.search_term(phrase_terms[0]);
        }

        // Collect posting lists for each term.
        let term_postings: Vec<Option<&Vec<Posting>>> = phrase_terms
            .iter()
            .map(|t| self.postings.get(&t.to_lowercase()))
            .collect();

        // All terms must exist.
        if term_postings.iter().any(|p| p.is_none()) {
            return Vec::new();
        }

        let first_postings = term_postings[0].unwrap();
        let mut results = Vec::new();

        for posting in first_postings {
            let doc_id = posting.doc_id;
            let field = &posting.field;

            'next_start_pos: for &start_pos in &posting.positions {
                // Check each subsequent term has the next position in the same field.
                for (offset, term_post_list) in term_postings.iter().enumerate().skip(1) {
                    let expected_pos = start_pos + offset as u32;
                    let list = term_post_list.unwrap();
                    let found = list.iter().any(|p| {
                        p.doc_id == doc_id
                            && p.field == *field
                            && p.positions.contains(&expected_pos)
                    });
                    if !found {
                        continue 'next_start_pos;
                    }
                }
                if !results.contains(&doc_id) {
                    results.push(doc_id);
                }
            }
        }

        results
    }

    // ---- Faceted counts ----

    /// Compute facet value counts among a set of document IDs.
    pub fn facet_counts(&self, facet: &str, doc_ids: &[u64]) -> Vec<(String, usize)> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for &id in doc_ids {
            if let Some(facets) = self.doc_facets.get(&id) {
                if let Some(val) = facets.get(facet) {
                    *counts.entry(val.clone()).or_insert(0) += 1;
                }
            }
        }
        let mut result: Vec<(String, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        result
    }

    /// Ranked search: score every matching doc by summed TF-IDF, return sorted desc.
    pub fn ranked_search(&self, terms: &[&str]) -> Vec<(u64, f64)> {
        let doc_ids = self.search_or(terms);
        let mut scored: Vec<(u64, f64)> = doc_ids
            .iter()
            .map(|id| {
                let score: f64 = terms.iter().map(|t| self.tfidf_boosted(t, *id)).sum();
                (*id, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

/// Whitespace tokenizer producing (term, position) pairs.
pub fn tokenize(text: &str) -> Vec<(String, u32)> {
    text.split_whitespace()
        .enumerate()
        .map(|(i, w)| (w.to_lowercase(), i as u32))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_index() -> SearchIndex {
        let mut idx = SearchIndex::new();
        idx.index_text(1, "title", "the quick brown fox");
        idx.index_text(2, "title", "the slow brown dog");
        idx.index_text(3, "title", "quick fox jumps over lazy dog");
        idx.set_facet(1, "category", "animals");
        idx.set_facet(2, "category", "animals");
        idx.set_facet(3, "category", "action");
        idx
    }

    #[test]
    fn test_total_documents() {
        let idx = sample_index();
        assert_eq!(idx.total_documents(), 3);
    }

    #[test]
    fn test_term_frequency() {
        let idx = sample_index();
        assert_eq!(idx.term_frequency("the", 1), 1);
        assert_eq!(idx.term_frequency("the", 2), 1);
        assert_eq!(idx.term_frequency("the", 3), 0);
    }

    #[test]
    fn test_document_frequency() {
        let idx = sample_index();
        assert_eq!(idx.document_frequency("the"), 2);
        assert_eq!(idx.document_frequency("quick"), 2);
        assert_eq!(idx.document_frequency("brown"), 2);
        assert_eq!(idx.document_frequency("jumps"), 1);
        assert_eq!(idx.document_frequency("nonexistent"), 0);
    }

    #[test]
    fn test_idf() {
        let idx = sample_index();
        // "jumps" appears in 1 of 3 docs -> idf = ln(1 + 3/1) = ln(4)
        let idf = idx.idf("jumps");
        assert!((idf - (4.0f64).ln()).abs() < 0.001);
        assert!((idx.idf("nonexistent") - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_tfidf() {
        let idx = sample_index();
        let score = idx.tfidf("jumps", 3);
        assert!(score > 0.0);
        // "jumps" not in doc 1 -> score 0
        assert!((idx.tfidf("jumps", 1) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_search_term() {
        let idx = sample_index();
        let docs = idx.search_term("quick");
        assert!(docs.contains(&1));
        assert!(docs.contains(&3));
        assert!(!docs.contains(&2));
    }

    #[test]
    fn test_search_and() {
        let idx = sample_index();
        let docs = idx.search_and(&["brown", "the"]);
        assert!(docs.contains(&1));
        assert!(docs.contains(&2));
        assert!(!docs.contains(&3));
    }

    #[test]
    fn test_search_or() {
        let idx = sample_index();
        let docs = idx.search_or(&["jumps", "slow"]);
        assert!(docs.contains(&2));
        assert!(docs.contains(&3));
    }

    #[test]
    fn test_search_not() {
        let idx = sample_index();
        let docs = idx.search_not("the", "quick");
        // doc 1 has both "the" and "quick" -> excluded
        // doc 2 has "the" but not "quick" -> included
        assert_eq!(docs, vec![2]);
    }

    #[test]
    fn test_search_and_empty() {
        let idx = sample_index();
        assert!(idx.search_and(&[]).is_empty());
    }

    #[test]
    fn test_phrase_search() {
        let idx = sample_index();
        // "quick brown" appears in doc 1 as positions 1,2
        let docs = idx.search_phrase(&["quick", "brown"]);
        assert!(docs.contains(&1));
        // "brown fox" appears in doc 1 as positions 2,3
        let docs2 = idx.search_phrase(&["brown", "fox"]);
        assert!(docs2.contains(&1));
        // "fox quick" is NOT a consecutive phrase in any doc
        let docs3 = idx.search_phrase(&["fox", "quick"]);
        assert!(docs3.is_empty());
    }

    #[test]
    fn test_phrase_single_term() {
        let idx = sample_index();
        let docs = idx.search_phrase(&["jumps"]);
        assert_eq!(docs, vec![3]);
    }

    #[test]
    fn test_phrase_empty() {
        let idx = sample_index();
        assert!(idx.search_phrase(&[]).is_empty());
    }

    #[test]
    fn test_facet_counts() {
        let idx = sample_index();
        let all_docs = vec![1, 2, 3];
        let counts = idx.facet_counts("category", &all_docs);
        // animals: 2, action: 1
        assert_eq!(counts.len(), 2);
        assert_eq!(counts[0].0, "animals");
        assert_eq!(counts[0].1, 2);
        assert_eq!(counts[1].0, "action");
        assert_eq!(counts[1].1, 1);
    }

    #[test]
    fn test_facet_counts_subset() {
        let idx = sample_index();
        let counts = idx.facet_counts("category", &[3]);
        assert_eq!(counts.len(), 1);
        assert_eq!(counts[0].0, "action");
    }

    #[test]
    fn test_field_boosting() {
        let mut idx = SearchIndex::new();
        idx.set_field_boost("title", 2.0);
        idx.set_field_boost("body", 1.0);
        idx.index_text(1, "title", "rust programming");
        idx.index_text(2, "body", "rust programming");

        let score1 = idx.tfidf_boosted("rust", 1);
        let score2 = idx.tfidf_boosted("rust", 2);
        // Doc 1 (title, boost=2) should score higher than doc 2 (body, boost=1).
        assert!(score1 > score2, "title boost should increase score");
    }

    #[test]
    fn test_ranked_search() {
        let idx = sample_index();
        let results = idx.ranked_search(&["quick", "fox"]);
        // Doc 1 and 3 have both terms, doc 1 has "the quick brown fox".
        assert!(!results.is_empty());
        // All scores should be non-negative.
        for (_, score) in &results {
            assert!(*score >= 0.0);
        }
    }

    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello World test");
        assert_eq!(tokens.len(), 3);
        assert_eq!(tokens[0], ("hello".to_string(), 0));
        assert_eq!(tokens[1], ("world".to_string(), 1));
        assert_eq!(tokens[2], ("test".to_string(), 2));
    }

    #[test]
    fn test_case_insensitive() {
        let mut idx = SearchIndex::new();
        idx.index_text(1, "body", "Hello WORLD");
        let docs = idx.search_term("hello");
        assert!(docs.contains(&1));
        let docs2 = idx.search_term("WORLD");
        assert!(docs2.contains(&1));
    }

    #[test]
    fn test_multi_field_same_doc() {
        let mut idx = SearchIndex::new();
        idx.index_text(1, "title", "rust lang");
        idx.index_text(1, "body", "systems programming in rust");
        // "rust" appears in both fields for doc 1.
        assert_eq!(idx.term_frequency("rust", 1), 2);
        assert_eq!(idx.document_frequency("rust"), 1);
    }

    #[test]
    fn test_empty_index() {
        let idx = SearchIndex::new();
        assert_eq!(idx.total_documents(), 0);
        assert!(idx.search_term("anything").is_empty());
        assert!((idx.idf("anything") - 0.0).abs() < f64::EPSILON);
    }
}
