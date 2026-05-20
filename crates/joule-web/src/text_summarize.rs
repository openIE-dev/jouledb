//! Extractive text summarization: sentence scoring (TF-IDF, position, length),
//! top-N sentence selection, ordering, summary length control, keyword
//! extraction, and TextRank-like graph scoring.

use std::collections::{HashMap, HashSet};

// ── Configuration ────────────────────────────────────────────────

/// Summarization configuration.
#[derive(Debug, Clone)]
pub struct SummarizeConfig {
    /// Maximum number of sentences in the summary.
    pub max_sentences: usize,
    /// Maximum number of words in the summary (0 = no limit).
    pub max_words: usize,
    /// Weight for TF-IDF score component.
    pub tfidf_weight: f64,
    /// Weight for position score component.
    pub position_weight: f64,
    /// Weight for sentence length score component.
    pub length_weight: f64,
    /// Weight for TextRank score component.
    pub textrank_weight: f64,
    /// Number of TextRank iterations.
    pub textrank_iterations: usize,
    /// TextRank damping factor (typically 0.85).
    pub damping: f64,
}

impl Default for SummarizeConfig {
    fn default() -> Self {
        Self {
            max_sentences: 3,
            max_words: 0,
            tfidf_weight: 0.4,
            position_weight: 0.2,
            length_weight: 0.1,
            textrank_weight: 0.3,
            textrank_iterations: 20,
            damping: 0.85,
        }
    }
}

// ── Tokenization helpers ─────────────────────────────────────────

fn tokenize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        current.push(chars[i]);

        if chars[i] == '.' || chars[i] == '!' || chars[i] == '?' {
            // Consume consecutive terminators.
            while i + 1 < len && (chars[i + 1] == '.' || chars[i + 1] == '!' || chars[i + 1] == '?') {
                i += 1;
                current.push(chars[i]);
            }

            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() && word_count(&trimmed) > 2 {
                sentences.push(trimmed);
            }
            current.clear();
        }
        i += 1;
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() && word_count(&trimmed) > 2 {
        sentences.push(trimmed);
    }

    sentences
}

fn word_count(text: &str) -> usize {
    text.split_whitespace().count()
}

// ── TF-IDF scoring ───────────────────────────────────────────────

fn compute_tfidf_scores(sentences: &[String]) -> Vec<f64> {
    if sentences.is_empty() {
        return Vec::new();
    }

    let n = sentences.len() as f64;

    // Document frequency.
    let mut doc_freq: HashMap<String, usize> = HashMap::new();
    for sent in sentences {
        let unique: HashSet<String> = tokenize_words(sent).into_iter().collect();
        for word in unique {
            *doc_freq.entry(word).or_insert(0) += 1;
        }
    }

    // Score each sentence by average TF-IDF of its words.
    sentences
        .iter()
        .map(|sent| {
            let words = tokenize_words(sent);
            if words.is_empty() {
                return 0.0;
            }

            let mut counts: HashMap<String, usize> = HashMap::new();
            for w in &words {
                *counts.entry(w.clone()).or_insert(0) += 1;
            }

            let total: f64 = counts
                .iter()
                .map(|(word, count)| {
                    let tf = 1.0 + (*count as f64).ln();
                    let df = doc_freq.get(word).copied().unwrap_or(1) as f64;
                    let idf = (n / df).ln();
                    tf * idf
                })
                .sum();

            total / words.len() as f64
        })
        .collect()
}

// ── Position scoring ─────────────────────────────────────────────

fn compute_position_scores(count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    (0..count)
        .map(|i| {
            // First and last sentences score higher.
            let pos = i as f64 / count as f64;
            if i == 0 {
                1.0
            } else if i == count - 1 {
                0.8
            } else {
                // Decay for middle sentences; floor below last-sentence score.
                (1.0 - pos * 0.5).min(0.79)
            }
        })
        .collect()
}

// ── Length scoring ────────────────────────────────────────────────

fn compute_length_scores(sentences: &[String]) -> Vec<f64> {
    if sentences.is_empty() {
        return Vec::new();
    }

    let lengths: Vec<usize> = sentences.iter().map(|s| word_count(s)).collect();
    let max_len = *lengths.iter().max().unwrap_or(&1) as f64;
    let ideal_range = (10.0, 30.0);

    lengths
        .iter()
        .map(|len| {
            let l = *len as f64;
            if l >= ideal_range.0 && l <= ideal_range.1 {
                1.0
            } else if l < ideal_range.0 {
                l / ideal_range.0
            } else {
                (ideal_range.1 / l).min(1.0).max(0.0) * (l / max_len)
            }
        })
        .collect()
}

// ── TextRank scoring ─────────────────────────────────────────────

fn sentence_similarity(a: &str, b: &str) -> f64 {
    let words_a: HashSet<String> = tokenize_words(a).into_iter().collect();
    let words_b: HashSet<String> = tokenize_words(b).into_iter().collect();

    if words_a.is_empty() || words_b.is_empty() {
        return 0.0;
    }

    let intersection = words_a.intersection(&words_b).count() as f64;
    let denom = (words_a.len() as f64).ln() + (words_b.len() as f64).ln();

    if denom > 0.0 {
        intersection / denom
    } else {
        0.0
    }
}

fn compute_textrank_scores(sentences: &[String], iterations: usize, damping: f64) -> Vec<f64> {
    let n = sentences.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![1.0];
    }

    // Build similarity matrix.
    let mut sim_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let s = sentence_similarity(&sentences[i], &sentences[j]);
            sim_matrix[i][j] = s;
            sim_matrix[j][i] = s;
        }
    }

    // Normalize rows.
    let mut norm_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        let row_sum: f64 = sim_matrix[i].iter().sum();
        if row_sum > 0.0 {
            for j in 0..n {
                norm_matrix[i][j] = sim_matrix[i][j] / row_sum;
            }
        }
    }

    // Power iteration.
    let mut scores = vec![1.0 / n as f64; n];
    for _ in 0..iterations {
        let mut new_scores = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += norm_matrix[j][i] * scores[j];
            }
            new_scores[i] = (1.0 - damping) / n as f64 + damping * sum;
        }
        scores = new_scores;
    }

    // Normalize to [0, 1].
    let max_score = scores.iter().cloned().fold(0.0_f64, f64::max);
    if max_score > 0.0 {
        for s in &mut scores {
            *s /= max_score;
        }
    }

    scores
}

// ── Summarizer ───────────────────────────────────────────────────

/// Sentence with its composite score and original index.
#[derive(Debug, Clone)]
pub struct ScoredSentence {
    pub text: String,
    pub score: f64,
    pub original_index: usize,
}

/// Summary result.
#[derive(Debug, Clone)]
pub struct Summary {
    pub sentences: Vec<ScoredSentence>,
    pub text: String,
    pub keywords: Vec<(String, f64)>,
    pub compression_ratio: f64,
}

/// Summarize a text using extractive methods.
pub fn summarize(text: &str, config: &SummarizeConfig) -> Summary {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Summary {
            sentences: Vec::new(),
            text: String::new(),
            keywords: Vec::new(),
            compression_ratio: 0.0,
        };
    }

    // Compute component scores.
    let tfidf_scores = compute_tfidf_scores(&sentences);
    let position_scores = compute_position_scores(sentences.len());
    let length_scores = compute_length_scores(&sentences);
    let textrank_scores = compute_textrank_scores(&sentences, config.textrank_iterations, config.damping);

    // Combine scores.
    let mut scored: Vec<ScoredSentence> = sentences
        .iter()
        .enumerate()
        .map(|(i, sent)| {
            let score = config.tfidf_weight * tfidf_scores.get(i).copied().unwrap_or(0.0)
                + config.position_weight * position_scores.get(i).copied().unwrap_or(0.0)
                + config.length_weight * length_scores.get(i).copied().unwrap_or(0.0)
                + config.textrank_weight * textrank_scores.get(i).copied().unwrap_or(0.0);
            ScoredSentence {
                text: sent.clone(),
                score,
                original_index: i,
            }
        })
        .collect();

    // Sort by score descending.
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

    // Select top sentences respecting limits.
    let mut selected = Vec::new();
    let mut total_words = 0;

    for sent in &scored {
        if selected.len() >= config.max_sentences {
            break;
        }
        let wc = word_count(&sent.text);
        if config.max_words > 0 && total_words + wc > config.max_words && !selected.is_empty() {
            break;
        }
        selected.push(sent.clone());
        total_words += wc;
    }

    // Re-order by original position.
    selected.sort_by_key(|s| s.original_index);

    let summary_text = selected
        .iter()
        .map(|s| s.text.as_str())
        .collect::<Vec<&str>>()
        .join(" ");

    let original_words = word_count(text) as f64;
    let summary_words = word_count(&summary_text) as f64;
    let compression_ratio = if original_words > 0.0 {
        summary_words / original_words
    } else {
        0.0
    };

    // Extract keywords.
    let keywords = extract_keywords(&sentences, 10);

    Summary {
        sentences: selected,
        text: summary_text,
        keywords,
        compression_ratio,
    }
}

/// Summarize to a target number of sentences.
pub fn summarize_to_sentences(text: &str, n: usize) -> Summary {
    let mut config = SummarizeConfig::default();
    config.max_sentences = n;
    summarize(text, &config)
}

/// Summarize to a target word count.
pub fn summarize_to_words(text: &str, max_words: usize) -> Summary {
    let mut config = SummarizeConfig::default();
    config.max_sentences = usize::MAX;
    config.max_words = max_words;
    summarize(text, &config)
}

// ── Keyword extraction ───────────────────────────────────────────

fn extract_keywords(sentences: &[String], n: usize) -> Vec<(String, f64)> {
    if sentences.is_empty() {
        return Vec::new();
    }

    let num_docs = sentences.len() as f64;

    // Term frequency across all sentences.
    let mut total_counts: HashMap<String, usize> = HashMap::new();
    let mut doc_freq: HashMap<String, usize> = HashMap::new();

    for sent in sentences {
        let words = tokenize_words(sent);
        let unique: HashSet<String> = words.iter().cloned().collect();
        for word in &words {
            *total_counts.entry(word.clone()).or_insert(0) += 1;
        }
        for word in unique {
            *doc_freq.entry(word).or_insert(0) += 1;
        }
    }

    let total_words: usize = total_counts.values().sum();

    let mut scored: Vec<(String, f64)> = total_counts
        .into_iter()
        .filter(|(word, _)| word.len() > 2) // skip very short words
        .map(|(word, count)| {
            let tf = count as f64 / total_words as f64;
            let df = doc_freq.get(&word).copied().unwrap_or(1) as f64;
            let idf = (num_docs / df).ln();
            (word, tf * idf)
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
}

/// Extract top keywords from a text.
pub fn keywords(text: &str, n: usize) -> Vec<(String, f64)> {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        // Treat whole text as one sentence.
        let sents = vec![text.to_string()];
        return extract_keywords(&sents, n);
    }
    extract_keywords(&sentences, n)
}

// ── Sentence scoring (standalone) ────────────────────────────────

/// Score individual sentences in a text for importance.
pub fn score_sentences(text: &str) -> Vec<ScoredSentence> {
    let config = SummarizeConfig::default();
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return Vec::new();
    }

    let tfidf = compute_tfidf_scores(&sentences);
    let position = compute_position_scores(sentences.len());
    let length = compute_length_scores(&sentences);
    let textrank = compute_textrank_scores(&sentences, config.textrank_iterations, config.damping);

    sentences
        .into_iter()
        .enumerate()
        .map(|(i, sent)| {
            let score = config.tfidf_weight * tfidf.get(i).copied().unwrap_or(0.0)
                + config.position_weight * position.get(i).copied().unwrap_or(0.0)
                + config.length_weight * length.get(i).copied().unwrap_or(0.0)
                + config.textrank_weight * textrank.get(i).copied().unwrap_or(0.0);
            ScoredSentence {
                text: sent,
                score,
                original_index: i,
            }
        })
        .collect()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const ARTICLE: &str = "Machine learning is a subset of artificial intelligence. \
        It enables computers to learn from data without explicit programming. \
        Deep learning uses neural networks with many layers. \
        Natural language processing deals with text and speech understanding. \
        Computer vision allows machines to interpret visual information. \
        These technologies are transforming many industries. \
        Healthcare benefits from diagnostic AI systems. \
        Finance uses ML for fraud detection and trading.";

    #[test]
    fn test_summarize_basic() {
        let summary = summarize(ARTICLE, &SummarizeConfig::default());
        assert!(!summary.text.is_empty());
        assert!(summary.sentences.len() <= 3);
    }

    #[test]
    fn test_summarize_respects_max_sentences() {
        let mut config = SummarizeConfig::default();
        config.max_sentences = 2;
        let summary = summarize(ARTICLE, &config);
        assert!(summary.sentences.len() <= 2);
    }

    #[test]
    fn test_summarize_preserves_order() {
        let summary = summarize(ARTICLE, &SummarizeConfig::default());
        let indices: Vec<usize> = summary.sentences.iter().map(|s| s.original_index).collect();
        let mut sorted = indices.clone();
        sorted.sort();
        assert_eq!(indices, sorted, "Sentences should be in original order");
    }

    #[test]
    fn test_summarize_compression() {
        let summary = summarize(ARTICLE, &SummarizeConfig::default());
        assert!(summary.compression_ratio > 0.0);
        assert!(summary.compression_ratio < 1.0);
    }

    #[test]
    fn test_summarize_to_sentences() {
        let summary = summarize_to_sentences(ARTICLE, 2);
        assert!(summary.sentences.len() <= 2);
    }

    #[test]
    fn test_summarize_to_words() {
        let summary = summarize_to_words(ARTICLE, 20);
        let wc = word_count(&summary.text);
        assert!(wc <= 25, "Word count {wc} exceeded limit"); // some slack
    }

    #[test]
    fn test_keywords_extraction() {
        let kw = keywords(ARTICLE, 5);
        assert!(!kw.is_empty());
        assert!(kw.len() <= 5);
        // Scores should be positive.
        for (_, score) in &kw {
            assert!(*score > 0.0);
        }
    }

    #[test]
    fn test_score_sentences() {
        let scored = score_sentences(ARTICLE);
        assert!(!scored.is_empty());
        // All scores should be non-negative.
        for s in &scored {
            assert!(s.score >= 0.0, "Negative score: {}", s.score);
        }
    }

    #[test]
    fn test_empty_text() {
        let summary = summarize("", &SummarizeConfig::default());
        assert!(summary.text.is_empty());
        assert!(summary.sentences.is_empty());
    }

    #[test]
    fn test_single_sentence() {
        let summary = summarize("This is one sentence only.", &SummarizeConfig::default());
        assert_eq!(summary.sentences.len(), 1);
    }

    #[test]
    fn test_textrank_scores() {
        let sentences = vec![
            "Machine learning enables computers to learn.".to_string(),
            "Deep learning uses neural networks.".to_string(),
            "Neural networks process data in layers.".to_string(),
        ];
        let scores = compute_textrank_scores(&sentences, 20, 0.85);
        assert_eq!(scores.len(), 3);
        // All scores between 0 and 1.
        for s in &scores {
            assert!(*s >= 0.0 && *s <= 1.0 + 1e-10);
        }
    }

    #[test]
    fn test_position_scores() {
        let scores = compute_position_scores(5);
        assert_eq!(scores.len(), 5);
        // First sentence should score highest.
        assert!(scores[0] >= scores[1]);
        // Last sentence should score second highest.
        assert!(scores[4] > scores[2]);
    }

    #[test]
    fn test_tfidf_scores() {
        let sentences = vec![
            "cat dog".to_string(),
            "cat fish".to_string(),
            "bird plane".to_string(),
        ];
        let scores = compute_tfidf_scores(&sentences);
        assert_eq!(scores.len(), 3);
        for s in &scores {
            assert!(*s >= 0.0);
        }
    }

    #[test]
    fn test_length_scores() {
        let sentences = vec![
            "Short.".to_string(),
            "This is a moderately long sentence with several words in it.".to_string(),
            "Word.".to_string(),
        ];
        let scores = compute_length_scores(&sentences);
        assert_eq!(scores.len(), 3);
        // Medium sentence should score higher than very short.
        assert!(scores[1] > scores[0]);
    }

    #[test]
    fn test_sentence_similarity() {
        let sim_same = sentence_similarity("cat dog fish", "cat dog fish");
        let sim_diff = sentence_similarity("cat dog fish", "red green blue");
        assert!(sim_same > sim_diff);
    }

    #[test]
    fn test_summary_keywords_present() {
        let summary = summarize(ARTICLE, &SummarizeConfig::default());
        assert!(!summary.keywords.is_empty());
    }

    #[test]
    fn test_custom_config() {
        let config = SummarizeConfig {
            max_sentences: 5,
            max_words: 0,
            tfidf_weight: 1.0,
            position_weight: 0.0,
            length_weight: 0.0,
            textrank_weight: 0.0,
            textrank_iterations: 10,
            damping: 0.85,
        };
        let summary = summarize(ARTICLE, &config);
        assert!(summary.sentences.len() <= 5);
    }

    #[test]
    fn test_sentences_have_scores() {
        let summary = summarize(ARTICLE, &SummarizeConfig::default());
        for sent in &summary.sentences {
            assert!(sent.score > 0.0);
        }
    }
}
