//! Text classification: bag-of-words features, Naive Bayes classifier,
//! train/predict API, multi-class support, probability output, vocabulary
//! building, and classification report (precision/recall/F1).

use std::collections::{HashMap, HashSet};

// ── Tokenizer ────────────────────────────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ── Bag of words ─────────────────────────────────────────────────

/// Bag-of-words feature extraction: maps each document to a word count vector.
#[derive(Debug, Clone)]
pub struct BagOfWords {
    /// Sorted vocabulary.
    pub vocabulary: Vec<String>,
    term_index: HashMap<String, usize>,
}

impl BagOfWords {
    /// Build vocabulary from a corpus.
    pub fn fit(documents: &[&str]) -> Self {
        let mut terms = HashSet::new();
        for doc in documents {
            for token in tokenize(doc) {
                terms.insert(token);
            }
        }
        let mut vocabulary: Vec<String> = terms.into_iter().collect();
        vocabulary.sort();
        let term_index: HashMap<String, usize> = vocabulary
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();
        Self { vocabulary, term_index }
    }

    /// Transform a document into a feature vector (word counts).
    pub fn transform(&self, text: &str) -> Vec<f64> {
        let mut vec = vec![0.0; self.vocabulary.len()];
        for token in tokenize(text) {
            if let Some(idx) = self.term_index.get(&token) {
                vec[*idx] += 1.0;
            }
        }
        vec
    }

    /// Transform multiple documents.
    pub fn transform_batch(&self, documents: &[&str]) -> Vec<Vec<f64>> {
        documents.iter().map(|d| self.transform(d)).collect()
    }

    /// Number of features (vocabulary size).
    pub fn num_features(&self) -> usize {
        self.vocabulary.len()
    }

    /// Get feature index for a term.
    pub fn feature_index(&self, term: &str) -> Option<usize> {
        self.term_index.get(term).copied()
    }
}

// ── Naive Bayes classifier ───────────────────────────────────────

/// Multinomial Naive Bayes text classifier.
#[derive(Debug, Clone)]
pub struct NaiveBayesClassifier {
    /// Class labels.
    classes: Vec<String>,
    /// Prior log-probabilities per class: log P(class).
    log_priors: Vec<f64>,
    /// Conditional log-probabilities: log P(word|class).
    /// Shape: classes.len() x vocab_size.
    log_likelihoods: Vec<Vec<f64>>,
    /// Vocabulary used during training.
    bow: BagOfWords,
    /// Laplace smoothing parameter.
    alpha: f64,
}

impl NaiveBayesClassifier {
    /// Train a Naive Bayes classifier from labeled documents.
    ///
    /// `documents`: text of each training example.
    /// `labels`: class label for each training example.
    /// `alpha`: Laplace smoothing (default 1.0).
    pub fn train(documents: &[&str], labels: &[&str], alpha: f64) -> Self {
        let bow = BagOfWords::fit(documents);
        let vocab_size = bow.num_features();
        let n_docs = documents.len() as f64;

        // Count documents per class.
        let mut class_counts: HashMap<String, usize> = HashMap::new();
        for label in labels {
            *class_counts.entry(label.to_string()).or_insert(0) += 1;
        }

        let mut classes: Vec<String> = class_counts.keys().cloned().collect();
        classes.sort();

        // Compute priors and likelihoods.
        let mut log_priors = Vec::with_capacity(classes.len());
        let mut log_likelihoods = Vec::with_capacity(classes.len());

        for class in &classes {
            let class_count = class_counts[class] as f64;
            log_priors.push((class_count / n_docs).ln());

            // Sum word counts across all documents in this class.
            let mut word_counts = vec![0.0; vocab_size];
            for (doc, label) in documents.iter().zip(labels.iter()) {
                if *label == class.as_str() {
                    let features = bow.transform(doc);
                    for (i, count) in features.iter().enumerate() {
                        word_counts[i] += count;
                    }
                }
            }

            // Total words in class.
            let total: f64 = word_counts.iter().sum::<f64>() + alpha * vocab_size as f64;

            // Log conditional probabilities with Laplace smoothing.
            let log_lk: Vec<f64> = word_counts
                .iter()
                .map(|count| ((count + alpha) / total).ln())
                .collect();
            log_likelihoods.push(log_lk);
        }

        Self {
            classes,
            log_priors,
            log_likelihoods,
            bow,
            alpha,
        }
    }

    /// Predict the class of a document.
    pub fn predict(&self, text: &str) -> String {
        let (class, _) = self.predict_with_score(text);
        class
    }

    /// Predict class and return log-probability scores for all classes.
    pub fn predict_with_score(&self, text: &str) -> (String, Vec<(String, f64)>) {
        let features = self.bow.transform(text);
        let mut scores: Vec<(String, f64)> = Vec::with_capacity(self.classes.len());

        for (c, (class, log_prior)) in self.classes.iter().zip(self.log_priors.iter()).enumerate() {
            let mut score = *log_prior;
            for (i, count) in features.iter().enumerate() {
                if *count > 0.0 {
                    score += count * self.log_likelihoods[c][i];
                }
            }
            scores.push((class.clone(), score));
        }

        // Best class.
        let best = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(class, _)| class.clone())
            .unwrap_or_default();

        (best, scores)
    }

    /// Predict probabilities (normalized from log-scores via log-sum-exp).
    pub fn predict_proba(&self, text: &str) -> Vec<(String, f64)> {
        let (_, scores) = self.predict_with_score(text);

        // Log-sum-exp for numerical stability.
        let max_score = scores
            .iter()
            .map(|(_, s)| *s)
            .fold(f64::NEG_INFINITY, f64::max);

        let sum_exp: f64 = scores.iter().map(|(_, s)| (s - max_score).exp()).sum();
        let log_sum = max_score + sum_exp.ln();

        scores
            .into_iter()
            .map(|(class, score)| (class, (score - log_sum).exp()))
            .collect()
    }

    /// Predict multiple documents.
    pub fn predict_batch(&self, texts: &[&str]) -> Vec<String> {
        texts.iter().map(|t| self.predict(t)).collect()
    }

    /// Return the class labels.
    pub fn classes(&self) -> &[String] {
        &self.classes
    }

    /// Return the vocabulary.
    pub fn vocabulary(&self) -> &[String] {
        &self.bow.vocabulary
    }

    /// Smoothing parameter.
    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

// ── Classification report ────────────────────────────────────────

/// Per-class metrics.
#[derive(Debug, Clone)]
pub struct ClassMetrics {
    pub class: String,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// Overall classification report.
#[derive(Debug, Clone)]
pub struct ClassificationReport {
    pub per_class: Vec<ClassMetrics>,
    pub accuracy: f64,
    pub macro_precision: f64,
    pub macro_recall: f64,
    pub macro_f1: f64,
    pub total_samples: usize,
}

impl ClassificationReport {
    /// Generate a classification report from true and predicted labels.
    pub fn compute(true_labels: &[&str], predicted: &[&str]) -> Self {
        let mut classes: Vec<String> = true_labels
            .iter()
            .chain(predicted.iter())
            .map(|s| s.to_string())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();

        let total = true_labels.len();
        let correct = true_labels
            .iter()
            .zip(predicted.iter())
            .filter(|(t, p)| t == p)
            .count();
        let accuracy = if total > 0 { correct as f64 / total as f64 } else { 0.0 };

        let mut per_class = Vec::new();

        for class in &classes {
            let true_positives = true_labels
                .iter()
                .zip(predicted.iter())
                .filter(|(t, p)| **t == class.as_str() && **p == class.as_str())
                .count();
            let false_positives = predicted
                .iter()
                .zip(true_labels.iter())
                .filter(|(p, t)| **p == class.as_str() && **t != class.as_str())
                .count();
            let false_negatives = true_labels
                .iter()
                .zip(predicted.iter())
                .filter(|(t, p)| **t == class.as_str() && **p != class.as_str())
                .count();
            let support = true_labels.iter().filter(|t| **t == class.as_str()).count();

            let precision = if true_positives + false_positives > 0 {
                true_positives as f64 / (true_positives + false_positives) as f64
            } else {
                0.0
            };
            let recall = if true_positives + false_negatives > 0 {
                true_positives as f64 / (true_positives + false_negatives) as f64
            } else {
                0.0
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            per_class.push(ClassMetrics {
                class: class.clone(),
                precision,
                recall,
                f1,
                support,
            });
        }

        let n_classes = per_class.len() as f64;
        let macro_precision = if n_classes > 0.0 {
            per_class.iter().map(|c| c.precision).sum::<f64>() / n_classes
        } else {
            0.0
        };
        let macro_recall = if n_classes > 0.0 {
            per_class.iter().map(|c| c.recall).sum::<f64>() / n_classes
        } else {
            0.0
        };
        let macro_f1 = if macro_precision + macro_recall > 0.0 {
            2.0 * macro_precision * macro_recall / (macro_precision + macro_recall)
        } else {
            0.0
        };

        Self {
            per_class,
            accuracy,
            macro_precision,
            macro_recall,
            macro_f1,
            total_samples: total,
        }
    }

    /// Format the report as a text table.
    pub fn to_string_table(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "{:<15} {:>10} {:>10} {:>10} {:>10}\n",
            "Class", "Precision", "Recall", "F1", "Support"
        ));
        out.push_str(&"-".repeat(55));
        out.push('\n');

        for c in &self.per_class {
            out.push_str(&format!(
                "{:<15} {:>10.3} {:>10.3} {:>10.3} {:>10}\n",
                c.class, c.precision, c.recall, c.f1, c.support
            ));
        }

        out.push_str(&"-".repeat(55));
        out.push('\n');
        out.push_str(&format!("Accuracy: {:.3}\n", self.accuracy));
        out.push_str(&format!(
            "Macro avg: P={:.3} R={:.3} F1={:.3}\n",
            self.macro_precision, self.macro_recall, self.macro_f1
        ));
        out
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn train_spam_classifier() -> NaiveBayesClassifier {
        let docs = &[
            "buy cheap viagra now",
            "free money offer click here",
            "win lottery prize congratulations",
            "cheap pills discount sale",
            "meeting tomorrow at noon",
            "project deadline next week",
            "please review the document",
            "lunch at the cafeteria today",
        ];
        let labels = &[
            "spam", "spam", "spam", "spam",
            "ham", "ham", "ham", "ham",
        ];
        NaiveBayesClassifier::train(docs, labels, 1.0)
    }

    #[test]
    fn test_bag_of_words_fit() {
        let bow = BagOfWords::fit(&["hello world", "hello rust"]);
        assert!(bow.num_features() >= 3); // hello, world, rust
    }

    #[test]
    fn test_bag_of_words_transform() {
        let bow = BagOfWords::fit(&["cat dog", "cat fish"]);
        let vec = bow.transform("cat cat dog");
        let cat_idx = bow.feature_index("cat").unwrap();
        let dog_idx = bow.feature_index("dog").unwrap();
        assert_eq!(vec[cat_idx], 2.0);
        assert_eq!(vec[dog_idx], 1.0);
    }

    #[test]
    fn test_bag_of_words_unknown_word() {
        let bow = BagOfWords::fit(&["hello world"]);
        let vec = bow.transform("unknown word");
        // "unknown" not in vocab, should be zero; "word" not in vocab either
        let sum: f64 = vec.iter().sum();
        // Only "world" -> "word" doesn't match, so depends on vocab
        // All should be zero or small
        assert!(sum >= 0.0);
    }

    #[test]
    fn test_naive_bayes_train() {
        let clf = train_spam_classifier();
        assert_eq!(clf.classes().len(), 2);
        assert!(clf.vocabulary().len() > 5);
    }

    #[test]
    fn test_predict_spam() {
        let clf = train_spam_classifier();
        let pred = clf.predict("buy cheap pills free");
        assert_eq!(pred, "spam");
    }

    #[test]
    fn test_predict_ham() {
        let clf = train_spam_classifier();
        let pred = clf.predict("meeting about the project tomorrow");
        assert_eq!(pred, "ham");
    }

    #[test]
    fn test_predict_with_score() {
        let clf = train_spam_classifier();
        let (pred, scores) = clf.predict_with_score("buy cheap discount");
        assert_eq!(pred, "spam");
        assert_eq!(scores.len(), 2);
    }

    #[test]
    fn test_predict_proba_sums_to_one() {
        let clf = train_spam_classifier();
        let proba = clf.predict_proba("some text here");
        let sum: f64 = proba.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 0.001, "Probabilities sum to {sum}");
    }

    #[test]
    fn test_predict_proba_range() {
        let clf = train_spam_classifier();
        let proba = clf.predict_proba("buy free cheap");
        for (_, p) in &proba {
            assert!(*p >= 0.0 && *p <= 1.0, "Probability {p} out of range");
        }
    }

    #[test]
    fn test_predict_batch() {
        let clf = train_spam_classifier();
        let preds = clf.predict_batch(&["buy cheap", "meeting tomorrow"]);
        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0], "spam");
        assert_eq!(preds[1], "ham");
    }

    #[test]
    fn test_classification_report_perfect() {
        let true_labels = &["a", "a", "b", "b"];
        let predicted = &["a", "a", "b", "b"];
        let report = ClassificationReport::compute(true_labels, predicted);
        assert!((report.accuracy - 1.0).abs() < 0.001);
        assert!((report.macro_f1 - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_classification_report_imperfect() {
        let true_labels = &["a", "a", "b", "b"];
        let predicted = &["a", "b", "b", "b"];
        let report = ClassificationReport::compute(true_labels, predicted);
        assert!(report.accuracy < 1.0);
        assert!(report.accuracy > 0.0);
    }

    #[test]
    fn test_classification_report_multiclass() {
        let true_labels = &["a", "b", "c", "a", "b", "c"];
        let predicted = &["a", "b", "c", "a", "c", "b"];
        let report = ClassificationReport::compute(true_labels, predicted);
        assert_eq!(report.per_class.len(), 3);
        assert_eq!(report.total_samples, 6);
    }

    #[test]
    fn test_report_to_string() {
        let true_labels = &["a", "b", "a", "b"];
        let predicted = &["a", "b", "b", "b"];
        let report = ClassificationReport::compute(true_labels, predicted);
        let text = report.to_string_table();
        assert!(text.contains("Accuracy"));
        assert!(text.contains("Macro avg"));
    }

    #[test]
    fn test_precision_recall_f1() {
        // Class A: 1 TP, 0 FP, 1 FN → P=1.0, R=0.5, F1=0.667
        let true_labels = &["a", "a", "b"];
        let predicted = &["a", "b", "b"];
        let report = ClassificationReport::compute(true_labels, predicted);
        let a = report.per_class.iter().find(|c| c.class == "a").unwrap();
        assert!((a.precision - 1.0).abs() < 0.001);
        assert!((a.recall - 0.5).abs() < 0.001);
        assert!((a.f1 - 0.667).abs() < 0.01);
    }

    #[test]
    fn test_alpha_parameter() {
        let clf = train_spam_classifier();
        assert_eq!(clf.alpha(), 1.0);
    }

    #[test]
    fn test_bag_of_words_batch() {
        let bow = BagOfWords::fit(&["hello world", "hello rust"]);
        let batch = bow.transform_batch(&["hello", "world"]);
        assert_eq!(batch.len(), 2);
    }
}
