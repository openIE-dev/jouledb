//! Lexicon-based sentiment analysis: word-level scoring, negation handling,
//! intensifier/diminisher modifiers, sentence and document sentiment,
//! and aspect-level extraction.

use std::collections::HashMap;

// ── Sentiment types ──────────────────────────────────────────────

/// Sentiment polarity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Polarity {
    Positive,
    Negative,
    Neutral,
}

/// Result of sentiment analysis on a text span.
#[derive(Debug, Clone)]
pub struct SentimentScore {
    /// Aggregate score: positive > 0, negative < 0.
    pub score: f64,
    /// Normalized to [-1.0, 1.0].
    pub normalized: f64,
    pub polarity: Polarity,
    /// Number of sentiment-bearing words found.
    pub word_count: usize,
}

impl SentimentScore {
    fn from_raw(score: f64, word_count: usize) -> Self {
        let normalized = if word_count > 0 {
            (score / word_count as f64).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        let polarity = if normalized > 0.05 {
            Polarity::Positive
        } else if normalized < -0.05 {
            Polarity::Negative
        } else {
            Polarity::Neutral
        };
        Self {
            score,
            normalized,
            polarity,
            word_count,
        }
    }
}

/// Aspect-level sentiment: sentiment tied to a specific topic/aspect.
#[derive(Debug, Clone)]
pub struct AspectSentiment {
    pub aspect: String,
    pub score: SentimentScore,
    /// Words that contributed to this aspect's sentiment.
    pub evidence: Vec<String>,
}

// ── Sentiment lexicon ────────────────────────────────────────────

/// A lexicon mapping words to sentiment scores.
#[derive(Debug, Clone)]
pub struct SentimentLexicon {
    words: HashMap<String, f64>,
    negators: Vec<String>,
    intensifiers: HashMap<String, f64>,
    diminishers: HashMap<String, f64>,
}

impl SentimentLexicon {
    /// Create an empty lexicon.
    pub fn new() -> Self {
        Self {
            words: HashMap::new(),
            negators: Vec::new(),
            intensifiers: HashMap::new(),
            diminishers: HashMap::new(),
        }
    }

    /// Create a lexicon with a built-in English word list.
    pub fn english() -> Self {
        let mut lex = Self::new();

        // Positive words (score 0.0 to 1.0)
        let positives: &[(&str, f64)] = &[
            ("good", 0.7), ("great", 0.9), ("excellent", 1.0), ("amazing", 0.95),
            ("wonderful", 0.9), ("fantastic", 0.95), ("awesome", 0.9), ("love", 0.8),
            ("happy", 0.8), ("joy", 0.85), ("beautiful", 0.8), ("brilliant", 0.9),
            ("outstanding", 0.95), ("superb", 0.9), ("perfect", 1.0), ("best", 0.9),
            ("nice", 0.6), ("pleasant", 0.6), ("enjoy", 0.7), ("like", 0.5),
            ("fine", 0.4), ("ok", 0.2), ("decent", 0.4), ("impressive", 0.8),
            ("remarkable", 0.8), ("delightful", 0.85), ("terrific", 0.9),
            ("positive", 0.6), ("strong", 0.5), ("powerful", 0.6), ("effective", 0.6),
            ("success", 0.7), ("successful", 0.7), ("win", 0.6), ("advantage", 0.5),
            ("recommend", 0.7), ("praised", 0.6), ("favorite", 0.7), ("well", 0.4),
            ("better", 0.6), ("improved", 0.5), ("reliable", 0.6), ("helpful", 0.6),
        ];

        // Negative words (score -1.0 to 0.0)
        let negatives: &[(&str, f64)] = &[
            ("bad", -0.7), ("terrible", -0.95), ("horrible", -0.95), ("awful", -0.9),
            ("worst", -1.0), ("hate", -0.9), ("ugly", -0.7), ("poor", -0.6),
            ("disappointing", -0.7), ("sad", -0.7), ("angry", -0.8), ("disgusting", -0.9),
            ("boring", -0.5), ("dull", -0.4), ("mediocre", -0.3), ("weak", -0.4),
            ("fail", -0.7), ("failure", -0.7), ("wrong", -0.5), ("broken", -0.6),
            ("useless", -0.8), ("waste", -0.6), ("annoying", -0.6), ("frustrating", -0.7),
            ("painful", -0.6), ("difficult", -0.3), ("problem", -0.4), ("issue", -0.3),
            ("negative", -0.5), ("damage", -0.6), ("loss", -0.5), ("lose", -0.5),
            ("worse", -0.6), ("inferior", -0.6), ("defect", -0.5), ("error", -0.4),
            ("bug", -0.3), ("crash", -0.6), ("slow", -0.3), ("expensive", -0.3),
            ("overpriced", -0.5), ("cheap", -0.2), ("unreliable", -0.6),
        ];

        for (word, score) in positives {
            lex.words.insert(word.to_string(), *score);
        }
        for (word, score) in negatives {
            lex.words.insert(word.to_string(), *score);
        }

        // Negators
        lex.negators = vec![
            "not".to_string(), "no".to_string(), "never".to_string(),
            "neither".to_string(), "nobody".to_string(), "nothing".to_string(),
            "nor".to_string(), "nowhere".to_string(), "hardly".to_string(),
            "barely".to_string(), "scarcely".to_string(), "don't".to_string(),
            "doesn't".to_string(), "didn't".to_string(), "wasn't".to_string(),
            "weren't".to_string(), "won't".to_string(), "wouldn't".to_string(),
            "couldn't".to_string(), "shouldn't".to_string(), "isn't".to_string(),
            "aren't".to_string(), "hasn't".to_string(), "haven't".to_string(),
            "without".to_string(), "lack".to_string(),
        ];

        // Intensifiers (multiply score)
        let intensifiers: &[(&str, f64)] = &[
            ("very", 1.5), ("extremely", 2.0), ("incredibly", 1.8),
            ("absolutely", 1.9), ("totally", 1.7), ("completely", 1.8),
            ("really", 1.4), ("truly", 1.5), ("highly", 1.5),
            ("so", 1.3), ("remarkably", 1.6), ("exceptionally", 1.8),
            ("especially", 1.3), ("particularly", 1.3), ("most", 1.4),
        ];
        for (word, mult) in intensifiers {
            lex.intensifiers.insert(word.to_string(), *mult);
        }

        // Diminishers (reduce score)
        let diminishers: &[(&str, f64)] = &[
            ("slightly", 0.5), ("somewhat", 0.6), ("rather", 0.7),
            ("fairly", 0.7), ("a bit", 0.5), ("a little", 0.5),
            ("kind of", 0.6), ("sort of", 0.6), ("barely", 0.3),
            ("marginally", 0.4), ("mildly", 0.5), ("partly", 0.6),
        ];
        for (word, mult) in diminishers {
            lex.diminishers.insert(word.to_string(), *mult);
        }

        lex
    }

    /// Add or update a word's sentiment score.
    pub fn set_word(&mut self, word: &str, score: f64) {
        self.words.insert(word.to_lowercase(), score.clamp(-1.0, 1.0));
    }

    /// Get a word's sentiment score.
    pub fn get_score(&self, word: &str) -> Option<f64> {
        self.words.get(&word.to_lowercase()).copied()
    }

    /// Add a negator word.
    pub fn add_negator(&mut self, word: &str) {
        let lower = word.to_lowercase();
        if !self.negators.contains(&lower) {
            self.negators.push(lower);
        }
    }

    /// Add an intensifier with its multiplier.
    pub fn add_intensifier(&mut self, word: &str, multiplier: f64) {
        self.intensifiers.insert(word.to_lowercase(), multiplier);
    }

    /// Add a diminisher with its multiplier.
    pub fn add_diminisher(&mut self, word: &str, multiplier: f64) {
        self.diminishers.insert(word.to_lowercase(), multiplier);
    }

    pub fn is_negator(&self, word: &str) -> bool {
        self.negators.contains(&word.to_lowercase())
    }

    pub fn get_intensifier(&self, word: &str) -> Option<f64> {
        self.intensifiers.get(&word.to_lowercase()).copied()
    }

    pub fn get_diminisher(&self, word: &str) -> Option<f64> {
        self.diminishers.get(&word.to_lowercase()).copied()
    }

    pub fn word_count(&self) -> usize {
        self.words.len()
    }
}

impl Default for SentimentLexicon {
    fn default() -> Self {
        Self::english()
    }
}

// ── Tokenizer (simple for sentiment) ─────────────────────────────

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|s| !s.is_empty())
        .map(|s| s.to_lowercase())
        .collect()
}

// ── Sentence-level scoring ───────────────────────────────────────

/// Score a sentence for sentiment, handling negation and modifiers.
pub fn score_sentence(text: &str, lexicon: &SentimentLexicon) -> SentimentScore {
    let tokens = tokenize(text);
    let mut total_score = 0.0;
    let mut sentiment_words = 0usize;

    let mut i = 0;
    while i < tokens.len() {
        let word = &tokens[i];

        // Check for multi-word diminishers (skip if matched).
        if i + 1 < tokens.len() {
            let bigram = format!("{} {}", word, tokens[i + 1]);
            if let Some(mult) = lexicon.get_diminisher(&bigram) {
                // Next sentiment word gets diminished.
                if i + 2 < tokens.len() {
                    if let Some(base_score) = lexicon.get_score(&tokens[i + 2]) {
                        total_score += base_score * mult;
                        sentiment_words += 1;
                        i += 3;
                        continue;
                    }
                }
                i += 2;
                continue;
            }
        }

        // Look for negation.
        let is_negated = if i > 0 { lexicon.is_negator(&tokens[i - 1]) } else { false };

        // Look for intensifier in previous position.
        let intensifier = if i > 0 {
            lexicon.get_intensifier(&tokens[i - 1])
        } else {
            None
        };

        // Look for diminisher in previous position.
        let diminisher = if i > 0 {
            lexicon.get_diminisher(&tokens[i - 1])
        } else {
            None
        };

        if let Some(mut base_score) = lexicon.get_score(word) {
            // Apply intensifier or diminisher.
            if let Some(mult) = intensifier {
                base_score *= mult;
            } else if let Some(mult) = diminisher {
                base_score *= mult;
            }

            // Apply negation (flip sign and reduce magnitude slightly).
            if is_negated {
                base_score = -base_score * 0.8;
            }

            total_score += base_score;
            sentiment_words += 1;
        }

        i += 1;
    }

    SentimentScore::from_raw(total_score, sentiment_words)
}

// ── Document-level scoring ───────────────────────────────────────

/// Score an entire document by averaging sentence scores.
pub fn score_document(text: &str, lexicon: &SentimentLexicon) -> SentimentScore {
    let sentences = split_sentences(text);
    if sentences.is_empty() {
        return SentimentScore::from_raw(0.0, 0);
    }

    let mut total_score = 0.0;
    let mut total_words = 0;

    for sentence in &sentences {
        let s = score_sentence(sentence, lexicon);
        total_score += s.score;
        total_words += s.word_count;
    }

    SentimentScore::from_raw(total_score, total_words)
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if ch == '.' || ch == '!' || ch == '?' {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
    }

    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }

    sentences
}

// ── Aspect-level sentiment ───────────────────────────────────────

/// Analyze sentiment for specific aspects mentioned in the text.
///
/// `aspects` is a list of aspect keywords to look for. For each found aspect,
/// sentiment is computed from the surrounding context window.
pub fn aspect_sentiment(
    text: &str,
    aspects: &[&str],
    lexicon: &SentimentLexicon,
    window_size: usize,
) -> Vec<AspectSentiment> {
    let tokens = tokenize(text);
    let mut results = Vec::new();

    let conjunctions: &[&str] = &["but", "however", "although", "yet", "though", "while", "whereas"];

    for aspect in aspects {
        let aspect_lower = aspect.to_lowercase();
        let mut aspect_score = 0.0;
        let mut aspect_words = 0;
        let mut evidence = Vec::new();

        for (i, token) in tokens.iter().enumerate() {
            if *token == aspect_lower {
                // Look at surrounding context window, stopping at conjunctions.
                let start = i.saturating_sub(window_size);
                let end = (i + window_size + 1).min(tokens.len());

                // Find effective backward boundary (stop at conjunctions).
                let mut effective_start = start;
                for k in (start..i).rev() {
                    if conjunctions.contains(&tokens[k].as_str()) {
                        effective_start = k + 1;
                        break;
                    }
                }

                // Find effective forward boundary (stop at conjunctions).
                let mut effective_end = end;
                for k in (i + 1)..end {
                    if conjunctions.contains(&tokens[k].as_str()) {
                        effective_end = k;
                        break;
                    }
                }

                for j in effective_start..effective_end {
                    if j == i {
                        continue;
                    }
                    let is_negated = if j > 0 {
                        lexicon.is_negator(&tokens[j - 1])
                    } else {
                        false
                    };

                    if let Some(mut score) = lexicon.get_score(&tokens[j]) {
                        if is_negated {
                            score = -score * 0.8;
                        }
                        aspect_score += score;
                        aspect_words += 1;
                        evidence.push(tokens[j].clone());
                    }
                }
            }
        }

        if aspect_words > 0 {
            results.push(AspectSentiment {
                aspect: aspect.to_string(),
                score: SentimentScore::from_raw(aspect_score, aspect_words),
                evidence,
            });
        }
    }

    results
}

// ── Batch analysis ───────────────────────────────────────────────

/// Score multiple texts at once.
pub fn batch_score(texts: &[&str], lexicon: &SentimentLexicon) -> Vec<SentimentScore> {
    texts.iter().map(|t| score_sentence(t, lexicon)).collect()
}

/// Classify texts into positive/negative/neutral.
pub fn classify_sentiment(text: &str, lexicon: &SentimentLexicon) -> Polarity {
    score_sentence(text, lexicon).polarity
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn lex() -> SentimentLexicon {
        SentimentLexicon::english()
    }

    #[test]
    fn test_positive_sentence() {
        let score = score_sentence("This is a great product", &lex());
        assert!(score.score > 0.0);
        assert_eq!(score.polarity, Polarity::Positive);
    }

    #[test]
    fn test_negative_sentence() {
        let score = score_sentence("This is a terrible product", &lex());
        assert!(score.score < 0.0);
        assert_eq!(score.polarity, Polarity::Negative);
    }

    #[test]
    fn test_neutral_sentence() {
        let score = score_sentence("The meeting is at noon", &lex());
        assert_eq!(score.polarity, Polarity::Neutral);
    }

    #[test]
    fn test_negation_flips_positive() {
        let pos = score_sentence("This is good", &lex());
        let neg = score_sentence("This is not good", &lex());
        assert!(pos.score > 0.0);
        assert!(neg.score < 0.0);
    }

    #[test]
    fn test_negation_flips_negative() {
        let neg = score_sentence("This is bad", &lex());
        let pos = score_sentence("This is not bad", &lex());
        assert!(neg.score < 0.0);
        assert!(pos.score > 0.0);
    }

    #[test]
    fn test_intensifier() {
        let normal = score_sentence("This is good", &lex());
        let intense = score_sentence("This is very good", &lex());
        assert!(intense.score > normal.score);
    }

    #[test]
    fn test_diminisher() {
        let normal = score_sentence("This is good", &lex());
        let diminished = score_sentence("This is slightly good", &lex());
        assert!(diminished.score < normal.score);
        assert!(diminished.score > 0.0); // still positive
    }

    #[test]
    fn test_document_scoring() {
        let text = "Great product. Love it. Works perfectly.";
        let score = score_document(text, &lex());
        assert!(score.score > 0.0);
        assert_eq!(score.polarity, Polarity::Positive);
    }

    #[test]
    fn test_mixed_document() {
        // Positive and negative cancel out somewhat.
        let text = "The design is great. But the performance is terrible.";
        let score = score_document(text, &lex());
        // With great(+0.9) and terrible(-0.95), overall near zero.
        assert!(score.normalized.abs() < 0.5);
    }

    #[test]
    fn test_aspect_sentiment_positive() {
        let text = "The camera quality is excellent but the battery is terrible";
        let aspects = aspect_sentiment(text, &["camera", "battery"], &lex(), 3);
        let camera = aspects.iter().find(|a| a.aspect == "camera").unwrap();
        let battery = aspects.iter().find(|a| a.aspect == "battery").unwrap();
        assert_eq!(camera.score.polarity, Polarity::Positive);
        assert_eq!(battery.score.polarity, Polarity::Negative);
    }

    #[test]
    fn test_custom_lexicon_word() {
        let mut lex = SentimentLexicon::new();
        lex.set_word("splendid", 0.9);
        let score = score_sentence("This is splendid", &lex);
        assert!(score.score > 0.0);
    }

    #[test]
    fn test_custom_negator() {
        let mut lex = SentimentLexicon::new();
        lex.set_word("good", 0.7);
        lex.add_negator("hardly");
        let score = score_sentence("This is hardly good", &lex);
        assert!(score.score < 0.0);
    }

    #[test]
    fn test_batch_score() {
        let texts = &["Great product!", "Terrible service.", "It works."];
        let scores = batch_score(texts, &lex());
        assert_eq!(scores.len(), 3);
        assert!(scores[0].score > 0.0);
        assert!(scores[1].score < 0.0);
    }

    #[test]
    fn test_classify_sentiment() {
        assert_eq!(classify_sentiment("excellent work", &lex()), Polarity::Positive);
        assert_eq!(classify_sentiment("awful result", &lex()), Polarity::Negative);
    }

    #[test]
    fn test_lexicon_word_count() {
        let lex = lex();
        assert!(lex.word_count() > 50);
    }

    #[test]
    fn test_empty_text() {
        let score = score_sentence("", &lex());
        assert_eq!(score.polarity, Polarity::Neutral);
        assert_eq!(score.word_count, 0);
    }

    #[test]
    fn test_score_normalized_range() {
        let score = score_sentence("very very extremely good great excellent", &lex());
        assert!(score.normalized >= -1.0 && score.normalized <= 1.0);
    }

    #[test]
    fn test_lexicon_is_negator() {
        let lex = lex();
        assert!(lex.is_negator("not"));
        assert!(lex.is_negator("don't"));
        assert!(!lex.is_negator("good"));
    }
}
