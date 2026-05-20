//! Beam search decoding with configurable beam width, length penalty, and n-gram blocking.
//!
//! Implements beam search for autoregressive sequence generation with
//! configurable beam width, length normalization (Wu et al. 2016),
//! n-gram repetition blocking, top-k and top-p (nucleus) filtering,
//! and early stopping when all beams have produced an end-of-sequence token.

use std::fmt;
use std::collections::HashMap;

// ── Beam Hypothesis ──────────────────────────────────────────────

/// A single hypothesis (partial sequence) maintained during beam search.
#[derive(Debug, Clone)]
pub struct Hypothesis {
    /// Token indices generated so far.
    pub tokens: Vec<usize>,
    /// Log-probability of the sequence.
    pub log_prob: f64,
    /// Whether this hypothesis has generated an EOS token.
    pub finished: bool,
}

impl Hypothesis {
    /// Create a new hypothesis starting from a BOS token.
    pub fn new(bos_token: usize) -> Self {
        Self {
            tokens: vec![bos_token],
            log_prob: 0.0,
            finished: false,
        }
    }

    /// Extend this hypothesis with a new token and its log-probability.
    pub fn extend(&self, token: usize, token_log_prob: f64) -> Self {
        let mut tokens = self.tokens.clone();
        tokens.push(token);
        Self {
            tokens,
            log_prob: self.log_prob + token_log_prob,
            finished: self.finished,
        }
    }

    /// Length of the generated sequence (excluding BOS).
    pub fn output_length(&self) -> usize {
        if self.tokens.is_empty() { 0 } else { self.tokens.len() - 1 }
    }

    /// Compute length-normalized score.
    pub fn normalized_score(&self, alpha: f64) -> f64 {
        let length_penalty = ((5.0 + self.output_length() as f64) / 6.0).powf(alpha);
        self.log_prob / length_penalty
    }

    /// Check if a specific n-gram would be repeated.
    pub fn would_repeat_ngram(&self, token: usize, n: usize) -> bool {
        if n == 0 || self.tokens.len() < n {
            return false;
        }
        // Build the candidate n-gram: last (n-1) tokens + new token
        let start = self.tokens.len() - (n - 1);
        let candidate: Vec<usize> = self.tokens[start..].iter().copied()
            .chain(std::iter::once(token))
            .collect();

        // Check if this n-gram appeared before
        for window_start in 0..=(self.tokens.len().saturating_sub(n)) {
            if self.tokens[window_start..window_start + n] == candidate[..] {
                return true;
            }
        }
        false
    }
}

impl fmt::Display for Hypothesis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Hypothesis(len={}, log_p={:.4}, finished={})",
            self.tokens.len(),
            self.log_prob,
            self.finished
        )
    }
}

// ── Beam Search Configuration ────────────────────────────────────

/// Configuration for beam search decoding.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams (hypotheses) to maintain at each step.
    pub beam_width: usize,
    /// Maximum number of tokens to generate.
    pub max_length: usize,
    /// Length penalty exponent (alpha). 0 = no penalty, 1 = linear.
    pub length_penalty: f64,
    /// N-gram size for repetition blocking. 0 = disabled.
    pub no_repeat_ngram: usize,
    /// Top-k filtering before beam selection. 0 = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) filtering threshold. 1.0 = disabled.
    pub top_p: f64,
    /// Temperature for logit scaling. 1.0 = no scaling.
    pub temperature: f64,
    /// End-of-sequence token ID.
    pub eos_token: usize,
    /// Beginning-of-sequence token ID.
    pub bos_token: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Stop early once all beams are finished.
    pub early_stopping: bool,
}

impl BeamSearchConfig {
    pub fn new(vocab_size: usize) -> Self {
        Self {
            beam_width: 4,
            max_length: 128,
            length_penalty: 0.6,
            no_repeat_ngram: 0,
            top_k: 0,
            top_p: 1.0,
            temperature: 1.0,
            eos_token: 1,
            bos_token: 0,
            vocab_size,
            early_stopping: true,
        }
    }

    pub fn with_beam_width(mut self, width: usize) -> Self {
        self.beam_width = width;
        self
    }

    pub fn with_max_length(mut self, len: usize) -> Self {
        self.max_length = len;
        self
    }

    pub fn with_length_penalty(mut self, alpha: f64) -> Self {
        self.length_penalty = alpha;
        self
    }

    pub fn with_no_repeat_ngram(mut self, n: usize) -> Self {
        self.no_repeat_ngram = n;
        self
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_top_p(mut self, p: f64) -> Self {
        self.top_p = p;
        self
    }

    pub fn with_temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    pub fn with_eos_token(mut self, token: usize) -> Self {
        self.eos_token = token;
        self
    }

    pub fn with_bos_token(mut self, token: usize) -> Self {
        self.bos_token = token;
        self
    }

    pub fn with_early_stopping(mut self, early: bool) -> Self {
        self.early_stopping = early;
        self
    }
}

// ── Logit Processing ─────────────────────────────────────────────

/// Apply temperature scaling to logits.
fn apply_temperature(logits: &[f64], temp: f64) -> Vec<f64> {
    if (temp - 1.0).abs() < 1e-12 {
        return logits.to_vec();
    }
    logits.iter().map(|l| l / temp).collect()
}

/// Convert logits to log-probabilities (log-softmax).
fn log_softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let log_sum_exp = logits.iter()
        .map(|l| (l - max_val).exp())
        .sum::<f64>()
        .ln() + max_val;
    logits.iter().map(|l| l - log_sum_exp).collect()
}

/// Softmax over logits.
fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_val = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

/// Top-k filtering: keep only the top k logits, set rest to -inf.
fn top_k_filter(logits: &[f64], k: usize) -> Vec<f64> {
    if k == 0 || k >= logits.len() {
        return logits.to_vec();
    }
    let mut indexed: Vec<(usize, f64)> = logits.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let threshold = indexed[k - 1].1;
    logits.iter().map(|v| {
        if *v >= threshold { *v } else { f64::NEG_INFINITY }
    }).collect()
}

/// Top-p (nucleus) filtering.
fn top_p_filter(logits: &[f64], p: f64) -> Vec<f64> {
    if p >= 1.0 {
        return logits.to_vec();
    }
    let probs = softmax(logits);
    let mut indexed: Vec<(usize, f64)> = probs.iter().enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumsum = 0.0;
    let mut keep = vec![false; logits.len()];
    for &(idx, prob) in &indexed {
        keep[idx] = true;
        cumsum += prob;
        if cumsum >= p {
            break;
        }
    }

    logits.iter().enumerate().map(|(i, &v)| {
        if keep[i] { v } else { f64::NEG_INFINITY }
    }).collect()
}

// ── Beam Search Engine ───────────────────────────────────────────

/// Beam search decoder.
///
/// Takes a scoring function that maps a token sequence to next-token logits
/// and performs beam search to find the highest-scoring output sequence.
#[derive(Debug, Clone)]
pub struct BeamSearch {
    pub config: BeamSearchConfig,
    /// Completed hypotheses.
    pub finished_hypotheses: Vec<Hypothesis>,
    /// Number of decoding steps executed.
    pub steps_taken: usize,
}

impl BeamSearch {
    pub fn new(config: BeamSearchConfig) -> Self {
        Self {
            config,
            finished_hypotheses: Vec::new(),
            steps_taken: 0,
        }
    }

    /// Run beam search using a scoring function.
    ///
    /// `score_fn` takes a token sequence and returns logits of length `vocab_size`.
    pub fn search<F>(&mut self, score_fn: F) -> Vec<Hypothesis>
    where
        F: Fn(&[usize]) -> Vec<f64>,
    {
        self.finished_hypotheses.clear();
        self.steps_taken = 0;

        let bw = self.config.beam_width;
        let mut beams = vec![Hypothesis::new(self.config.bos_token)];

        for step in 0..self.config.max_length {
            self.steps_taken = step + 1;
            let mut candidates: Vec<Hypothesis> = Vec::new();

            for beam in &beams {
                if beam.finished {
                    candidates.push(beam.clone());
                    continue;
                }

                // Get logits for this beam's token sequence
                let mut logits = score_fn(&beam.tokens);
                assert_eq!(logits.len(), self.config.vocab_size,
                    "score_fn must return vocab_size logits");

                // Apply temperature
                logits = apply_temperature(&logits, self.config.temperature);

                // Apply top-k filtering
                logits = top_k_filter(&logits, self.config.top_k);

                // Apply top-p filtering
                logits = top_p_filter(&logits, self.config.top_p);

                // N-gram blocking
                if self.config.no_repeat_ngram > 0 {
                    for token in 0..self.config.vocab_size {
                        if beam.would_repeat_ngram(token, self.config.no_repeat_ngram) {
                            logits[token] = f64::NEG_INFINITY;
                        }
                    }
                }

                // Convert to log-probabilities
                let log_probs = log_softmax(&logits);

                // Collect top beam_width candidates from this beam
                let mut token_scores: Vec<(usize, f64)> = log_probs.iter().enumerate()
                    .map(|(i, &lp)| (i, lp))
                    .filter(|(_, lp)| lp.is_finite())
                    .collect();
                token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                token_scores.truncate(bw);

                for (token, lp) in token_scores {
                    let mut new_hyp = beam.extend(token, lp);
                    if token == self.config.eos_token {
                        new_hyp.finished = true;
                        self.finished_hypotheses.push(new_hyp.clone());
                    }
                    candidates.push(new_hyp);
                }
            }

            // Select top beam_width candidates by normalized score
            candidates.sort_by(|a, b| {
                let sa = a.normalized_score(self.config.length_penalty);
                let sb = b.normalized_score(self.config.length_penalty);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            candidates.truncate(bw);

            beams = candidates;

            // Early stopping
            if self.config.early_stopping && beams.iter().all(|b| b.finished) {
                break;
            }
        }

        // Add unfinished beams to finished set
        for beam in &beams {
            if !beam.finished {
                self.finished_hypotheses.push(beam.clone());
            }
        }

        // Sort by normalized score
        self.finished_hypotheses.sort_by(|a, b| {
            let sa = a.normalized_score(self.config.length_penalty);
            let sb = b.normalized_score(self.config.length_penalty);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        self.finished_hypotheses.clone()
    }

    /// Get the best hypothesis after search.
    pub fn best(&self) -> Option<&Hypothesis> {
        self.finished_hypotheses.first()
    }

    /// Get the n-best hypotheses.
    pub fn n_best(&self, n: usize) -> &[Hypothesis] {
        let len = self.finished_hypotheses.len().min(n);
        &self.finished_hypotheses[..len]
    }
}

impl fmt::Display for BeamSearch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BeamSearch(width={}, max_len={}, penalty={:.2}, ngram_block={}, steps={})",
            self.config.beam_width,
            self.config.max_length,
            self.config.length_penalty,
            self.config.no_repeat_ngram,
            self.steps_taken,
        )
    }
}

// ── Diversity-Promoting Beam Search ──────────────────────────────

/// Group-diverse beam search that penalizes hypotheses in the same group
/// for selecting the same tokens, encouraging diversity.
#[derive(Debug, Clone)]
pub struct DiverseBeamSearch {
    pub config: BeamSearchConfig,
    pub num_groups: usize,
    pub diversity_penalty: f64,
}

impl DiverseBeamSearch {
    pub fn new(config: BeamSearchConfig, num_groups: usize, diversity_penalty: f64) -> Self {
        Self { config, num_groups, diversity_penalty }
    }

    /// Run diverse beam search.
    pub fn search<F>(&self, score_fn: F) -> Vec<Hypothesis>
    where
        F: Fn(&[usize]) -> Vec<f64>,
    {
        let bw = self.config.beam_width;
        let beams_per_group = bw / self.num_groups;
        let mut all_results: Vec<Hypothesis> = Vec::new();

        for group in 0..self.num_groups {
            let mut beams = vec![Hypothesis::new(self.config.bos_token)];

            for _step in 0..self.config.max_length {
                let mut candidates: Vec<Hypothesis> = Vec::new();
                // Track which tokens previous groups selected (for diversity)
                let mut selected_tokens: HashMap<usize, usize> = HashMap::new();
                for prev in &all_results {
                    if let Some(&last) = prev.tokens.last() {
                        *selected_tokens.entry(last).or_insert(0) += 1;
                    }
                }

                for beam in &beams {
                    if beam.finished {
                        candidates.push(beam.clone());
                        continue;
                    }

                    let mut logits = score_fn(&beam.tokens);

                    // Penalize tokens selected by previous groups
                    for (&token, &count) in &selected_tokens {
                        if token < logits.len() {
                            logits[token] -= self.diversity_penalty * count as f64;
                        }
                    }

                    logits = apply_temperature(&logits, self.config.temperature);
                    let log_probs = log_softmax(&logits);

                    let mut token_scores: Vec<(usize, f64)> = log_probs.iter().enumerate()
                        .map(|(i, &lp)| (i, lp))
                        .filter(|(_, lp)| lp.is_finite())
                        .collect();
                    token_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
                    token_scores.truncate(beams_per_group);

                    for (token, lp) in token_scores {
                        let mut new_hyp = beam.extend(token, lp);
                        if token == self.config.eos_token {
                            new_hyp.finished = true;
                        }
                        candidates.push(new_hyp);
                    }
                }

                candidates.sort_by(|a, b| {
                    let sa = a.normalized_score(self.config.length_penalty);
                    let sb = b.normalized_score(self.config.length_penalty);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
                candidates.truncate(beams_per_group);
                beams = candidates;

                if beams.iter().all(|b| b.finished) {
                    break;
                }
            }
            all_results.extend(beams);
        }

        all_results.sort_by(|a, b| {
            let sa = a.normalized_score(self.config.length_penalty);
            let sb = b.normalized_score(self.config.length_penalty);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        all_results
    }
}

impl fmt::Display for DiverseBeamSearch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "DiverseBeamSearch(width={}, groups={}, penalty={:.2})",
            self.config.beam_width, self.num_groups, self.diversity_penalty
        )
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
        (a - b).abs() < eps
    }

    #[test]
    fn test_hypothesis_new() {
        let h = Hypothesis::new(0);
        assert_eq!(h.tokens, vec![0]);
        assert_eq!(h.log_prob, 0.0);
        assert!(!h.finished);
    }

    #[test]
    fn test_hypothesis_extend() {
        let h = Hypothesis::new(0);
        let h2 = h.extend(5, -0.5);
        assert_eq!(h2.tokens, vec![0, 5]);
        assert!(approx_eq(h2.log_prob, -0.5, 1e-12));
    }

    #[test]
    fn test_hypothesis_output_length() {
        let h = Hypothesis::new(0);
        assert_eq!(h.output_length(), 0);
        let h2 = h.extend(1, -0.1).extend(2, -0.2);
        assert_eq!(h2.output_length(), 2);
    }

    #[test]
    fn test_hypothesis_normalized_score() {
        let h = Hypothesis::new(0).extend(1, -2.0).extend(2, -1.0);
        let score_no_penalty = h.normalized_score(0.0);
        assert!(approx_eq(score_no_penalty, -3.0, 1e-10));
        let score_with_penalty = h.normalized_score(1.0);
        assert!(score_with_penalty > -3.0); // Length penalty reduces magnitude
    }

    #[test]
    fn test_ngram_blocking_bigram() {
        let h = Hypothesis::new(0).extend(1, -0.1).extend(2, -0.1);
        // Tokens: [0, 1, 2]. Adding 1 would create bigram (2,1) -- not a repeat
        assert!(!h.would_repeat_ngram(1, 2));
        // Build: [0, 1, 2] -> try adding token that creates repeated bigram
        let h2 = h.extend(1, -0.1); // Tokens: [0, 1, 2, 1]
        // Adding 2 would create bigram (1,2) which exists at positions 0-1
        assert!(h2.would_repeat_ngram(2, 2));
    }

    #[test]
    fn test_ngram_blocking_trigram() {
        let h = Hypothesis::new(0).extend(1, -0.1).extend(2, -0.1).extend(1, -0.1);
        // Tokens: [0, 1, 2, 1]. Adding 2 creates trigram (2,1,2) -- check for (1,2,1) at start: no
        // Need to check: trigram (2, 1, 2) vs existing trigrams: (0,1,2), (1,2,1)
        assert!(!h.would_repeat_ngram(2, 3));
    }

    #[test]
    fn test_hypothesis_display() {
        let h = Hypothesis::new(0).extend(1, -0.5);
        let s = format!("{}", h);
        assert!(s.contains("len=2"));
        assert!(s.contains("log_p="));
    }

    #[test]
    fn test_config_defaults() {
        let cfg = BeamSearchConfig::new(100);
        assert_eq!(cfg.beam_width, 4);
        assert_eq!(cfg.max_length, 128);
        assert_eq!(cfg.vocab_size, 100);
        assert_eq!(cfg.eos_token, 1);
        assert_eq!(cfg.bos_token, 0);
    }

    #[test]
    fn test_config_builder() {
        let cfg = BeamSearchConfig::new(1000)
            .with_beam_width(8)
            .with_max_length(256)
            .with_length_penalty(1.0)
            .with_no_repeat_ngram(3)
            .with_top_k(50)
            .with_top_p(0.9)
            .with_temperature(0.7)
            .with_eos_token(2)
            .with_bos_token(1)
            .with_early_stopping(false);
        assert_eq!(cfg.beam_width, 8);
        assert_eq!(cfg.no_repeat_ngram, 3);
        assert_eq!(cfg.top_k, 50);
    }

    #[test]
    fn test_log_softmax_sum() {
        let logits = vec![1.0, 2.0, 3.0];
        let lp = log_softmax(&logits);
        // exp(log_softmax) should sum to 1
        let sum: f64 = lp.iter().map(|l| l.exp()).sum();
        assert!(approx_eq(sum, 1.0, 1e-10));
    }

    #[test]
    fn test_top_k_filter_basic() {
        let logits = vec![1.0, 5.0, 3.0, 2.0, 4.0];
        let filtered = top_k_filter(&logits, 2);
        assert_eq!(filtered[1], 5.0);
        assert_eq!(filtered[4], 4.0);
        assert!(filtered[0].is_infinite());
    }

    #[test]
    fn test_top_p_filter_basic() {
        let logits = vec![10.0, 1.0, 0.0, -10.0];
        let filtered = top_p_filter(&logits, 0.9);
        assert_eq!(filtered[0], 10.0);
    }

    #[test]
    fn test_beam_search_basic() {
        let cfg = BeamSearchConfig::new(5)
            .with_beam_width(2)
            .with_max_length(3)
            .with_early_stopping(false);
        let mut bs = BeamSearch::new(cfg);
        // Simple scoring: prefer token 2
        let results = bs.search(|_tokens| {
            vec![0.0, -1.0, 5.0, -1.0, -1.0]
        });
        assert!(!results.is_empty());
        let best = &results[0];
        // Should have generated mostly token 2
        assert!(best.tokens.contains(&2));
    }

    #[test]
    fn test_beam_search_eos() {
        let cfg = BeamSearchConfig::new(5)
            .with_beam_width(2)
            .with_max_length(10)
            .with_eos_token(1);
        let mut bs = BeamSearch::new(cfg);
        // Always output EOS immediately
        let results = bs.search(|_tokens| {
            vec![-10.0, 10.0, -10.0, -10.0, -10.0]
        });
        let best = results.first().unwrap();
        assert!(best.finished);
        assert_eq!(best.tokens.len(), 2); // BOS + EOS
    }

    #[test]
    fn test_beam_search_n_best() {
        let cfg = BeamSearchConfig::new(5)
            .with_beam_width(3)
            .with_max_length(2);
        let mut bs = BeamSearch::new(cfg);
        bs.search(|_| {
            vec![0.0, 1.0, 2.0, 1.5, 0.5]
        });
        let top2 = bs.n_best(2);
        assert!(top2.len() <= 2);
    }

    #[test]
    fn test_beam_search_steps_tracked() {
        let cfg = BeamSearchConfig::new(5)
            .with_beam_width(2)
            .with_max_length(5)
            .with_early_stopping(false);
        let mut bs = BeamSearch::new(cfg);
        bs.search(|_| vec![1.0; 5]);
        assert!(bs.steps_taken > 0);
        assert!(bs.steps_taken <= 5);
    }

    #[test]
    fn test_beam_search_display() {
        let cfg = BeamSearchConfig::new(100)
            .with_beam_width(8)
            .with_no_repeat_ngram(3);
        let bs = BeamSearch::new(cfg);
        let s = format!("{}", bs);
        assert!(s.contains("width=8"));
        assert!(s.contains("ngram_block=3"));
    }

    #[test]
    fn test_diverse_beam_search() {
        let cfg = BeamSearchConfig::new(5)
            .with_beam_width(4)
            .with_max_length(3);
        let dbs = DiverseBeamSearch::new(cfg, 2, 1.0);
        let results = dbs.search(|_| vec![1.0, 2.0, 3.0, 2.5, 1.5]);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_diverse_beam_search_display() {
        let cfg = BeamSearchConfig::new(100).with_beam_width(8);
        let dbs = DiverseBeamSearch::new(cfg, 4, 0.5);
        let s = format!("{}", dbs);
        assert!(s.contains("groups=4"));
    }
}
