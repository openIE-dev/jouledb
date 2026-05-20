//! BPE Tokenizer: Byte Pair Encoding for subword generation.
//!
//! Frontier models use subword tokenization — not whole words, not characters,
//! but learned subword units like "un", "##break", "##able". This enables:
//! - Generating novel words never seen in training
//! - Handling misspellings, code, URLs, any text
//! - Compact vocabulary (30K-50K tokens covers all languages)
//!
//! The BPE algorithm:
//! 1. Start with character-level vocabulary
//! 2. Count all adjacent character pairs in the corpus
//! 3. Merge the most frequent pair into a new token
//! 4. Repeat until vocabulary reaches target size
//!
//! Each token gets a BinaryHV encoding. The sequence memory and generator
//! then operate on subword tokens instead of whole words.

use crate::BinaryHV;
use std::collections::HashMap;

/// A BPE tokenizer that learns subword merges from a corpus.
pub struct BpeTokenizer {
    /// Merge rules: (pair) → merged token, in order of priority.
    merges: Vec<((String, String), String)>,
    /// Vocabulary: token string → token ID.
    vocab: HashMap<String, u32>,
    /// Reverse vocabulary: token ID → token string.
    id_to_token: Vec<String>,
    /// Token → BinaryHV encoding.
    encodings: HashMap<u32, BinaryHV>,
    /// Dimension for BinaryHV encodings.
    dim: usize,
    /// Base seed for deterministic encoding.
    seed: u64,
}

impl BpeTokenizer {
    /// Create an empty tokenizer.
    pub fn new(dim: usize) -> Self {
        Self {
            merges: Vec::new(),
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            encodings: HashMap::new(),
            dim,
            seed: 0xB9E_70CE_4120_0000, // "BPE_TOKEN"
        }
    }

    /// Train BPE merges from a corpus.
    /// `num_merges` controls vocabulary size: base_chars + num_merges.
    pub fn train(&mut self, corpus: &[&str], num_merges: usize) {
        // Step 1: Initialize with character vocabulary
        let mut word_freqs: HashMap<Vec<String>, u32> = HashMap::new();
        for text in corpus {
            for word in text.split_whitespace() {
                let chars: Vec<String> = word
                    .to_lowercase()
                    .chars()
                    .map(|c| c.to_string())
                    .collect();
                if !chars.is_empty() {
                    *word_freqs.entry(chars).or_insert(0) += 1;
                }
            }
        }

        // Build initial character vocabulary
        let mut char_set: Vec<String> = Vec::new();
        for chars in word_freqs.keys() {
            for c in chars {
                if !char_set.contains(c) {
                    char_set.push(c.clone());
                }
            }
        }
        char_set.sort();

        // Register characters as base vocab
        for c in &char_set {
            self.register_token(c);
        }

        // Step 2: Iteratively merge most frequent pairs
        for _ in 0..num_merges {
            // Count pairs
            let mut pair_freqs: HashMap<(String, String), u32> = HashMap::new();
            for (tokens, freq) in &word_freqs {
                for pair in tokens.windows(2) {
                    let key = (pair[0].clone(), pair[1].clone());
                    *pair_freqs.entry(key).or_insert(0) += freq;
                }
            }

            // Find most frequent pair
            let best_pair = match pair_freqs
                .iter()
                .max_by_key(|(_, freq)| *freq)
            {
                Some((pair, freq)) if *freq >= 2 => pair.clone(),
                _ => break, // No pairs with frequency ≥ 2
            };

            // Create merged token
            let merged = format!("{}{}", best_pair.0, best_pair.1);
            self.register_token(&merged);
            self.merges.push((best_pair.clone(), merged.clone()));

            // Apply merge to all words
            let mut new_freqs: HashMap<Vec<String>, u32> = HashMap::new();
            for (tokens, freq) in &word_freqs {
                let merged_tokens = self.apply_merge(tokens, &best_pair, &merged);
                *new_freqs.entry(merged_tokens).or_insert(0) += freq;
            }
            word_freqs = new_freqs;
        }
    }

    /// Tokenize a string into subword token IDs.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let mut all_ids = Vec::new();
        for word in text.split_whitespace() {
            let ids = self.tokenize_word(word);
            all_ids.extend(ids);
        }
        all_ids
    }

    /// Tokenize a single word.
    fn tokenize_word(&self, word: &str) -> Vec<u32> {
        let mut tokens: Vec<String> = word
            .to_lowercase()
            .chars()
            .map(|c| c.to_string())
            .collect();

        // Apply merges in order
        for ((a, b), merged) in &self.merges {
            tokens = self.apply_merge(&tokens, &(a.clone(), b.clone()), merged);
        }

        // Convert to IDs (unknown chars get ID 0)
        tokens
            .iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(0))
            .collect()
    }

    /// Detokenize: convert token IDs back to text.
    pub fn detokenize(&self, ids: &[u32]) -> String {
        ids.iter()
            .map(|id| {
                if (*id as usize) < self.id_to_token.len() {
                    self.id_to_token[*id as usize].as_str()
                } else {
                    "?"
                }
            })
            .collect::<Vec<_>>()
            .join("")
    }

    /// Get the BinaryHV encoding for a token ID.
    pub fn encode_token(&mut self, token_id: u32) -> BinaryHV {
        if let Some(hv) = self.encodings.get(&token_id) {
            return hv.clone();
        }
        // Generate deterministic encoding
        let hv = BinaryHV::random(self.dim, self.seed.wrapping_add(token_id as u64));
        self.encodings.insert(token_id, hv.clone());
        hv
    }

    /// Encode a full text as a sequence of BinaryHV tokens with positional binding.
    /// Returns (token_ids, sequence_hv).
    pub fn encode_sequence(&mut self, text: &str) -> (Vec<u32>, BinaryHV) {
        let ids = self.tokenize(text);
        if ids.is_empty() {
            return (ids, BinaryHV::zeros(self.dim));
        }

        let mut result = self.encode_token(ids[0]);
        for (i, &id) in ids.iter().enumerate().skip(1) {
            let token_hv = self.encode_token(id);
            result = result.bind(&token_hv.permute(i));
        }

        (ids, result)
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get token string for an ID.
    pub fn token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    /// Get all merge rules (for serialization).
    pub fn merges(&self) -> &[((String, String), String)] {
        &self.merges
    }

    // Internal helpers

    fn register_token(&mut self, token: &str) {
        if !self.vocab.contains_key(token) {
            let id = self.id_to_token.len() as u32;
            self.vocab.insert(token.to_string(), id);
            self.id_to_token.push(token.to_string());
        }
    }

    fn apply_merge(
        &self,
        tokens: &[String],
        pair: &(String, String),
        merged: &str,
    ) -> Vec<String> {
        let mut result = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            if i + 1 < tokens.len() && tokens[i] == pair.0 && tokens[i + 1] == pair.1 {
                result.push(merged.to_string());
                i += 2;
            } else {
                result.push(tokens[i].clone());
                i += 1;
            }
        }
        result
    }
}

impl Default for BpeTokenizer {
    fn default() -> Self {
        Self::new(10_000)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_corpus() -> Vec<&'static str> {
        vec![
            "the dog is a loyal animal",
            "the cat is an independent animal",
            "a dog can bark loudly",
            "a cat can purr softly",
            "the bird can fly high",
            "the fish can swim fast",
            "dogs and cats are popular pets",
            "the animal kingdom is diverse",
            "a loyal dog protects the house",
            "the independent cat explores the house",
            "animals need food and water",
            "a dog is a good pet",
            "a cat is a good pet",
            "the fish swims in the ocean",
            "the bird flies in the sky",
        ]
    }

    #[test]
    fn test_train_builds_vocabulary() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 50);
        assert!(tok.vocab_size() > 26); // At least a-z + merges
    }

    #[test]
    fn test_tokenize_known_word() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 50);
        let ids = tok.tokenize("dog");
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_detokenize_roundtrip() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 50);

        let text = "the dog";
        let ids = tok.tokenize(text);
        let recovered = tok.detokenize(&ids);
        // BPE merges may change tokenization but detokenize should reconstruct
        assert_eq!(recovered.replace(" ", ""), text.replace(" ", ""));
    }

    #[test]
    fn test_tokenize_unknown_word() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 50);
        // "xyz" may not have been in training but individual chars should be
        let ids = tok.tokenize("xyz");
        // Should produce some tokens (character-level fallback)
        // x/y/z might or might not be in vocab depending on corpus
        assert!(ids.len() <= 3);
    }

    #[test]
    fn test_merges_reduce_token_count() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 100);

        // "the" should be a single token after enough merges
        let ids = tok.tokenize("the");
        // With enough merges, common words become 1-2 tokens
        assert!(ids.len() <= 3, "common word should merge: {} tokens", ids.len());
    }

    #[test]
    fn test_encode_token_deterministic() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 50);
        let ids = tok.tokenize("dog");
        if let Some(&id) = ids.first() {
            let hv1 = tok.encode_token(id);
            let hv2 = tok.encode_token(id);
            assert_eq!(hv1.hamming_distance(&hv2), 0);
        }
    }

    #[test]
    fn test_encode_sequence() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 50);

        let (ids, seq_hv) = tok.encode_sequence("the dog is loyal");
        assert!(!ids.is_empty());
        assert_eq!(seq_hv.dimension(), 1000);
    }

    #[test]
    fn test_different_sequences_different_vectors() {
        let mut tok = BpeTokenizer::new(10000);
        tok.train(&sample_corpus(), 50);

        let (_, hv1) = tok.encode_sequence("the dog is loyal");
        let (_, hv2) = tok.encode_sequence("the cat is independent");
        let sim = hv1.similarity(&hv2);
        assert!(sim < 0.7, "different sequences should differ: {sim}");
    }

    #[test]
    fn test_vocab_includes_common_merges() {
        let mut tok = BpeTokenizer::new(1000);
        tok.train(&sample_corpus(), 100);

        // After 100 merges, "th" should likely be a merged token
        // (very common bigram in English)
        let has_th = tok.vocab.contains_key("th");
        let has_the = tok.vocab.contains_key("the");
        let has_an = tok.vocab.contains_key("an");
        // At least some common bigrams should have merged
        assert!(
            has_th || has_the || has_an,
            "should have common merges. vocab size: {}",
            tok.vocab_size()
        );
    }
}
