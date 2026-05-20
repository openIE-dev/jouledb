//! Byte-pair encoding (BPE) and WordPiece tokenizers.
//!
//! Vocabulary mapping, merge rules, encode/decode, special tokens,
//! truncation, and padding. Pure Rust — no sentencepiece or HF deps.

use std::collections::HashMap;

// ── Special tokens ──────────────────────────────────────────────

/// Well-known special token strings.
pub const PAD_TOKEN: &str = "[PAD]";
pub const UNK_TOKEN: &str = "[UNK]";
pub const CLS_TOKEN: &str = "[CLS]";
pub const SEP_TOKEN: &str = "[SEP]";
pub const MASK_TOKEN: &str = "[MASK]";

/// All special tokens.
pub const SPECIAL_TOKENS: [&str; 5] = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN];

// ── BPE tokenizer ───────────────────────────────────────────────

/// A byte-pair encoding tokenizer.
#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    /// Token string → ID.
    pub vocab: HashMap<String, u32>,
    /// ID → token string (inverse).
    pub id_to_token: HashMap<u32, String>,
    /// Ordered merge rules: (pair_a, pair_b) → merged token.
    pub merges: Vec<(String, String)>,
    /// Precomputed merge priority: (pair_a, pair_b) → rank (lower = higher priority).
    merge_rank: HashMap<(String, String), usize>,
}

impl BpeTokenizer {
    /// Create a BPE tokenizer from vocabulary and merge rules.
    ///
    /// `vocab` maps token strings to IDs. `merges` is an ordered list
    /// of merge rules (most frequent first).
    pub fn new(vocab: HashMap<String, u32>, merges: Vec<(String, String)>) -> Self {
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        let merge_rank: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();
        Self { vocab, id_to_token, merges, merge_rank }
    }

    /// Build a simple tokenizer from a list of tokens and merge rules.
    pub fn from_tokens(tokens: &[&str], merges: &[(&str, &str)]) -> Self {
        let mut vocab = HashMap::new();
        // Insert special tokens first.
        for (i, st) in SPECIAL_TOKENS.iter().enumerate() {
            vocab.insert(st.to_string(), i as u32);
        }
        let base = SPECIAL_TOKENS.len() as u32;
        for (i, tok) in tokens.iter().enumerate() {
            vocab.entry(tok.to_string()).or_insert(base + i as u32);
        }
        let merge_vec: Vec<(String, String)> = merges
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        Self::new(vocab, merge_vec)
    }

    /// Encode text into token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        // Split into words by whitespace, then tokenize each.
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            let word_ids = self.encode_word(word);
            ids.extend(word_ids);
        }
        ids
    }

    fn encode_word(&self, word: &str) -> Vec<u32> {
        // Start with characters.
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        // Iteratively apply the highest-priority merge.
        loop {
            if tokens.len() < 2 {
                break;
            }
            let mut best_rank = usize::MAX;
            let mut best_idx = None;

            for i in 0..tokens.len() - 1 {
                let pair = (tokens[i].clone(), tokens[i + 1].clone());
                if let Some(rank) = self.merge_rank.get(&pair) {
                    if *rank < best_rank {
                        best_rank = *rank;
                        best_idx = Some(i);
                    }
                }
            }

            match best_idx {
                Some(idx) => {
                    let merged = format!("{}{}", tokens[idx], tokens[idx + 1]);
                    tokens[idx] = merged;
                    tokens.remove(idx + 1);
                }
                None => break,
            }
        }

        // Map tokens to IDs.
        tokens
            .into_iter()
            .map(|t| {
                self.vocab
                    .get(&t)
                    .copied()
                    .unwrap_or_else(|| self.vocab.get(UNK_TOKEN).copied().unwrap_or(0))
            })
            .collect()
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|id| self.id_to_token.get(id))
            .cloned()
            .collect::<Vec<_>>()
            .join("")
    }

    /// Look up a token ID. Returns UNK id if not found.
    pub fn token_to_id(&self, token: &str) -> u32 {
        self.vocab
            .get(token)
            .copied()
            .unwrap_or_else(|| self.vocab.get(UNK_TOKEN).copied().unwrap_or(0))
    }

    /// Look up an ID.
    pub fn id_to_token_str(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(|s| s.as_str())
    }

    /// Vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }
}

// ── WordPiece tokenizer ─────────────────────────────────────────

/// A WordPiece tokenizer (## prefix for continuation tokens).
#[derive(Debug, Clone)]
pub struct WordPieceTokenizer {
    pub vocab: HashMap<String, u32>,
    pub id_to_token: HashMap<u32, String>,
    /// Maximum length of a token to try.
    pub max_word_len: usize,
    /// Continuation prefix (default "##").
    pub prefix: String,
}

impl WordPieceTokenizer {
    pub fn new(vocab: HashMap<String, u32>) -> Self {
        let id_to_token: HashMap<u32, String> =
            vocab.iter().map(|(k, v)| (*v, k.clone())).collect();
        Self {
            vocab,
            id_to_token,
            max_word_len: 200,
            prefix: "##".to_string(),
        }
    }

    /// Build from a list of tokens (auto-assigns IDs, includes specials).
    pub fn from_tokens(tokens: &[&str]) -> Self {
        let mut vocab = HashMap::new();
        for (i, st) in SPECIAL_TOKENS.iter().enumerate() {
            vocab.insert(st.to_string(), i as u32);
        }
        let base = SPECIAL_TOKENS.len() as u32;
        for (i, tok) in tokens.iter().enumerate() {
            vocab.entry(tok.to_string()).or_insert(base + i as u32);
        }
        Self::new(vocab)
    }

    /// Encode a single word using greedy longest-match WordPiece.
    pub fn encode_word(&self, word: &str) -> Vec<u32> {
        if word.len() > self.max_word_len {
            return vec![self.vocab.get(UNK_TOKEN).copied().unwrap_or(0)];
        }

        let mut ids = Vec::new();
        let mut start = 0;
        let chars: Vec<char> = word.chars().collect();

        while start < chars.len() {
            let mut end = chars.len();
            let mut found = false;

            while start < end {
                let substr: String = chars[start..end].iter().collect();
                let candidate = if start > 0 {
                    format!("{}{}", self.prefix, substr)
                } else {
                    substr
                };

                if let Some(id) = self.vocab.get(&candidate) {
                    ids.push(*id);
                    found = true;
                    start = end;
                    break;
                }
                end -= 1;
            }

            if !found {
                ids.push(self.vocab.get(UNK_TOKEN).copied().unwrap_or(0));
                start += 1;
            }
        }
        ids
    }

    /// Encode text (split by whitespace, then WordPiece each word).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for word in text.split_whitespace() {
            ids.extend(self.encode_word(word));
        }
        ids
    }

    /// Decode IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();
        for id in ids {
            if let Some(tok) = self.id_to_token.get(id) {
                if let Some(stripped) = tok.strip_prefix("##") {
                    result.push_str(stripped);
                } else {
                    if !result.is_empty() {
                        result.push(' ');
                    }
                    result.push_str(tok);
                }
            }
        }
        result
    }
}

// ── Truncation and padding ──────────────────────────────────────

/// Truncate a token ID sequence to `max_len`.
pub fn truncate(ids: &[u32], max_len: usize) -> Vec<u32> {
    if ids.len() <= max_len {
        ids.to_vec()
    } else {
        ids[..max_len].to_vec()
    }
}

/// Pad a token ID sequence to `max_len` with `pad_id`.
pub fn pad(ids: &[u32], max_len: usize, pad_id: u32) -> Vec<u32> {
    let mut result = ids.to_vec();
    while result.len() < max_len {
        result.push(pad_id);
    }
    result
}

/// Truncate then pad to exactly `max_len`.
pub fn truncate_and_pad(ids: &[u32], max_len: usize, pad_id: u32) -> Vec<u32> {
    pad(&truncate(ids, max_len), max_len, pad_id)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_bpe() -> BpeTokenizer {
        // Vocab: individual chars + merged tokens.
        BpeTokenizer::from_tokens(
            &["h", "e", "l", "o", "w", "r", "d", "he", "ll", "lo", "hell", "hello", "wo", "wor", "world"],
            &[
                ("h", "e"),      // he
                ("l", "l"),      // ll
                ("l", "o"),      // lo
                ("he", "ll"),    // hell
                ("hell", "o"),   // hello
                ("w", "o"),      // wo
                ("wo", "r"),     // wor
                ("wor", "ld"),   // world — but "ld" not in vocab, so this won't fire
            ],
        )
    }

    #[test]
    fn test_bpe_encode_single_word() {
        let tok = make_bpe();
        let ids = tok.encode("hello");
        // Should merge to "hello" token
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_bpe_encode_unknown() {
        let tok = make_bpe();
        let ids = tok.encode("x");
        // 'x' not in vocab → UNK
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], tok.token_to_id(UNK_TOKEN));
    }

    #[test]
    fn test_bpe_vocab_size() {
        let tok = make_bpe();
        // 5 specials + 15 tokens = 20
        assert_eq!(tok.vocab_size(), 20);
    }

    #[test]
    fn test_bpe_special_tokens() {
        let tok = make_bpe();
        assert!(tok.vocab.contains_key(PAD_TOKEN));
        assert!(tok.vocab.contains_key(UNK_TOKEN));
        assert!(tok.vocab.contains_key(CLS_TOKEN));
        assert!(tok.vocab.contains_key(SEP_TOKEN));
        assert!(tok.vocab.contains_key(MASK_TOKEN));
    }

    #[test]
    fn test_bpe_empty_input() {
        let tok = make_bpe();
        assert!(tok.encode("").is_empty());
    }

    #[test]
    fn test_wordpiece_basic() {
        let tok = WordPieceTokenizer::from_tokens(&[
            "hello", "world", "##ing", "##ed", "play", "un",
        ]);
        let ids = tok.encode("hello");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "hello");
    }

    #[test]
    fn test_wordpiece_continuation() {
        let tok = WordPieceTokenizer::from_tokens(&[
            "play", "##ing", "##ed", "##s",
        ]);
        let ids = tok.encode("playing");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "playing");
    }

    #[test]
    fn test_wordpiece_unknown() {
        let tok = WordPieceTokenizer::from_tokens(&["cat", "dog"]);
        let ids = tok.encode("xyz");
        // All chars unknown
        for id in &ids {
            assert_eq!(*id, tok.vocab.get(UNK_TOKEN).copied().unwrap());
        }
    }

    #[test]
    fn test_truncate() {
        let ids = vec![1, 2, 3, 4, 5];
        assert_eq!(truncate(&ids, 3), vec![1, 2, 3]);
        assert_eq!(truncate(&ids, 10), ids);
    }

    #[test]
    fn test_pad() {
        let ids = vec![1, 2, 3];
        assert_eq!(pad(&ids, 5, 0), vec![1, 2, 3, 0, 0]);
        assert_eq!(pad(&ids, 2, 0), ids); // no padding needed
    }

    #[test]
    fn test_truncate_and_pad() {
        let ids = vec![1, 2, 3, 4, 5];
        let result = truncate_and_pad(&ids, 3, 0);
        assert_eq!(result, vec![1, 2, 3]);

        let short = vec![1, 2];
        let result2 = truncate_and_pad(&short, 4, 99);
        assert_eq!(result2, vec![1, 2, 99, 99]);
    }

    #[test]
    fn test_bpe_id_to_token() {
        let tok = make_bpe();
        let id = tok.token_to_id("hello");
        let back = tok.id_to_token_str(id);
        assert_eq!(back, Some("hello"));
    }
}
