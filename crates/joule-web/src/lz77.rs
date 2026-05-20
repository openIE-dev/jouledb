//! LZ77 compression — sliding window with hash-accelerated matching.
//!
//! Replaces JavaScript LZ compression libraries with a pure Rust
//! implementation. Produces a token stream of literals and
//! back-reference matches that can be replayed to reconstruct
//! the original data.

// ── Constants ────────────────────────────────────────────────────────

/// Default sliding window size (4 KB).
pub const DEFAULT_WINDOW_SIZE: usize = 4096;

/// Minimum match length.
const MIN_MATCH: usize = 3;

/// Maximum match length.
const MAX_MATCH: usize = 258;

/// Hash table size.
const HASH_SIZE: usize = 1 << 14;
const HASH_MASK: usize = HASH_SIZE - 1;

// ── Token ────────────────────────────────────────────────────────────

/// A single LZ77 token — either a literal byte or a back-reference match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    /// A literal byte that could not be compressed.
    Literal(u8),
    /// A match: copy `length` bytes from `offset` positions back.
    Match { offset: usize, length: usize },
}

// ── Encoder Configuration ────────────────────────────────────────────

/// Configuration for the LZ77 encoder.
#[derive(Debug, Clone)]
pub struct Lz77Config {
    /// Maximum distance for back-references.
    pub window_size: usize,
    /// Maximum hash chain depth to search.
    pub max_chain: usize,
    /// Enable lazy matching (check if next position has a better match).
    pub lazy_matching: bool,
}

impl Default for Lz77Config {
    fn default() -> Self {
        Self {
            window_size: DEFAULT_WINDOW_SIZE,
            max_chain: 64,
            lazy_matching: false,
        }
    }
}

impl Lz77Config {
    /// Create a config with lazy matching enabled.
    pub fn lazy() -> Self {
        Self {
            lazy_matching: true,
            ..Default::default()
        }
    }

    /// Set window size.
    pub fn with_window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set max chain depth.
    pub fn with_max_chain(mut self, depth: usize) -> Self {
        self.max_chain = depth;
        self
    }
}

// ── Hash ─────────────────────────────────────────────────────────────

/// Hash 3 bytes starting at `pos`.
fn hash3(data: &[u8], pos: usize) -> usize {
    let h = (data[pos] as usize).wrapping_mul(31)
        ^ (data[pos + 1] as usize).wrapping_mul(37)
        ^ (data[pos + 2] as usize);
    h & HASH_MASK
}

// ── Find Best Match ──────────────────────────────────────────────────

fn find_best_match(
    data: &[u8],
    pos: usize,
    head: &[u32],
    prev: &[u32],
    config: &Lz77Config,
) -> Option<(usize, usize)> {
    if pos + 2 >= data.len() {
        return None;
    }

    let h = hash3(data, pos);
    let mut chain = head[h];
    let mut best_len = MIN_MATCH - 1;
    let mut best_offset = 0usize;
    let mut chain_count = 0usize;

    while chain > 0 && chain_count < config.max_chain {
        let candidate = (chain - 1) as usize;
        let dist = pos - candidate;
        if dist > config.window_size || candidate >= pos {
            break;
        }

        let max_len = MAX_MATCH.min(data.len() - pos);
        let mut len = 0;
        while len < max_len && data[candidate + len] == data[pos + len] {
            len += 1;
        }

        if len > best_len {
            best_len = len;
            best_offset = dist;
            if len >= MAX_MATCH {
                break;
            }
        }

        chain = prev[candidate];
        chain_count += 1;
    }

    if best_len >= MIN_MATCH {
        Some((best_offset, best_len))
    } else {
        None
    }
}

// ── Encode ───────────────────────────────────────────────────────────

/// Encode input data into LZ77 tokens.
pub fn encode(data: &[u8]) -> Vec<Token> {
    encode_with_config(data, &Lz77Config::default())
}

/// Encode input data into LZ77 tokens with custom configuration.
pub fn encode_with_config(data: &[u8], config: &Lz77Config) -> Vec<Token> {
    let mut tokens = Vec::new();
    if data.is_empty() {
        return tokens;
    }

    let mut head = vec![0u32; HASH_SIZE];
    let mut prev = vec![0u32; data.len()];
    let mut pos = 0usize;

    while pos < data.len() {
        // Find match BEFORE inserting current position into hash.
        let best = find_best_match(data, pos, &head, &prev, config);

        // Insert hash for current position.
        if pos + 2 < data.len() {
            let h = hash3(data, pos);
            prev[pos] = head[h];
            head[h] = (pos + 1) as u32;
        }

        if let Some((offset, length)) = best {
            if config.lazy_matching && pos + 1 < data.len() && pos + 3 < data.len() {
                // Insert hash for next position to check lazy match.
                let next_pos = pos + 1;
                if next_pos + 2 < data.len() {
                    let h2 = hash3(data, next_pos);
                    prev[next_pos] = head[h2];
                    head[h2] = (next_pos + 1) as u32;
                }

                let lazy = find_best_match(data, next_pos, &head, &prev, config);
                if let Some((_, lazy_len)) = lazy {
                    if lazy_len > length + 1 {
                        // Lazy match is better — emit current byte as literal.
                        tokens.push(Token::Literal(data[pos]));
                        pos += 1;
                        continue;
                    }
                }
            }

            tokens.push(Token::Match { offset, length });
            // Insert hashes for positions within the match.
            for i in 1..length {
                let p = pos + i;
                if p + 2 < data.len() {
                    let h = hash3(data, p);
                    prev[p] = head[h];
                    head[h] = (p + 1) as u32;
                }
            }
            pos += length;
        } else {
            tokens.push(Token::Literal(data[pos]));
            pos += 1;
        }
    }

    tokens
}

// ── Decode ───────────────────────────────────────────────────────────

/// Decode LZ77 tokens back into the original data.
pub fn decode(tokens: &[Token]) -> Result<Vec<u8>, Lz77Error> {
    let mut output = Vec::new();

    for token in tokens {
        match token {
            Token::Literal(b) => {
                output.push(*b);
            }
            Token::Match { offset, length } => {
                if *offset == 0 || *offset > output.len() {
                    return Err(Lz77Error::InvalidBackReference {
                        offset: *offset,
                        output_size: output.len(),
                    });
                }
                let start = output.len() - offset;
                for i in 0..*length {
                    let b = output[start + (i % *offset)];
                    output.push(b);
                }
            }
        }
    }

    Ok(output)
}

// ── Errors ───────────────────────────────────────────────────────────

/// Errors produced during LZ77 operations.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Lz77Error {
    #[error("invalid back-reference: offset {offset} exceeds output size {output_size}")]
    InvalidBackReference { offset: usize, output_size: usize },
}

// ── Token Serialization ──────────────────────────────────────────────

/// Serialize tokens to a byte stream for storage.
/// Format: tag byte (0=literal, 1=match), then data.
pub fn serialize_tokens(tokens: &[Token]) -> Vec<u8> {
    let mut out = Vec::new();
    for token in tokens {
        match token {
            Token::Literal(b) => {
                out.push(0);
                out.push(*b);
            }
            Token::Match { offset, length } => {
                out.push(1);
                out.extend_from_slice(&(*offset as u32).to_le_bytes());
                out.extend_from_slice(&(*length as u32).to_le_bytes());
            }
        }
    }
    out
}

/// Deserialize tokens from a byte stream.
pub fn deserialize_tokens(data: &[u8]) -> Result<Vec<Token>, Lz77Error> {
    let mut tokens = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        match data[pos] {
            0 => {
                if pos + 1 >= data.len() {
                    break;
                }
                tokens.push(Token::Literal(data[pos + 1]));
                pos += 2;
            }
            1 => {
                if pos + 8 >= data.len() {
                    break;
                }
                let offset = u32::from_le_bytes([
                    data[pos + 1], data[pos + 2], data[pos + 3], data[pos + 4],
                ]) as usize;
                let length = u32::from_le_bytes([
                    data[pos + 5], data[pos + 6], data[pos + 7], data[pos + 8],
                ]) as usize;
                tokens.push(Token::Match { offset, length });
                pos += 9;
            }
            _ => {
                pos += 1;
            }
        }
    }
    Ok(tokens)
}

/// Count the number of literal and match tokens.
pub fn token_stats(tokens: &[Token]) -> (usize, usize) {
    let mut literals = 0;
    let mut matches = 0;
    for token in tokens {
        match token {
            Token::Literal(_) => literals += 1,
            Token::Match { .. } => matches += 1,
        }
    }
    (literals, matches)
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_empty() {
        let tokens = encode(b"");
        assert!(tokens.is_empty());
        let decoded = decode(&tokens).unwrap();
        assert!(decoded.is_empty());
    }

    #[test]
    fn roundtrip_single_byte() {
        let data = b"X";
        let tokens = encode(data);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_no_matches() {
        let data = b"abcdefgh";
        let tokens = encode(data);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_with_matches() {
        let data = b"abcabcabcabc";
        let tokens = encode(data);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
        // Should have at least one Match token.
        let (_, matches) = token_stats(&tokens);
        assert!(matches > 0);
    }

    #[test]
    fn roundtrip_repeated_byte() {
        let data = vec![b'A'; 1000];
        let tokens = encode(&data);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn roundtrip_binary_data() {
        let data: Vec<u8> = (0..512).map(|i| (i * 7 % 256) as u8).collect();
        let tokens = encode(&data);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn lazy_matching() {
        let data = b"abcXabcYabcXabcY";
        let config = Lz77Config::lazy();
        let tokens = encode_with_config(data, &config);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn custom_window_size() {
        let data = b"hello world hello world hello";
        let config = Lz77Config::default().with_window_size(16);
        let tokens = encode_with_config(data, &config);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn token_serialization_roundtrip() {
        let data = b"abcabcabc";
        let tokens = encode(data);
        let serialized = serialize_tokens(&tokens);
        let deserialized = deserialize_tokens(&serialized).unwrap();
        assert_eq!(tokens, deserialized);
    }

    #[test]
    fn invalid_back_reference() {
        let tokens = vec![Token::Match { offset: 10, length: 5 }];
        let result = decode(&tokens);
        assert!(result.is_err());
    }

    #[test]
    fn match_within_itself() {
        // A match where the copied region overlaps the source.
        let tokens = vec![
            Token::Literal(b'a'),
            Token::Match { offset: 1, length: 10 },
        ];
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, vec![b'a'; 11]);
    }

    #[test]
    fn token_stats_correct() {
        let tokens = vec![
            Token::Literal(b'a'),
            Token::Literal(b'b'),
            Token::Match { offset: 2, length: 3 },
        ];
        let (lits, matches) = token_stats(&tokens);
        assert_eq!(lits, 2);
        assert_eq!(matches, 1);
    }

    #[test]
    fn roundtrip_lorem() {
        let data = b"Lorem ipsum dolor sit amet, consectetur adipiscing elit. \
                      Lorem ipsum dolor sit amet, consectetur adipiscing elit.";
        let tokens = encode(data);
        let decoded = decode(&tokens).unwrap();
        assert_eq!(decoded, data);
    }

    #[test]
    fn config_builder() {
        let config = Lz77Config::default()
            .with_window_size(2048)
            .with_max_chain(32);
        assert_eq!(config.window_size, 2048);
        assert_eq!(config.max_chain, 32);
        assert!(!config.lazy_matching);
    }
}
