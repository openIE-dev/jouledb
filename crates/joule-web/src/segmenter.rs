//! Text segmentation — grapheme clusters, words, sentences, line breaks.
//!
//! Simplified implementation of UAX #29 (grapheme/word/sentence boundaries)
//! and UAX #14 (line break opportunities) — pure Rust, no ICU dependency.

use std::fmt;

// ── Boundary type ───────────────────────────────────────────────

/// The type of segment boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryType {
    /// No boundary (characters are in the same segment).
    None,
    /// A segment boundary exists here.
    Break,
    /// A soft/optional boundary (e.g., line break opportunity).
    Opportunity,
}

// ── Segment ─────────────────────────────────────────────────────

/// A segment of text with its boundaries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    /// The text content of this segment.
    pub text: String,
    /// Byte offset of the start of this segment in the original string.
    pub start: usize,
    /// Byte offset of the end (exclusive) of this segment.
    pub end: usize,
    /// The kind of segment (for word segmentation).
    pub kind: SegmentKind,
}

/// Classification of a word segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    /// A word (letters/digits).
    Word,
    /// Whitespace.
    Whitespace,
    /// Punctuation or other.
    Other,
}

impl fmt::Display for SegmentKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Word => write!(f, "word"),
            Self::Whitespace => write!(f, "whitespace"),
            Self::Other => write!(f, "other"),
        }
    }
}

// ── Grapheme segmenter ──────────────────────────────────────────

/// Segments text into grapheme clusters (simplified).
///
/// This handles common cases: ASCII characters, CR+LF pairs, and
/// combining mark sequences. Full UAX #29 requires a large table.
pub struct GraphemeSegmenter;

impl GraphemeSegmenter {
    /// Iterate over grapheme cluster boundaries.
    pub fn segment(text: &str) -> Vec<Segment> {
        let mut segments = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        if chars.is_empty() {
            return segments;
        }

        let mut start_byte = 0usize;
        let mut cluster = String::new();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];

            // CR+LF treated as single grapheme cluster
            if ch == '\r' && i + 1 < chars.len() && chars[i + 1] == '\n' {
                if !cluster.is_empty() {
                    let end_byte = start_byte + cluster.len();
                    segments.push(Segment {
                        text: cluster.clone(),
                        start: start_byte,
                        end: end_byte,
                        kind: classify_segment(&cluster),
                    });
                    start_byte = end_byte;
                    cluster.clear();
                }
                let crlf = "\r\n".to_string();
                let end_byte = start_byte + crlf.len();
                segments.push(Segment {
                    text: crlf,
                    start: start_byte,
                    end: end_byte,
                    kind: SegmentKind::Whitespace,
                });
                start_byte = end_byte;
                i += 2;
                continue;
            }

            // Combining marks attach to the previous character
            if is_combining_mark(ch) && !cluster.is_empty() {
                cluster.push(ch);
                i += 1;
                continue;
            }

            // Start new cluster
            if !cluster.is_empty() {
                let end_byte = start_byte + cluster.len();
                segments.push(Segment {
                    text: cluster.clone(),
                    start: start_byte,
                    end: end_byte,
                    kind: classify_segment(&cluster),
                });
                start_byte = end_byte;
                cluster.clear();
            }
            cluster.push(ch);
            i += 1;
        }

        if !cluster.is_empty() {
            let end_byte = start_byte + cluster.len();
            segments.push(Segment {
                text: cluster,
                start: start_byte,
                end: end_byte,
                kind: SegmentKind::Other,
            });
        }

        segments
    }
}

// ── Word segmenter ──────────────────────────────────────────────

/// Segments text into words, whitespace, and punctuation.
pub struct WordSegmenter;

impl WordSegmenter {
    /// Segment text into words.
    pub fn segment(text: &str) -> Vec<Segment> {
        let mut segments = Vec::new();
        let mut current = String::new();
        let mut current_kind = None;
        let mut start_byte = 0usize;

        for ch in text.chars() {
            let kind = char_word_class(ch);
            if let Some(prev_kind) = current_kind {
                if kind != prev_kind {
                    let end_byte = start_byte + current.len();
                    segments.push(Segment {
                        text: current.clone(),
                        start: start_byte,
                        end: end_byte,
                        kind: prev_kind,
                    });
                    start_byte = end_byte;
                    current.clear();
                }
            }
            current.push(ch);
            current_kind = Some(kind);
        }

        if !current.is_empty() {
            if let Some(kind) = current_kind {
                let end_byte = start_byte + current.len();
                segments.push(Segment {
                    text: current,
                    start: start_byte,
                    end: end_byte,
                    kind,
                });
            }
        }

        segments
    }

    /// Extract only word segments (filtering out whitespace/punctuation).
    pub fn words(text: &str) -> Vec<String> {
        Self::segment(text)
            .into_iter()
            .filter(|s| s.kind == SegmentKind::Word)
            .map(|s| s.text)
            .collect()
    }

    /// Count words in the text.
    pub fn word_count(text: &str) -> usize {
        Self::words(text).len()
    }
}

// ── Sentence segmenter ──────────────────────────────────────────

/// Segments text into sentences (simplified).
pub struct SentenceSegmenter;

impl SentenceSegmenter {
    /// Segment text into sentences.
    pub fn segment(text: &str) -> Vec<Segment> {
        let mut sentences = Vec::new();
        let mut current = String::new();
        let mut start_byte = 0usize;
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];
            current.push(ch);
            i += 1;

            // Sentence ends at . ! ? followed by whitespace or end-of-string
            if is_sentence_terminal(ch) {
                // Consume any trailing whitespace
                while i < chars.len() && chars[i].is_whitespace() {
                    current.push(chars[i]);
                    i += 1;
                }
                let end_byte = start_byte + current.len();
                sentences.push(Segment {
                    text: current.clone(),
                    start: start_byte,
                    end: end_byte,
                    kind: SegmentKind::Other,
                });
                start_byte = end_byte;
                current.clear();
            }
        }

        if !current.is_empty() {
            let end_byte = start_byte + current.len();
            sentences.push(Segment {
                text: current,
                start: start_byte,
                end: end_byte,
                kind: SegmentKind::Other,
            });
        }

        sentences
    }

    /// Count sentences in the text.
    pub fn count(text: &str) -> usize {
        Self::segment(text).len()
    }
}

// ── Line break segmenter ────────────────────────────────────────

/// Finds line break opportunities (UAX #14 simplified).
pub struct LineBreakSegmenter;

impl LineBreakSegmenter {
    /// Find byte offsets where line breaks may occur.
    pub fn opportunities(text: &str) -> Vec<usize> {
        let mut breaks = Vec::new();
        let mut byte_offset = 0;
        let chars: Vec<char> = text.chars().collect();

        for (i, ch) in chars.iter().enumerate() {
            let ch_len = ch.len_utf8();

            // Mandatory break after newline
            if *ch == '\n' {
                breaks.push(byte_offset + ch_len);
            }
            // Break opportunity after space (but not before punctuation)
            else if ch.is_whitespace() && i + 1 < chars.len() {
                let next = chars[i + 1];
                if !is_close_punctuation(next) {
                    breaks.push(byte_offset + ch_len);
                }
            }
            // Break after hyphens
            else if *ch == '-' && i + 1 < chars.len() {
                breaks.push(byte_offset + ch_len);
            }

            byte_offset += ch_len;
        }
        breaks
    }

    /// Segment text into lines that fit within `max_width` characters.
    pub fn wrap(text: &str, max_width: usize) -> Vec<String> {
        if max_width == 0 {
            return vec![text.to_string()];
        }

        let mut lines = Vec::new();
        let mut current_line = String::new();

        for word in text.split_whitespace() {
            if current_line.is_empty() {
                current_line = word.to_string();
            } else if current_line.len() + 1 + word.len() <= max_width {
                current_line.push(' ');
                current_line.push_str(word);
            } else {
                lines.push(current_line);
                current_line = word.to_string();
            }
        }
        if !current_line.is_empty() {
            lines.push(current_line);
        }
        lines
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn is_combining_mark(ch: char) -> bool {
    let cp = ch as u32;
    // Combining Diacritical Marks: U+0300..U+036F
    // Combining Diacritical Marks Extended: U+1AB0..U+1AFF
    // Combining Diacritical Marks Supplement: U+1DC0..U+1DFF
    (0x0300..=0x036F).contains(&cp)
        || (0x1AB0..=0x1AFF).contains(&cp)
        || (0x1DC0..=0x1DFF).contains(&cp)
        || (0x20D0..=0x20FF).contains(&cp)
        || (0xFE20..=0xFE2F).contains(&cp)
}

fn classify_segment(s: &str) -> SegmentKind {
    let first = s.chars().next().unwrap_or(' ');
    if first.is_alphanumeric() {
        SegmentKind::Word
    } else if first.is_whitespace() {
        SegmentKind::Whitespace
    } else {
        SegmentKind::Other
    }
}

fn char_word_class(ch: char) -> SegmentKind {
    if ch.is_alphanumeric() || ch == '_' || ch == '\'' {
        SegmentKind::Word
    } else if ch.is_whitespace() {
        SegmentKind::Whitespace
    } else {
        SegmentKind::Other
    }
}

fn is_sentence_terminal(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '\u{2026}')
}

fn is_close_punctuation(ch: char) -> bool {
    matches!(ch, ')' | ']' | '}' | '.' | ',' | ';' | ':' | '!' | '?')
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grapheme_ascii() {
        let segs = GraphemeSegmenter::segment("abc");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[0].text, "a");
        assert_eq!(segs[1].text, "b");
        assert_eq!(segs[2].text, "c");
    }

    #[test]
    fn grapheme_crlf() {
        let segs = GraphemeSegmenter::segment("a\r\nb");
        assert_eq!(segs.len(), 3);
        assert_eq!(segs[1].text, "\r\n");
    }

    #[test]
    fn grapheme_combining() {
        // e + combining acute accent = single grapheme cluster
        let segs = GraphemeSegmenter::segment("e\u{0301}x");
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "e\u{0301}");
        assert_eq!(segs[1].text, "x");
    }

    #[test]
    fn word_segmentation() {
        let words = WordSegmenter::words("Hello, world! How are you?");
        assert_eq!(words, vec!["Hello", "world", "How", "are", "you"]);
    }

    #[test]
    fn word_count() {
        assert_eq!(WordSegmenter::word_count("One two three"), 3);
        assert_eq!(WordSegmenter::word_count(""), 0);
    }

    #[test]
    fn word_segments_with_offsets() {
        let segs = WordSegmenter::segment("Hi there");
        assert_eq!(segs.len(), 3); // "Hi", " ", "there"
        assert_eq!(segs[0].kind, SegmentKind::Word);
        assert_eq!(segs[1].kind, SegmentKind::Whitespace);
        assert_eq!(segs[2].kind, SegmentKind::Word);
    }

    #[test]
    fn sentence_segmentation() {
        let sents = SentenceSegmenter::segment("Hello world. How are you? Fine!");
        assert_eq!(sents.len(), 3);
        assert!(sents[0].text.starts_with("Hello world."));
        assert!(sents[1].text.starts_with("How are you?"));
        assert!(sents[2].text.starts_with("Fine!"));
    }

    #[test]
    fn sentence_count() {
        assert_eq!(SentenceSegmenter::count("One. Two. Three."), 3);
    }

    #[test]
    fn line_break_opportunities() {
        let breaks = LineBreakSegmenter::opportunities("Hello world test");
        // Break opportunity after "Hello " and "world "
        assert!(breaks.len() >= 2);
    }

    #[test]
    fn line_wrap() {
        let lines = LineBreakSegmenter::wrap("The quick brown fox jumps over the lazy dog", 20);
        for line in &lines {
            assert!(line.len() <= 20, "Line too long: '{line}' ({} chars)", line.len());
        }
        assert!(lines.len() >= 2);
    }

    #[test]
    fn line_wrap_single_word() {
        let lines = LineBreakSegmenter::wrap("superlongword", 5);
        // Single word that exceeds width stays on one line
        assert_eq!(lines.len(), 1);
    }

    #[test]
    fn empty_input() {
        assert!(GraphemeSegmenter::segment("").is_empty());
        assert!(WordSegmenter::segment("").is_empty());
        assert!(SentenceSegmenter::segment("").is_empty());
    }

    #[test]
    fn segment_kind_display() {
        assert_eq!(SegmentKind::Word.to_string(), "word");
        assert_eq!(SegmentKind::Whitespace.to_string(), "whitespace");
        assert_eq!(SegmentKind::Other.to_string(), "other");
    }

    #[test]
    fn word_with_apostrophe() {
        let words = WordSegmenter::words("don't stop");
        assert_eq!(words, vec!["don't", "stop"]);
    }
}
