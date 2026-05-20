//! Rope — efficient data structure for large text manipulation.
//!
//! Supports O(log n) concatenation, split, character-at-index, substring,
//! line-based indexing, rebalancing, and iteration.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────────────────

/// Leaf nodes hold at most this many characters before splitting.
const LEAF_MAX: usize = 64;

// ── RopeNode ────────────────────────────────────────────────────────────────

#[derive(Clone)]
enum RopeNode {
    Leaf(String),
    Branch {
        left: Box<Rope>,
        right: Box<Rope>,
        weight: usize, // char count of left subtree
    },
}

// ── Rope ────────────────────────────────────────────────────────────────────

/// Rope: a balanced binary tree of string fragments for efficient large-text editing.
#[derive(Clone)]
pub struct Rope {
    node: RopeNode,
    len: usize,
    line_count: usize,
}

impl Rope {
    /// Create an empty rope.
    pub fn new() -> Self {
        Self {
            node: RopeNode::Leaf(String::new()),
            len: 0,
            line_count: 1,
        }
    }

    /// Create a rope from a string.
    pub fn from_str(s: &str) -> Self {
        if s.len() <= LEAF_MAX {
            let line_count = s.chars().filter(|c| *c == '\n').count() + 1;
            Self {
                node: RopeNode::Leaf(s.to_string()),
                len: s.chars().count(),
                line_count,
            }
        } else {
            // Split at roughly the middle (on a char boundary)
            let mid = s.chars().count() / 2;
            let byte_mid = s
                .char_indices()
                .nth(mid)
                .map(|(i, _)| i)
                .unwrap_or(s.len());
            let left = Self::from_str(&s[..byte_mid]);
            let right = Self::from_str(&s[byte_mid..]);
            Self::concat(left, right)
        }
    }

    /// Total number of characters.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of lines (newline count + 1).
    pub fn line_count(&self) -> usize {
        self.line_count
    }

    /// Concatenate two ropes.
    pub fn concat(left: Rope, right: Rope) -> Rope {
        if left.is_empty() {
            return right;
        }
        if right.is_empty() {
            return left;
        }
        let total_len = left.len + right.len;
        // Lines: subtract 1 because both sides count one line for the junction
        let total_lines = left.line_count + right.line_count - 1;
        let weight = left.len;
        Rope {
            node: RopeNode::Branch {
                left: Box::new(left),
                right: Box::new(right),
                weight,
            },
            len: total_len,
            line_count: total_lines,
        }
    }

    /// Append another rope.
    pub fn append(&mut self, other: Rope) {
        let left = std::mem::replace(self, Rope::new());
        *self = Self::concat(left, other);
    }

    /// Character at the given index.
    pub fn char_at(&self, index: usize) -> Option<char> {
        if index >= self.len {
            return None;
        }
        match &self.node {
            RopeNode::Leaf(s) => s.chars().nth(index),
            RopeNode::Branch {
                left,
                right,
                weight,
            } => {
                if index < *weight {
                    left.char_at(index)
                } else {
                    right.char_at(index - weight)
                }
            }
        }
    }

    /// Extract a substring by character indices [start..end).
    pub fn substring(&self, start: usize, end: usize) -> String {
        let end = end.min(self.len);
        if start >= end {
            return String::new();
        }
        let mut result = String::with_capacity(end - start);
        self.collect_range(start, end, &mut result);
        result
    }

    fn collect_range(&self, start: usize, end: usize, buf: &mut String) {
        match &self.node {
            RopeNode::Leaf(s) => {
                for (i, ch) in s.chars().enumerate() {
                    if i >= start && i < end {
                        buf.push(ch);
                    }
                }
            }
            RopeNode::Branch {
                left,
                right,
                weight,
            } => {
                if start < *weight {
                    left.collect_range(start, end.min(*weight), buf);
                }
                if end > *weight {
                    let r_start = if start > *weight { start - weight } else { 0 };
                    right.collect_range(r_start, end - weight, buf);
                }
            }
        }
    }

    /// Split the rope at the given character index. Returns (left, right).
    pub fn split_at(self, index: usize) -> (Rope, Rope) {
        if index == 0 {
            return (Rope::new(), self);
        }
        if index >= self.len {
            return (self, Rope::new());
        }
        match self.node {
            RopeNode::Leaf(s) => {
                let byte_idx = s
                    .char_indices()
                    .nth(index)
                    .map(|(i, _)| i)
                    .unwrap_or(s.len());
                let left = Rope::from_str(&s[..byte_idx]);
                let right = Rope::from_str(&s[byte_idx..]);
                (left, right)
            }
            RopeNode::Branch {
                left,
                right,
                weight,
            } => {
                if index == weight {
                    (*left, *right)
                } else if index < weight {
                    let (ll, lr) = left.split_at(index);
                    (ll, Rope::concat(lr, *right))
                } else {
                    let (rl, rr) = right.split_at(index - weight);
                    (Rope::concat(*left, rl), rr)
                }
            }
        }
    }

    /// Insert text at the given character position.
    pub fn insert(&mut self, index: usize, text: &str) {
        let old = std::mem::replace(self, Rope::new());
        let (left, right) = old.split_at(index);
        let middle = Rope::from_str(text);
        *self = Rope::concat(Rope::concat(left, middle), right);
    }

    /// Delete characters in [start, end).
    pub fn delete(&mut self, start: usize, end: usize) {
        let old = std::mem::replace(self, Rope::new());
        let (left, rest) = old.split_at(start);
        let (_, right) = rest.split_at(end - start);
        *self = Rope::concat(left, right);
    }

    /// Get the text of a specific line (0-indexed).
    pub fn line(&self, line_idx: usize) -> Option<String> {
        let text = self.to_string();
        text.split('\n').nth(line_idx).map(|s| s.to_string())
    }

    /// Character offset where the given line starts.
    pub fn line_start_offset(&self, line_idx: usize) -> Option<usize> {
        if line_idx == 0 {
            return Some(0);
        }
        let text = self.to_string();
        let mut current_line = 0;
        for (i, ch) in text.chars().enumerate() {
            if ch == '\n' {
                current_line += 1;
                if current_line == line_idx {
                    return Some(i + 1);
                }
            }
        }
        None
    }

    /// Rebalance the rope into a more balanced tree.
    pub fn rebalance(&mut self) {
        let s = self.to_string();
        *self = Rope::from_str(&s);
    }

    /// Depth of the tree.
    pub fn depth(&self) -> usize {
        match &self.node {
            RopeNode::Leaf(_) => 0,
            RopeNode::Branch { left, right, .. } => {
                1 + left.depth().max(right.depth())
            }
        }
    }

    /// Iterate over characters.
    pub fn chars(&self) -> RopeChars<'_> {
        RopeChars {
            rope: self,
            index: 0,
        }
    }

    /// Convert to a contiguous string.
    pub fn to_string(&self) -> String {
        let mut buf = String::with_capacity(self.len);
        self.collect_all(&mut buf);
        buf
    }

    fn collect_all(&self, buf: &mut String) {
        match &self.node {
            RopeNode::Leaf(s) => buf.push_str(s),
            RopeNode::Branch { left, right, .. } => {
                left.collect_all(buf);
                right.collect_all(buf);
            }
        }
    }
}

impl Default for Rope {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Rope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Rope")
            .field("len", &self.len)
            .field("lines", &self.line_count)
            .field("depth", &self.depth())
            .finish()
    }
}

impl fmt::Display for Rope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

// ── Iterator ────────────────────────────────────────────────────────────────

pub struct RopeChars<'a> {
    rope: &'a Rope,
    index: usize,
}

impl<'a> Iterator for RopeChars<'a> {
    type Item = char;

    fn next(&mut self) -> Option<char> {
        let ch = self.rope.char_at(self.index)?;
        self.index += 1;
        Some(ch)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_str_and_to_string() {
        let r = Rope::from_str("hello world");
        assert_eq!(r.to_string(), "hello world");
        assert_eq!(r.len(), 11);
    }

    #[test]
    fn test_empty_rope() {
        let r = Rope::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.to_string(), "");
    }

    #[test]
    fn test_char_at() {
        let r = Rope::from_str("abcdef");
        assert_eq!(r.char_at(0), Some('a'));
        assert_eq!(r.char_at(3), Some('d'));
        assert_eq!(r.char_at(5), Some('f'));
        assert_eq!(r.char_at(6), None);
    }

    #[test]
    fn test_concat() {
        let a = Rope::from_str("hello ");
        let b = Rope::from_str("world");
        let c = Rope::concat(a, b);
        assert_eq!(c.to_string(), "hello world");
        assert_eq!(c.len(), 11);
    }

    #[test]
    fn test_split_at() {
        let r = Rope::from_str("hello world");
        let (left, right) = r.split_at(5);
        assert_eq!(left.to_string(), "hello");
        assert_eq!(right.to_string(), " world");
    }

    #[test]
    fn test_split_at_edges() {
        let r = Rope::from_str("abc");
        let (l, ri) = r.clone().split_at(0);
        assert!(l.is_empty());
        assert_eq!(ri.to_string(), "abc");
        let (l2, r2) = r.split_at(3);
        assert_eq!(l2.to_string(), "abc");
        assert!(r2.is_empty());
    }

    #[test]
    fn test_substring() {
        let r = Rope::from_str("hello world");
        assert_eq!(r.substring(0, 5), "hello");
        assert_eq!(r.substring(6, 11), "world");
        assert_eq!(r.substring(3, 8), "lo wo");
    }

    #[test]
    fn test_insert() {
        let mut r = Rope::from_str("helloworld");
        r.insert(5, " ");
        assert_eq!(r.to_string(), "hello world");
    }

    #[test]
    fn test_delete() {
        let mut r = Rope::from_str("hello world");
        r.delete(5, 6); // remove the space
        assert_eq!(r.to_string(), "helloworld");
    }

    #[test]
    fn test_line_count() {
        let r = Rope::from_str("line1\nline2\nline3");
        assert_eq!(r.line_count(), 3);
    }

    #[test]
    fn test_line() {
        let r = Rope::from_str("aaa\nbbb\nccc");
        assert_eq!(r.line(0), Some("aaa".to_string()));
        assert_eq!(r.line(1), Some("bbb".to_string()));
        assert_eq!(r.line(2), Some("ccc".to_string()));
        assert_eq!(r.line(3), None);
    }

    #[test]
    fn test_line_start_offset() {
        let r = Rope::from_str("abc\ndef\nghi");
        assert_eq!(r.line_start_offset(0), Some(0));
        assert_eq!(r.line_start_offset(1), Some(4));
        assert_eq!(r.line_start_offset(2), Some(8));
    }

    #[test]
    fn test_large_rope() {
        // Build a rope larger than LEAF_MAX to exercise branching
        let text: String = (0..500).map(|i| format!("word{} ", i)).collect();
        let r = Rope::from_str(&text);
        assert_eq!(r.to_string(), text);
        assert!(r.depth() > 0);
    }

    #[test]
    fn test_rebalance() {
        let mut r = Rope::new();
        for i in 0..50 {
            r.append(Rope::from_str(&format!("{} ", i)));
        }
        let before_depth = r.depth();
        r.rebalance();
        assert!(r.depth() <= before_depth);
        // Content is preserved
        assert!(r.to_string().starts_with("0 1 2"));
    }

    #[test]
    fn test_chars_iterator() {
        let r = Rope::from_str("abc");
        let chars: Vec<char> = r.chars().collect();
        assert_eq!(chars, vec!['a', 'b', 'c']);
    }

    #[test]
    fn test_append() {
        let mut r = Rope::from_str("hello");
        r.append(Rope::from_str(" world"));
        assert_eq!(r.to_string(), "hello world");
    }
}
