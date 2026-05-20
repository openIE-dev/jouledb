//! Selection model for rich text editors.
//!
//! Provides cursor, range, and node selection types with position resolution
//! against the document tree. Replaces ProseMirror selection with pure Rust.

use crate::doc_model::Node;

// ── Position ────────────────────────────────────────────────────

/// A position in the document tree: a path to a node + an offset within it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position {
    /// Path of child indices from root to the target node.
    pub path: Vec<usize>,
    /// Character offset within the target node (for text nodes).
    pub offset: usize,
}

impl Position {
    pub fn new(path: Vec<usize>, offset: usize) -> Self {
        Self { path, offset }
    }

    /// A position at the very start of the document.
    pub fn start() -> Self {
        Self {
            path: Vec::new(),
            offset: 0,
        }
    }

    /// Compare two positions. Returns Ordering based on path then offset.
    pub fn cmp_pos(&self, other: &Position) -> std::cmp::Ordering {
        for (a, b) in self.path.iter().zip(other.path.iter()) {
            match a.cmp(b) {
                std::cmp::Ordering::Equal => continue,
                ord => return ord,
            }
        }
        match self.path.len().cmp(&other.path.len()) {
            std::cmp::Ordering::Equal => self.offset.cmp(&other.offset),
            ord => ord,
        }
    }

    /// Resolve this position against a document, returning the referenced node.
    pub fn resolve<'a>(&self, doc: &'a Node) -> Option<&'a Node> {
        doc.node_at(&self.path)
    }

    /// Check if this position is valid for the given document.
    pub fn is_valid(&self, doc: &Node) -> bool {
        match doc.node_at(&self.path) {
            Some(Node::Text { text, .. }) => self.offset <= text.len(),
            Some(_) => self.offset == 0,
            None => false,
        }
    }
}

// ── Selection types ─────────────────────────────────────────────

/// A selection in the document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Selection {
    /// A cursor (collapsed selection). anchor == head.
    Cursor(Position),
    /// A text range selection with anchor and head (head can be before anchor).
    Range { anchor: Position, head: Position },
    /// A whole-node selection (selects an entire node).
    NodeSelection { path: Vec<usize> },
}

impl Selection {
    /// Create a cursor at the given position.
    pub fn cursor(path: Vec<usize>, offset: usize) -> Self {
        Selection::Cursor(Position::new(path, offset))
    }

    /// Create a range selection.
    pub fn range(anchor: Position, head: Position) -> Self {
        if anchor == head {
            Selection::Cursor(anchor)
        } else {
            Selection::Range { anchor, head }
        }
    }

    /// Create a node selection.
    pub fn node(path: Vec<usize>) -> Self {
        Selection::NodeSelection { path }
    }

    /// Is this a collapsed (cursor) selection?
    pub fn is_collapsed(&self) -> bool {
        matches!(self, Selection::Cursor(_))
    }

    /// Get the anchor position.
    pub fn anchor(&self) -> Position {
        match self {
            Selection::Cursor(pos) => pos.clone(),
            Selection::Range { anchor, .. } => anchor.clone(),
            Selection::NodeSelection { path } => Position::new(path.clone(), 0),
        }
    }

    /// Get the head position.
    pub fn head(&self) -> Position {
        match self {
            Selection::Cursor(pos) => pos.clone(),
            Selection::Range { head, .. } => head.clone(),
            Selection::NodeSelection { path } => Position::new(path.clone(), 0),
        }
    }

    /// Get the earlier (from) position of the selection.
    pub fn from_pos(&self) -> Position {
        let a = self.anchor();
        let h = self.head();
        if a.cmp_pos(&h) == std::cmp::Ordering::Less {
            a
        } else {
            h
        }
    }

    /// Get the later (to) position of the selection.
    pub fn to_pos(&self) -> Position {
        let a = self.anchor();
        let h = self.head();
        if a.cmp_pos(&h) == std::cmp::Ordering::Greater {
            a
        } else {
            h
        }
    }

    /// Collapse selection to the start (from position).
    pub fn collapse_to_start(&self) -> Selection {
        Selection::Cursor(self.from_pos())
    }

    /// Collapse selection to the end (to position).
    pub fn collapse_to_end(&self) -> Selection {
        Selection::Cursor(self.to_pos())
    }

    /// Select all content in the document.
    pub fn select_all(doc: &Node) -> Selection {
        let start = find_first_text_position(doc);
        let end = find_last_text_position(doc);
        match (start, end) {
            (Some(s), Some(e)) => Selection::Range {
                anchor: s,
                head: e,
            },
            _ => Selection::Cursor(Position::start()),
        }
    }

    /// Validate this selection against a document.
    pub fn is_valid(&self, doc: &Node) -> bool {
        match self {
            Selection::Cursor(pos) => pos.is_valid(doc),
            Selection::Range { anchor, head } => anchor.is_valid(doc) && head.is_valid(doc),
            Selection::NodeSelection { path } => doc.node_at(path).is_some(),
        }
    }

    /// Extend the selection forward by one character (shift+right logic).
    pub fn extend_forward(&self, doc: &Node) -> Selection {
        let head = self.head();
        if let Some(new_head) = next_position(doc, &head) {
            Selection::Range {
                anchor: self.anchor(),
                head: new_head,
            }
        } else {
            self.clone()
        }
    }

    /// Extend the selection backward by one character (shift+left logic).
    pub fn extend_backward(&self, doc: &Node) -> Selection {
        let head = self.head();
        if let Some(new_head) = prev_position(doc, &head) {
            Selection::range(self.anchor(), new_head)
        } else {
            self.clone()
        }
    }
}

// ── Word boundary detection ─────────────────────────────────────

/// Detect if a character is a word character (Unicode-aware).
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Find the start of the word at the given offset in a string.
pub fn word_start(text: &str, offset: usize) -> usize {
    let bytes: Vec<char> = text.chars().collect();
    if offset == 0 || offset > bytes.len() {
        return offset;
    }
    let mut pos = offset;
    // Move back while we see word characters
    while pos > 0 && is_word_char(bytes[pos - 1]) {
        pos -= 1;
    }
    pos
}

/// Find the end of the word at the given offset in a string.
pub fn word_end(text: &str, offset: usize) -> usize {
    let chars: Vec<char> = text.chars().collect();
    if offset >= chars.len() {
        return chars.len();
    }
    let mut pos = offset;
    while pos < chars.len() && is_word_char(chars[pos]) {
        pos += 1;
    }
    pos
}

/// Select the word at the given position (double-click behavior).
pub fn select_word(text: &str, offset: usize) -> (usize, usize) {
    let start = word_start(text, offset);
    let end = word_end(text, start);
    (start, end)
}

// ── Navigation helpers ──────────────────────────────────────────

/// Find the first text position in the document (depth-first).
fn find_first_text_position(node: &Node) -> Option<Position> {
    find_first_text_position_inner(node, &mut Vec::new())
}

fn find_first_text_position_inner(node: &Node, path: &mut Vec<usize>) -> Option<Position> {
    match node {
        Node::Text { .. } => Some(Position::new(path.clone(), 0)),
        other => {
            if let Some(children) = other.children() {
                for (i, child) in children.iter().enumerate() {
                    path.push(i);
                    if let Some(pos) = find_first_text_position_inner(child, path) {
                        return Some(pos);
                    }
                    path.pop();
                }
            }
            None
        }
    }
}

/// Find the last text position in the document (depth-first, last child).
fn find_last_text_position(node: &Node) -> Option<Position> {
    find_last_text_position_inner(node, &mut Vec::new())
}

fn find_last_text_position_inner(node: &Node, path: &mut Vec<usize>) -> Option<Position> {
    match node {
        Node::Text { text, .. } => Some(Position::new(path.clone(), text.len())),
        other => {
            if let Some(children) = other.children() {
                for (i, child) in children.iter().enumerate().rev() {
                    path.push(i);
                    if let Some(pos) = find_last_text_position_inner(child, path) {
                        return Some(pos);
                    }
                    path.pop();
                }
            }
            None
        }
    }
}

/// Move one position forward (next character or next text node).
fn next_position(doc: &Node, pos: &Position) -> Option<Position> {
    if let Some(Node::Text { text, .. }) = doc.node_at(&pos.path) {
        if pos.offset < text.len() {
            // Move forward within the same text node.
            let next_offset = text
                .char_indices()
                .find(|(i, _)| *i > pos.offset)
                .map(|(i, _)| i)
                .unwrap_or(text.len());
            return Some(Position::new(pos.path.clone(), next_offset));
        }
    }
    // Try to find the next text node
    find_next_text_node(doc, &pos.path)
}

/// Move one position backward.
fn prev_position(doc: &Node, pos: &Position) -> Option<Position> {
    if pos.offset > 0 {
        if let Some(Node::Text { text, .. }) = doc.node_at(&pos.path) {
            let prev_offset = text
                .char_indices()
                .rev()
                .find(|(i, _)| *i < pos.offset)
                .map(|(i, _)| i)
                .unwrap_or(0);
            return Some(Position::new(pos.path.clone(), prev_offset));
        }
    }
    // Try to find previous text node
    find_prev_text_node(doc, &pos.path)
}

/// Find the next text node after the one at `path`.
fn find_next_text_node(doc: &Node, path: &[usize]) -> Option<Position> {
    if path.is_empty() {
        return None;
    }
    let parent_path = &path[..path.len() - 1];
    let idx = path[path.len() - 1];
    let parent = doc.node_at(parent_path)?;
    let children = parent.children()?;

    // Search siblings after current
    for i in (idx + 1)..children.len() {
        let mut new_path = parent_path.to_vec();
        new_path.push(i);
        if let Some(pos) = find_first_text_position_inner(&children[i], &mut new_path) {
            return Some(pos);
        }
    }

    // Go up and look for next sibling of parent
    find_next_text_node(doc, parent_path)
}

/// Find the previous text node before the one at `path`.
fn find_prev_text_node(doc: &Node, path: &[usize]) -> Option<Position> {
    if path.is_empty() {
        return None;
    }
    let parent_path = &path[..path.len() - 1];
    let idx = path[path.len() - 1];
    let parent = doc.node_at(parent_path)?;
    let children = parent.children()?;

    // Search siblings before current (in reverse)
    for i in (0..idx).rev() {
        let mut new_path = parent_path.to_vec();
        new_path.push(i);
        if let Some(pos) = find_last_text_position_inner(&children[i], &mut new_path) {
            return Some(pos);
        }
    }

    find_prev_text_node(doc, parent_path)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_doc() -> Node {
        Node::doc(vec![
            Node::paragraph(vec![Node::text("Hello world")]),
            Node::paragraph(vec![Node::text("Second paragraph")]),
        ])
    }

    #[test]
    fn cursor_is_collapsed() {
        let sel = Selection::cursor(vec![0, 0], 3);
        assert!(sel.is_collapsed());
    }

    #[test]
    fn range_not_collapsed() {
        let sel = Selection::Range {
            anchor: Position::new(vec![0, 0], 0),
            head: Position::new(vec![0, 0], 5),
        };
        assert!(!sel.is_collapsed());
    }

    #[test]
    fn range_collapses_when_equal() {
        let sel = Selection::range(
            Position::new(vec![0, 0], 3),
            Position::new(vec![0, 0], 3),
        );
        assert!(sel.is_collapsed());
    }

    #[test]
    fn from_to_ordering() {
        let sel = Selection::Range {
            anchor: Position::new(vec![1, 0], 5),
            head: Position::new(vec![0, 0], 2),
        };
        let from = sel.from_pos();
        let to = sel.to_pos();
        assert_eq!(from.path, vec![0, 0]);
        assert_eq!(to.path, vec![1, 0]);
    }

    #[test]
    fn position_validity() {
        let doc = sample_doc();
        assert!(Position::new(vec![0, 0], 5).is_valid(&doc));
        assert!(Position::new(vec![0, 0], 11).is_valid(&doc)); // "Hello world" len=11
        assert!(!Position::new(vec![0, 0], 12).is_valid(&doc)); // beyond end
        assert!(!Position::new(vec![5, 0], 0).is_valid(&doc)); // path doesn't exist
    }

    #[test]
    fn selection_validity() {
        let doc = sample_doc();
        let valid = Selection::cursor(vec![0, 0], 3);
        assert!(valid.is_valid(&doc));

        let invalid = Selection::cursor(vec![9, 0], 0);
        assert!(!invalid.is_valid(&doc));
    }

    #[test]
    fn select_all() {
        let doc = sample_doc();
        let sel = Selection::select_all(&doc);
        match &sel {
            Selection::Range { anchor, head } => {
                assert_eq!(anchor.path, vec![0, 0]);
                assert_eq!(anchor.offset, 0);
                assert_eq!(head.path, vec![1, 0]);
                assert_eq!(head.offset, 16); // "Second paragraph".len()
            }
            _ => panic!("expected range selection"),
        }
    }

    #[test]
    fn word_boundary_basic() {
        let text = "Hello world test";
        assert_eq!(word_start(text, 3), 0); // in "Hello"
        assert_eq!(word_end(text, 0), 5); // "Hello"
        assert_eq!(select_word(text, 7), (6, 11)); // "world"
    }

    #[test]
    fn word_boundary_punctuation() {
        let text = "foo.bar baz";
        assert_eq!(select_word(text, 1), (0, 3)); // "foo"
        assert_eq!(select_word(text, 5), (4, 7)); // "bar"
    }

    #[test]
    fn extend_forward() {
        let doc = sample_doc();
        let sel = Selection::cursor(vec![0, 0], 0);
        let extended = sel.extend_forward(&doc);
        match &extended {
            Selection::Range { anchor, head } => {
                assert_eq!(anchor.offset, 0);
                assert_eq!(head.offset, 1);
            }
            _ => panic!("expected range"),
        }
    }

    #[test]
    fn extend_backward() {
        let doc = sample_doc();
        let sel = Selection::cursor(vec![0, 0], 5);
        let extended = sel.extend_backward(&doc);
        match &extended {
            Selection::Range { anchor, head } => {
                assert_eq!(anchor.offset, 5);
                assert_eq!(head.offset, 4);
            }
            _ => panic!("expected range"),
        }
    }

    #[test]
    fn collapse_to_start_and_end() {
        let sel = Selection::Range {
            anchor: Position::new(vec![0, 0], 3),
            head: Position::new(vec![0, 0], 8),
        };
        let start = sel.collapse_to_start();
        let end = sel.collapse_to_end();
        assert!(start.is_collapsed());
        assert!(end.is_collapsed());
        assert_eq!(start.head().offset, 3);
        assert_eq!(end.head().offset, 8);
    }

    #[test]
    fn position_comparison() {
        let a = Position::new(vec![0, 0], 5);
        let b = Position::new(vec![0, 0], 10);
        let c = Position::new(vec![1, 0], 0);
        assert_eq!(a.cmp_pos(&b), std::cmp::Ordering::Less);
        assert_eq!(b.cmp_pos(&a), std::cmp::Ordering::Greater);
        assert_eq!(a.cmp_pos(&c), std::cmp::Ordering::Less);
    }
}
