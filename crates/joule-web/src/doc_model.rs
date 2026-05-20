//! ProseMirror-style document model.
//!
//! Defines the node tree and mark system for rich text documents.
//! Replaces ProseMirror/TipTap document model with a pure Rust implementation.

use serde::{Deserialize, Serialize};

// ── Mark types ──────────────────────────────────────────────────

/// Inline marks applied to text nodes.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Mark {
    Bold,
    Italic,
    Code,
    Link { href: String },
    Strikethrough,
    Underline,
}

impl Mark {
    /// Priority for ordering marks consistently.
    fn priority(&self) -> u8 {
        match self {
            Mark::Link { .. } => 0,
            Mark::Bold => 1,
            Mark::Italic => 2,
            Mark::Underline => 3,
            Mark::Strikethrough => 4,
            Mark::Code => 5,
        }
    }

    /// Sort marks by priority so serialization is deterministic.
    pub fn sort_marks(marks: &mut Vec<Mark>) {
        marks.sort_by_key(|m| m.priority());
    }
}

// ── Node types ──────────────────────────────────────────────────

/// List kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListKind {
    Ordered,
    Unordered,
}

/// Attributes for image nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ImageAttrs {
    pub src: String,
    pub alt: String,
    pub title: Option<String>,
}

/// A node in the document tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Node {
    /// Root document node containing block-level children.
    Doc { children: Vec<Node> },
    /// A paragraph containing inline content.
    Paragraph { children: Vec<Node> },
    /// A heading with level 1-6.
    Heading { level: u8, children: Vec<Node> },
    /// A block quote.
    BlockQuote { children: Vec<Node> },
    /// A fenced code block with optional language.
    CodeBlock { language: Option<String>, text: String },
    /// A list (ordered or unordered) containing list items.
    List { kind: ListKind, children: Vec<Node> },
    /// A list item containing block-level content.
    ListItem { children: Vec<Node> },
    /// A text node with optional marks.
    Text { text: String, marks: Vec<Mark> },
    /// A hard line break.
    HardBreak,
    /// An inline image.
    Image(ImageAttrs),
    /// A horizontal rule.
    HorizontalRule,
}

impl Node {
    // ── Constructors ────────────────────────────────────────────

    /// Create a new empty document.
    pub fn doc(children: Vec<Node>) -> Self {
        Node::Doc { children }
    }

    /// Create a paragraph with given inline children.
    pub fn paragraph(children: Vec<Node>) -> Self {
        Node::Paragraph { children }
    }

    /// Create a heading.
    pub fn heading(level: u8, children: Vec<Node>) -> Self {
        let level = level.clamp(1, 6);
        Node::Heading { level, children }
    }

    /// Create a plain text node with no marks.
    pub fn text(s: &str) -> Self {
        Node::Text {
            text: s.to_string(),
            marks: Vec::new(),
        }
    }

    /// Create a text node with marks.
    pub fn text_with_marks(s: &str, marks: Vec<Mark>) -> Self {
        Node::Text {
            text: s.to_string(),
            marks,
        }
    }

    /// Create an ordered list.
    pub fn ordered_list(items: Vec<Node>) -> Self {
        Node::List {
            kind: ListKind::Ordered,
            children: items,
        }
    }

    /// Create an unordered list.
    pub fn unordered_list(items: Vec<Node>) -> Self {
        Node::List {
            kind: ListKind::Unordered,
            children: items,
        }
    }

    /// Create a list item.
    pub fn list_item(children: Vec<Node>) -> Self {
        Node::ListItem { children }
    }

    /// Create a code block.
    pub fn code_block(lang: Option<&str>, text: &str) -> Self {
        Node::CodeBlock {
            language: lang.map(|s| s.to_string()),
            text: text.to_string(),
        }
    }

    /// Create a block quote.
    pub fn block_quote(children: Vec<Node>) -> Self {
        Node::BlockQuote { children }
    }

    /// Create an image node.
    pub fn image(src: &str, alt: &str) -> Self {
        Node::Image(ImageAttrs {
            src: src.to_string(),
            alt: alt.to_string(),
            title: None,
        })
    }

    // ── Queries ─────────────────────────────────────────────────

    /// Get the children of a container node, if any.
    pub fn children(&self) -> Option<&[Node]> {
        match self {
            Node::Doc { children }
            | Node::Paragraph { children }
            | Node::Heading { children, .. }
            | Node::BlockQuote { children }
            | Node::List { children, .. }
            | Node::ListItem { children } => Some(children),
            _ => None,
        }
    }

    /// Get mutable children.
    pub fn children_mut(&mut self) -> Option<&mut Vec<Node>> {
        match self {
            Node::Doc { children }
            | Node::Paragraph { children }
            | Node::Heading { children, .. }
            | Node::BlockQuote { children }
            | Node::List { children, .. }
            | Node::ListItem { children } => Some(children),
            _ => None,
        }
    }

    /// Is this an inline node?
    pub fn is_inline(&self) -> bool {
        matches!(
            self,
            Node::Text { .. } | Node::HardBreak | Node::Image(_)
        )
    }

    /// Is this a block node?
    pub fn is_block(&self) -> bool {
        !self.is_inline()
    }

    /// Extract the plain text content of this node and all descendants.
    pub fn text_content(&self) -> String {
        let mut buf = String::new();
        self.collect_text(&mut buf);
        buf
    }

    fn collect_text(&self, buf: &mut String) {
        match self {
            Node::Text { text, .. } => buf.push_str(text),
            Node::HardBreak => buf.push('\n'),
            Node::CodeBlock { text, .. } => buf.push_str(text),
            Node::HorizontalRule | Node::Image(_) => {}
            other => {
                if let Some(children) = other.children() {
                    for child in children {
                        child.collect_text(buf);
                    }
                }
            }
        }
    }

    /// Count the total number of nodes in the tree (inclusive).
    pub fn node_count(&self) -> usize {
        let mut count = 1;
        if let Some(children) = self.children() {
            for child in children {
                count += child.node_count();
            }
        }
        count
    }

    /// Check whether any text node in the subtree has the given mark.
    pub fn has_mark(&self, target: &Mark) -> bool {
        match self {
            Node::Text { marks, .. } => marks.contains(target),
            other => {
                if let Some(children) = other.children() {
                    children.iter().any(|c| c.has_mark(target))
                } else {
                    false
                }
            }
        }
    }

    /// Find a node at a given path (indices into children at each level).
    pub fn node_at(&self, path: &[usize]) -> Option<&Node> {
        if path.is_empty() {
            return Some(self);
        }
        let children = self.children()?;
        let idx = path[0];
        children.get(idx)?.node_at(&path[1..])
    }

    /// Find a mutable node at a given path.
    pub fn node_at_mut(&mut self, path: &[usize]) -> Option<&mut Node> {
        if path.is_empty() {
            return Some(self);
        }
        let children = self.children_mut()?;
        let idx = path[0];
        children.get_mut(idx)?.node_at_mut(&path[1..])
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty JSON.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Apply a mark to all text nodes in this subtree.
    pub fn add_mark(&mut self, mark: Mark) {
        match self {
            Node::Text { marks, .. } => {
                if !marks.contains(&mark) {
                    marks.push(mark);
                    Mark::sort_marks(marks);
                }
            }
            other => {
                if let Some(children) = other.children_mut() {
                    for child in children {
                        child.add_mark(mark.clone());
                    }
                }
            }
        }
    }

    /// Remove a mark from all text nodes in this subtree.
    pub fn remove_mark(&mut self, mark: &Mark) {
        match self {
            Node::Text { marks, .. } => marks.retain(|m| m != mark),
            other => {
                if let Some(children) = other.children_mut() {
                    for child in children {
                        child.remove_mark(mark);
                    }
                }
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_simple_doc() {
        let doc = Node::doc(vec![Node::paragraph(vec![Node::text("Hello, world!")])]);
        assert_eq!(doc.text_content(), "Hello, world!");
    }

    #[test]
    fn heading_level_clamped() {
        let h = Node::heading(10, vec![Node::text("Title")]);
        match &h {
            Node::Heading { level, .. } => assert_eq!(*level, 6),
            _ => panic!("expected heading"),
        }
    }

    #[test]
    fn text_with_marks() {
        let t = Node::text_with_marks("bold", vec![Mark::Bold]);
        match &t {
            Node::Text { marks, .. } => {
                assert!(marks.contains(&Mark::Bold));
            }
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn node_at_path() {
        let doc = Node::doc(vec![
            Node::paragraph(vec![Node::text("first")]),
            Node::paragraph(vec![Node::text("second")]),
        ]);
        let node = doc.node_at(&[1, 0]).unwrap();
        match node {
            Node::Text { text, .. } => assert_eq!(text, "second"),
            _ => panic!("expected text node"),
        }
    }

    #[test]
    fn node_count() {
        let doc = Node::doc(vec![
            Node::paragraph(vec![Node::text("a"), Node::text("b")]),
            Node::paragraph(vec![Node::text("c")]),
        ]);
        // doc(1) + para(1) + text(2) + para(1) + text(1) = 6
        assert_eq!(doc.node_count(), 6);
    }

    #[test]
    fn is_inline_vs_block() {
        assert!(Node::text("x").is_inline());
        assert!(Node::HardBreak.is_inline());
        assert!(Node::image("a.png", "img").is_inline());
        assert!(Node::paragraph(vec![]).is_block());
        assert!(Node::HorizontalRule.is_block());
    }

    #[test]
    fn json_roundtrip() {
        let doc = Node::doc(vec![
            Node::paragraph(vec![
                Node::text("Hello "),
                Node::text_with_marks("world", vec![Mark::Bold]),
            ]),
            Node::code_block(Some("rust"), "fn main() {}"),
        ]);
        let json = doc.to_json().unwrap();
        let restored = Node::from_json(&json).unwrap();
        assert_eq!(doc, restored);
    }

    #[test]
    fn add_and_remove_mark() {
        let mut doc = Node::doc(vec![Node::paragraph(vec![
            Node::text("hello"),
            Node::text("world"),
        ])]);
        doc.add_mark(Mark::Italic);
        assert!(doc.has_mark(&Mark::Italic));
        doc.remove_mark(&Mark::Italic);
        assert!(!doc.has_mark(&Mark::Italic));
    }

    #[test]
    fn list_construction() {
        let list = Node::ordered_list(vec![
            Node::list_item(vec![Node::paragraph(vec![Node::text("one")])]),
            Node::list_item(vec![Node::paragraph(vec![Node::text("two")])]),
        ]);
        assert_eq!(list.text_content(), "onetwo");
        assert_eq!(list.node_count(), 7);
    }

    #[test]
    fn block_quote_text_content() {
        let bq = Node::block_quote(vec![Node::paragraph(vec![Node::text("quoted")])]);
        assert_eq!(bq.text_content(), "quoted");
    }

    #[test]
    fn mark_sort_deterministic() {
        let mut marks = vec![Mark::Code, Mark::Bold, Mark::Link { href: "x".into() }];
        Mark::sort_marks(&mut marks);
        assert_eq!(
            marks,
            vec![Mark::Link { href: "x".into() }, Mark::Bold, Mark::Code]
        );
    }

    #[test]
    fn hard_break_text_content() {
        let para = Node::paragraph(vec![
            Node::text("line1"),
            Node::HardBreak,
            Node::text("line2"),
        ]);
        assert_eq!(para.text_content(), "line1\nline2");
    }
}
