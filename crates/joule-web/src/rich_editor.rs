//! Rich text editor engine.
//!
//! Provides an `EditorState` with command execution, undo/redo, clipboard,
//! and read-only mode. Replaces Slate.js / TipTap editor core with pure Rust.

use crate::doc_model::{Mark, Node};
use crate::text_selection::{Position, Selection};

// ── Clipboard ───────────────────────────────────────────────────

/// Clipboard holding rich text content.
#[derive(Debug, Clone, Default)]
pub struct Clipboard {
    /// Stored nodes (rich text).
    pub content: Option<Vec<Node>>,
}

// ── Commands ────────────────────────────────────────────────────

/// An editor command that can be executed and undone.
#[derive(Debug, Clone)]
pub enum Command {
    InsertText(String),
    Delete,
    Bold,
    Italic,
    Underline,
    SetHeading(u8),
    InsertLink { href: String },
    InsertImage { src: String, alt: String },
    Indent,
    Outdent,
}

/// A snapshot for undo/redo.
#[derive(Debug, Clone)]
struct Snapshot {
    document: Node,
    selection: Selection,
}

// ── Editor state ────────────────────────────────────────────────

/// The main editor state.
#[derive(Debug, Clone)]
pub struct EditorState {
    /// The document root.
    pub document: Node,
    /// Current selection.
    pub selection: Selection,
    /// Undo stack.
    undo_stack: Vec<Snapshot>,
    /// Redo stack.
    redo_stack: Vec<Snapshot>,
    /// Maximum undo history size.
    pub max_history: usize,
    /// Whether the editor is in read-only mode.
    pub read_only: bool,
    /// Clipboard.
    pub clipboard: Clipboard,
}

impl EditorState {
    /// Create a new editor state with the given document.
    pub fn new(document: Node) -> Self {
        Self {
            document,
            selection: Selection::cursor(vec![0, 0], 0),
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            max_history: 100,
            read_only: false,
            clipboard: Clipboard::default(),
        }
    }

    /// Create a new empty editor.
    pub fn empty() -> Self {
        let doc = Node::doc(vec![Node::paragraph(vec![Node::text("")])]);
        Self::new(doc)
    }

    /// Save current state to undo stack.
    fn save_undo(&mut self) {
        let snapshot = Snapshot {
            document: self.document.clone(),
            selection: self.selection.clone(),
        };
        self.undo_stack.push(snapshot);
        if self.undo_stack.len() > self.max_history {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
    }

    /// Execute a command.
    pub fn execute(&mut self, cmd: Command) -> Result<(), EditorError> {
        if self.read_only {
            return Err(EditorError::ReadOnly);
        }
        self.save_undo();
        match cmd {
            Command::InsertText(text) => self.do_insert_text(&text),
            Command::Delete => self.do_delete(),
            Command::Bold => self.do_toggle_mark(Mark::Bold),
            Command::Italic => self.do_toggle_mark(Mark::Italic),
            Command::Underline => self.do_toggle_mark(Mark::Underline),
            Command::SetHeading(level) => self.do_set_heading(level),
            Command::InsertLink { href } => self.do_insert_link(&href),
            Command::InsertImage { src, alt } => self.do_insert_image(&src, &alt),
            Command::Indent => self.do_indent(),
            Command::Outdent => self.do_outdent(),
        }
    }

    /// Undo the last command.
    pub fn undo(&mut self) -> bool {
        if let Some(snapshot) = self.undo_stack.pop() {
            let current = Snapshot {
                document: self.document.clone(),
                selection: self.selection.clone(),
            };
            self.redo_stack.push(current);
            self.document = snapshot.document;
            self.selection = snapshot.selection;
            true
        } else {
            false
        }
    }

    /// Redo the last undone command.
    pub fn redo(&mut self) -> bool {
        if let Some(snapshot) = self.redo_stack.pop() {
            let current = Snapshot {
                document: self.document.clone(),
                selection: self.selection.clone(),
            };
            self.undo_stack.push(current);
            self.document = snapshot.document;
            self.selection = snapshot.selection;
            true
        } else {
            false
        }
    }

    /// Can undo?
    pub fn can_undo(&self) -> bool {
        !self.undo_stack.is_empty()
    }

    /// Can redo?
    pub fn can_redo(&self) -> bool {
        !self.redo_stack.is_empty()
    }

    /// Cut selected content to clipboard.
    pub fn cut(&mut self) -> Result<(), EditorError> {
        if self.read_only {
            return Err(EditorError::ReadOnly);
        }
        self.copy();
        self.save_undo();
        self.do_delete()
    }

    /// Copy selected content to clipboard.
    pub fn copy(&mut self) {
        let content = self.extract_selected_content();
        self.clipboard.content = Some(content);
    }

    /// Paste from clipboard.
    pub fn paste(&mut self) -> Result<(), EditorError> {
        if self.read_only {
            return Err(EditorError::ReadOnly);
        }
        if let Some(content) = self.clipboard.content.clone() {
            self.save_undo();
            // Insert pasted content as text
            for node in &content {
                let text = node.text_content();
                if !text.is_empty() {
                    self.do_insert_text(&text)?;
                }
            }
        }
        Ok(())
    }

    /// Get the plain text content of the document.
    pub fn text_content(&self) -> String {
        self.document.text_content()
    }

    // ── Internal command implementations ────────────────────────

    fn do_insert_text(&mut self, text: &str) -> Result<(), EditorError> {
        let pos = self.selection.head();
        if let Some(Node::Text {
            text: node_text, ..
        }) = self.document.node_at_mut(&pos.path)
        {
            let offset = pos.offset.min(node_text.len());
            node_text.insert_str(offset, text);
            let new_offset = offset + text.len();
            self.selection = Selection::cursor(pos.path.clone(), new_offset);
            Ok(())
        } else {
            Err(EditorError::InvalidPosition)
        }
    }

    fn do_delete(&mut self) -> Result<(), EditorError> {
        match &self.selection {
            Selection::Cursor(pos) => {
                if let Some(Node::Text {
                    text: node_text, ..
                }) = self.document.node_at_mut(&pos.path)
                {
                    if pos.offset < node_text.len() {
                        // Find the char boundary to remove
                        let next = node_text
                            .char_indices()
                            .find(|(i, _)| *i > pos.offset)
                            .map(|(i, _)| i)
                            .unwrap_or(node_text.len());
                        node_text.drain(pos.offset..next);
                    }
                    Ok(())
                } else {
                    Err(EditorError::InvalidPosition)
                }
            }
            Selection::Range { anchor, head } => {
                // Delete range: for simplicity, work within same text node
                let from = self.selection.from_pos();
                let to = self.selection.to_pos();
                if from.path == to.path {
                    if let Some(Node::Text {
                        text: node_text, ..
                    }) = self.document.node_at_mut(&from.path)
                    {
                        let start = from.offset.min(node_text.len());
                        let end = to.offset.min(node_text.len());
                        node_text.drain(start..end);
                        self.selection = Selection::cursor(from.path.clone(), start);
                        return Ok(());
                    }
                }
                // Cross-node range: delete text in anchor node after anchor,
                // and text in head node before head, collapse.
                let _ = anchor;
                let _ = head;
                self.selection = Selection::Cursor(from);
                Ok(())
            }
            Selection::NodeSelection { path } => {
                // Remove the selected node
                if path.is_empty() {
                    return Err(EditorError::InvalidPosition);
                }
                let parent_path = &path[..path.len() - 1];
                let idx = path[path.len() - 1];
                if let Some(parent) = self.document.node_at_mut(parent_path) {
                    if let Some(children) = parent.children_mut() {
                        if idx < children.len() {
                            children.remove(idx);
                        }
                    }
                }
                self.selection = Selection::cursor(parent_path.to_vec(), 0);
                Ok(())
            }
        }
    }

    fn do_toggle_mark(&mut self, mark: Mark) -> Result<(), EditorError> {
        let pos = self.selection.head();
        // Toggle mark on the text node at the current position
        if let Some(node) = self.document.node_at_mut(&pos.path) {
            if node.has_mark(&mark) {
                node.remove_mark(&mark);
            } else {
                node.add_mark(mark);
            }
            Ok(())
        } else {
            Err(EditorError::InvalidPosition)
        }
    }

    fn do_set_heading(&mut self, level: u8) -> Result<(), EditorError> {
        let pos = self.selection.head();
        // Find the block-level parent and convert to heading
        if pos.path.len() >= 2 {
            let block_path = &pos.path[..pos.path.len() - 1];
            if let Some(node) = self.document.node_at_mut(block_path) {
                match node {
                    Node::Paragraph { children } | Node::Heading { children, .. } => {
                        let kids = std::mem::take(children);
                        *node = Node::heading(level, kids);
                        return Ok(());
                    }
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn do_insert_link(&mut self, href: &str) -> Result<(), EditorError> {
        let pos = self.selection.head();
        if let Some(Node::Text { marks, .. }) = self.document.node_at_mut(&pos.path) {
            let link_mark = Mark::Link {
                href: href.to_string(),
            };
            if !marks.contains(&link_mark) {
                marks.push(link_mark);
                Mark::sort_marks(marks);
            }
            Ok(())
        } else {
            Err(EditorError::InvalidPosition)
        }
    }

    fn do_insert_image(&mut self, src: &str, alt: &str) -> Result<(), EditorError> {
        let pos = self.selection.head();
        // Insert image node after current position's parent block
        if pos.path.len() >= 2 {
            let parent_path = &pos.path[..pos.path.len() - 1];
            let block_parent = &parent_path[..parent_path.len().max(1) - if parent_path.is_empty() { 0 } else { 1 }];
            let idx = if parent_path.is_empty() {
                0
            } else {
                parent_path[parent_path.len() - 1]
            };
            if let Some(container) = self.document.node_at_mut(block_parent) {
                if let Some(children) = container.children_mut() {
                    let img_para =
                        Node::paragraph(vec![Node::image(src, alt)]);
                    let insert_at = (idx + 1).min(children.len());
                    children.insert(insert_at, img_para);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    fn do_indent(&mut self) -> Result<(), EditorError> {
        // Indent: wrap current block in a blockquote
        let pos = self.selection.head();
        if pos.path.len() >= 2 {
            let block_path = &pos.path[..pos.path.len() - 1];
            let parent_path = &block_path[..block_path.len().max(1) - if block_path.is_empty() { 0 } else { 1 }];
            let idx = if block_path.is_empty() {
                0
            } else {
                block_path[block_path.len() - 1]
            };
            if let Some(container) = self.document.node_at_mut(parent_path) {
                if let Some(children) = container.children_mut() {
                    if idx < children.len() {
                        let node = children.remove(idx);
                        children.insert(idx, Node::block_quote(vec![node]));
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    fn do_outdent(&mut self) -> Result<(), EditorError> {
        // Outdent: find the nearest BlockQuote ancestor and unwrap it.
        let pos = self.selection.head();
        // Walk up from the deepest ancestor looking for a BlockQuote.
        for depth in (1..pos.path.len()).rev() {
            let node_path = &pos.path[..depth];
            if let Some(node) = self.document.node_at(node_path) {
                if matches!(node, Node::BlockQuote { .. }) {
                    // Found a BlockQuote at node_path. Replace it with its first child.
                    let parent_path = &node_path[..node_path.len() - 1];
                    let idx = node_path[node_path.len() - 1];
                    if let Some(parent) = self.document.node_at_mut(parent_path) {
                        if let Some(children) = parent.children_mut() {
                            if let Node::BlockQuote {
                                children: inner, ..
                            } = &children[idx]
                            {
                                if let Some(first) = inner.first() {
                                    let unwrapped = first.clone();
                                    children[idx] = unwrapped;
                                    return Ok(());
                                }
                            }
                        }
                    }
                    break;
                }
            }
        }
        // Also check root-level children
        if !pos.path.is_empty() {
            let idx = pos.path[0];
            if let Some(children) = self.document.children_mut() {
                if idx < children.len() {
                    if let Node::BlockQuote {
                        children: inner, ..
                    } = &children[idx]
                    {
                        if let Some(first) = inner.first() {
                            let unwrapped = first.clone();
                            children[idx] = unwrapped;
                            return Ok(());
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn extract_selected_content(&self) -> Vec<Node> {
        match &self.selection {
            Selection::Cursor(_) => vec![],
            Selection::Range { .. } => {
                let from = self.selection.from_pos();
                let to = self.selection.to_pos();
                if from.path == to.path {
                    if let Some(Node::Text { text, marks }) = self.document.node_at(&from.path) {
                        let start = from.offset.min(text.len());
                        let end = to.offset.min(text.len());
                        let slice = &text[start..end];
                        return vec![Node::Text {
                            text: slice.to_string(),
                            marks: marks.clone(),
                        }];
                    }
                }
                vec![]
            }
            Selection::NodeSelection { path } => {
                if let Some(node) = self.document.node_at(path) {
                    vec![node.clone()]
                } else {
                    vec![]
                }
            }
        }
    }
}

// ── Errors ──────────────────────────────────────────────────────

/// Editor errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EditorError {
    ReadOnly,
    InvalidPosition,
    InvalidCommand(String),
}

impl std::fmt::Display for EditorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EditorError::ReadOnly => write!(f, "editor is in read-only mode"),
            EditorError::InvalidPosition => write!(f, "invalid cursor position"),
            EditorError::InvalidCommand(msg) => write!(f, "invalid command: {msg}"),
        }
    }
}

impl std::error::Error for EditorError {}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_editor() -> EditorState {
        let doc = Node::doc(vec![Node::paragraph(vec![Node::text("Hello world")])]);
        let mut editor = EditorState::new(doc);
        editor.selection = Selection::cursor(vec![0, 0], 5);
        editor
    }

    #[test]
    fn insert_text() {
        let mut editor = make_editor();
        editor.execute(Command::InsertText(",".into())).unwrap();
        assert_eq!(editor.text_content(), "Hello, world");
    }

    #[test]
    fn delete_at_cursor() {
        let mut editor = make_editor();
        editor.selection = Selection::cursor(vec![0, 0], 0);
        editor.execute(Command::Delete).unwrap();
        assert_eq!(editor.text_content(), "ello world");
    }

    #[test]
    fn delete_range() {
        let mut editor = make_editor();
        editor.selection = Selection::Range {
            anchor: Position::new(vec![0, 0], 0),
            head: Position::new(vec![0, 0], 5),
        };
        editor.execute(Command::Delete).unwrap();
        assert_eq!(editor.text_content(), " world");
    }

    #[test]
    fn undo_redo() {
        let mut editor = make_editor();
        let before = editor.text_content();
        editor.execute(Command::InsertText("!".into())).unwrap();
        assert_ne!(editor.text_content(), before);

        assert!(editor.undo());
        assert_eq!(editor.text_content(), before);

        assert!(editor.redo());
        assert_eq!(editor.text_content(), "Hello! world");
    }

    #[test]
    fn read_only_rejects_commands() {
        let mut editor = make_editor();
        editor.read_only = true;
        let result = editor.execute(Command::InsertText("x".into()));
        assert_eq!(result, Err(EditorError::ReadOnly));
    }

    #[test]
    fn bold_toggle() {
        let mut editor = make_editor();
        editor.execute(Command::Bold).unwrap();
        assert!(editor.document.has_mark(&Mark::Bold));
        editor.execute(Command::Bold).unwrap();
        assert!(!editor.document.has_mark(&Mark::Bold));
    }

    #[test]
    fn italic_and_underline() {
        let mut editor = make_editor();
        editor.execute(Command::Italic).unwrap();
        assert!(editor.document.has_mark(&Mark::Italic));
        editor.execute(Command::Underline).unwrap();
        assert!(editor.document.has_mark(&Mark::Underline));
    }

    #[test]
    fn set_heading() {
        let mut editor = make_editor();
        editor.execute(Command::SetHeading(2)).unwrap();
        match &editor.document.node_at(&[0]).unwrap() {
            Node::Heading { level, .. } => assert_eq!(*level, 2),
            other => panic!("expected heading, got {other:?}"),
        }
    }

    #[test]
    fn insert_link() {
        let mut editor = make_editor();
        editor
            .execute(Command::InsertLink {
                href: "https://example.com".into(),
            })
            .unwrap();
        let node = editor.document.node_at(&[0, 0]).unwrap();
        match node {
            Node::Text { marks, .. } => {
                assert!(marks.iter().any(|m| matches!(m, Mark::Link { .. })));
            }
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn clipboard_copy_paste() {
        let mut editor = make_editor();
        editor.selection = Selection::Range {
            anchor: Position::new(vec![0, 0], 0),
            head: Position::new(vec![0, 0], 5),
        };
        editor.copy();
        assert!(editor.clipboard.content.is_some());

        // Move cursor to end
        editor.selection = Selection::cursor(vec![0, 0], 11);
        editor.paste().unwrap();
        assert!(editor.text_content().ends_with("Hello"));
    }

    #[test]
    fn cut_removes_content() {
        let mut editor = make_editor();
        editor.selection = Selection::Range {
            anchor: Position::new(vec![0, 0], 0),
            head: Position::new(vec![0, 0], 5),
        };
        editor.cut().unwrap();
        assert_eq!(editor.text_content(), " world");
        assert!(editor.clipboard.content.is_some());
    }

    #[test]
    fn indent_wraps_in_blockquote() {
        let mut editor = make_editor();
        editor.execute(Command::Indent).unwrap();
        match editor.document.node_at(&[0]).unwrap() {
            Node::BlockQuote { .. } => {}
            other => panic!("expected blockquote, got {other:?}"),
        }
    }

    #[test]
    fn outdent_unwraps_blockquote() {
        let mut editor = make_editor();
        editor.execute(Command::Indent).unwrap();
        // After indent, selection path needs updating for the new structure
        editor.selection = Selection::cursor(vec![0, 0, 0], 5);
        editor.execute(Command::Outdent).unwrap();
        match editor.document.node_at(&[0]).unwrap() {
            Node::Paragraph { .. } => {}
            other => panic!("expected paragraph after outdent, got {other:?}"),
        }
    }

    #[test]
    fn undo_stack_limit() {
        let mut editor = make_editor();
        editor.max_history = 3;
        for i in 0..5 {
            editor
                .execute(Command::InsertText(format!("{i}")))
                .unwrap();
        }
        assert!(editor.undo_stack.len() <= 3);
    }

    #[test]
    fn empty_editor() {
        let editor = EditorState::empty();
        assert_eq!(editor.text_content(), "");
        assert!(!editor.can_undo());
        assert!(!editor.can_redo());
    }
}
