//! Operational transformation.
//!
//! Insert, delete, and retain operations for collaborative text editing.
//! Supports transforming pairs of concurrent operations, composing sequential
//! operations, applying to strings, cursor position transform, operation
//! inversion, and history tracking.

use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// Error type for OT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtError {
    /// The operation does not match the document length.
    LengthMismatch { expected: usize, actual: usize },
    /// Composition failed due to incompatible operations.
    ComposeFailed(String),
    /// Transform failed.
    TransformFailed(String),
    /// The operation could not be applied.
    ApplyFailed(String),
}

impl fmt::Display for OtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => {
                write!(f, "length mismatch: expected {}, got {}", expected, actual)
            }
            Self::ComposeFailed(s) => write!(f, "compose failed: {}", s),
            Self::TransformFailed(s) => write!(f, "transform failed: {}", s),
            Self::ApplyFailed(s) => write!(f, "apply failed: {}", s),
        }
    }
}

/// A single component of an operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpComponent {
    /// Retain `n` characters (skip them unchanged).
    Retain(usize),
    /// Insert a string at the current position.
    Insert(String),
    /// Delete `n` characters at the current position.
    Delete(usize),
}

/// An operation: a sequence of components that transforms a document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operation {
    /// The components of this operation.
    pub components: Vec<OpComponent>,
    /// The length of the document this operation expects as input.
    pub base_len: usize,
    /// The length of the document after this operation is applied.
    pub target_len: usize,
}

impl Operation {
    /// Create a new empty operation for a document of the given length.
    pub fn new() -> Self {
        Self {
            components: Vec::new(),
            base_len: 0,
            target_len: 0,
        }
    }

    /// Append a retain component.
    pub fn retain(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        self.base_len += n;
        self.target_len += n;
        // Merge with previous retain if possible.
        if let Some(OpComponent::Retain(prev)) = self.components.last_mut() {
            *prev += n;
        } else {
            self.components.push(OpComponent::Retain(n));
        }
    }

    /// Append an insert component.
    pub fn insert(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        let len = s.chars().count();
        self.target_len += len;
        // Merge with previous insert if possible.
        if let Some(OpComponent::Insert(prev)) = self.components.last_mut() {
            prev.push_str(s);
        } else {
            self.components.push(OpComponent::Insert(s.to_string()));
        }
    }

    /// Append a delete component.
    pub fn delete(&mut self, n: usize) {
        if n == 0 {
            return;
        }
        self.base_len += n;
        // Merge with previous delete if possible.
        if let Some(OpComponent::Delete(prev)) = self.components.last_mut() {
            *prev += n;
        } else {
            self.components.push(OpComponent::Delete(n));
        }
    }

    /// Check if this operation is a no-op.
    pub fn is_noop(&self) -> bool {
        self.components.iter().all(|c| matches!(c, OpComponent::Retain(_)))
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, comp) in self.components.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            match comp {
                OpComponent::Retain(n) => write!(f, "retain({})", n)?,
                OpComponent::Insert(s) => write!(f, "insert(\"{}\")", s)?,
                OpComponent::Delete(n) => write!(f, "delete({})", n)?,
            }
        }
        Ok(())
    }
}

// ── Apply ──────────────────────────────────────────────────────────

/// Apply an operation to a string, producing a new string.
pub fn apply(doc: &str, op: &Operation) -> Result<String, OtError> {
    let chars: Vec<char> = doc.chars().collect();
    let doc_len = chars.len();
    if op.base_len != doc_len {
        return Err(OtError::LengthMismatch {
            expected: op.base_len,
            actual: doc_len,
        });
    }

    let mut result = String::new();
    let mut pos = 0;

    for comp in &op.components {
        match comp {
            OpComponent::Retain(n) => {
                let end = pos + n;
                if end > chars.len() {
                    return Err(OtError::ApplyFailed(format!(
                        "retain past end: pos={}, n={}, len={}",
                        pos, n, chars.len()
                    )));
                }
                for c in &chars[pos..end] {
                    result.push(*c);
                }
                pos = end;
            }
            OpComponent::Insert(s) => {
                result.push_str(s);
            }
            OpComponent::Delete(n) => {
                let end = pos + n;
                if end > chars.len() {
                    return Err(OtError::ApplyFailed(format!(
                        "delete past end: pos={}, n={}, len={}",
                        pos, n, chars.len()
                    )));
                }
                pos = end;
            }
        }
    }

    // Remaining characters (shouldn't happen if base_len is correct).
    for c in &chars[pos..] {
        result.push(*c);
    }

    Ok(result)
}

// ── Transform ──────────────────────────────────────────────────────

/// Transform two concurrent operations so they can be applied in sequence.
///
/// Given operations `a` and `b` that were both generated against the same
/// document state, returns `(a', b')` such that:
///   apply(apply(doc, a), b') == apply(apply(doc, b), a')
pub fn transform(a: &Operation, b: &Operation) -> Result<(Operation, Operation), OtError> {
    if a.base_len != b.base_len {
        return Err(OtError::TransformFailed(format!(
            "base lengths differ: {} vs {}",
            a.base_len, b.base_len
        )));
    }

    let mut a_prime = Operation::new();
    let mut b_prime = Operation::new();

    let mut ai = ComponentIter::new(&a.components);
    let mut bi = ComponentIter::new(&b.components);

    loop {
        let a_comp = ai.peek();
        let b_comp = bi.peek();

        if a_comp.is_none() && b_comp.is_none() {
            break;
        }

        // If a inserts, a' inserts and b' retains over it.
        if let Some(OpComponent::Insert(s)) = a_comp {
            let len = s.chars().count();
            a_prime.insert(&s);
            b_prime.retain(len);
            ai.advance();
            continue;
        }

        // If b inserts, b' inserts and a' retains over it.
        if let Some(OpComponent::Insert(s)) = b_comp {
            let len = s.chars().count();
            b_prime.insert(&s);
            a_prime.retain(len);
            bi.advance();
            continue;
        }

        match (a_comp, b_comp) {
            (Some(OpComponent::Retain(an)), Some(OpComponent::Retain(bn))) => {
                let min = an.min(bn);
                a_prime.retain(min);
                b_prime.retain(min);
                ai.consume(min);
                bi.consume(min);
            }
            (Some(OpComponent::Delete(an)), Some(OpComponent::Delete(bn))) => {
                let min = an.min(bn);
                // Both delete the same characters — they cancel out.
                ai.consume(min);
                bi.consume(min);
            }
            (Some(OpComponent::Delete(an)), Some(OpComponent::Retain(bn))) => {
                let min = an.min(bn);
                a_prime.delete(min);
                ai.consume(min);
                bi.consume(min);
            }
            (Some(OpComponent::Retain(an)), Some(OpComponent::Delete(bn))) => {
                let min = an.min(bn);
                b_prime.delete(min);
                ai.consume(min);
                bi.consume(min);
            }
            _ => break,
        }
    }

    Ok((a_prime, b_prime))
}

/// Iterator over operation components, supporting partial consumption.
struct ComponentIter<'a> {
    components: &'a [OpComponent],
    index: usize,
    offset: usize, // partial consumption within current component
}

impl<'a> ComponentIter<'a> {
    fn new(components: &'a [OpComponent]) -> Self {
        Self {
            components,
            index: 0,
            offset: 0,
        }
    }

    fn peek(&self) -> Option<OpComponent> {
        let comp = self.components.get(self.index)?;
        Some(match comp {
            OpComponent::Retain(n) => OpComponent::Retain(*n - self.offset),
            OpComponent::Insert(s) => {
                if self.offset == 0 {
                    OpComponent::Insert(s.clone())
                } else {
                    let remaining: String = s.chars().skip(self.offset).collect();
                    OpComponent::Insert(remaining)
                }
            }
            OpComponent::Delete(n) => OpComponent::Delete(*n - self.offset),
        })
    }

    fn advance(&mut self) {
        self.index += 1;
        self.offset = 0;
    }

    fn consume(&mut self, n: usize) {
        let comp = &self.components[self.index];
        let total = match comp {
            OpComponent::Retain(t) => *t,
            OpComponent::Delete(t) => *t,
            OpComponent::Insert(s) => s.chars().count(),
        };
        let remaining = total - self.offset;
        if n >= remaining {
            self.index += 1;
            self.offset = 0;
        } else {
            self.offset += n;
        }
    }
}

// ── Compose ────────────────────────────────────────────────────────

/// Compose two sequential operations into one.
///
/// Given `a` applied first and `b` applied second, returns an operation `c`
/// such that `apply(doc, c) == apply(apply(doc, a), b)`.
pub fn compose(a: &Operation, b: &Operation) -> Result<Operation, OtError> {
    if a.target_len != b.base_len {
        return Err(OtError::ComposeFailed(format!(
            "a.target_len ({}) != b.base_len ({})",
            a.target_len, b.base_len
        )));
    }

    let mut result = Operation::new();
    let mut ai = ComponentIter::new(&a.components);
    let mut bi = ComponentIter::new(&b.components);

    loop {
        let a_comp = ai.peek();
        let b_comp = bi.peek();

        if a_comp.is_none() && b_comp.is_none() {
            break;
        }

        // If a inserts, check what b does with those characters.
        if let Some(OpComponent::Delete(an)) = a_comp {
            result.delete(an);
            ai.advance();
            continue;
        }

        // If b inserts, it goes straight to result.
        if let Some(OpComponent::Insert(s)) = b_comp {
            result.insert(&s);
            bi.advance();
            continue;
        }

        match (a_comp, b_comp) {
            (Some(OpComponent::Retain(an)), Some(OpComponent::Retain(bn))) => {
                let min = an.min(bn);
                result.retain(min);
                ai.consume(min);
                bi.consume(min);
            }
            (Some(OpComponent::Insert(s)), Some(OpComponent::Delete(bn))) => {
                let len = s.chars().count();
                let min = len.min(bn);
                // b deletes what a inserted — they cancel.
                ai.consume(min);
                bi.consume(min);
            }
            (Some(OpComponent::Insert(s)), Some(OpComponent::Retain(bn))) => {
                let len = s.chars().count();
                let min = len.min(bn);
                let insert_text: String = s.chars().take(min).collect();
                result.insert(&insert_text);
                ai.consume(min);
                bi.consume(min);
            }
            (Some(OpComponent::Retain(an)), Some(OpComponent::Delete(bn))) => {
                let min = an.min(bn);
                result.delete(min);
                ai.consume(min);
                bi.consume(min);
            }
            _ => break,
        }
    }

    Ok(result)
}

// ── Inversion ──────────────────────────────────────────────────────

/// Invert an operation given the document it was applied to.
/// The inverted operation undoes the original.
pub fn invert(op: &Operation, doc: &str) -> Operation {
    let chars: Vec<char> = doc.chars().collect();
    let mut inv = Operation::new();
    let mut pos = 0;

    for comp in &op.components {
        match comp {
            OpComponent::Retain(n) => {
                inv.retain(*n);
                pos += n;
            }
            OpComponent::Insert(s) => {
                let len = s.chars().count();
                inv.delete(len);
            }
            OpComponent::Delete(n) => {
                let end = (pos + n).min(chars.len());
                let deleted: String = chars[pos..end].iter().collect();
                inv.insert(&deleted);
                pos = end;
            }
        }
    }

    inv
}

// ── Cursor transform ──────────────────────────────────────────────

/// Transform a cursor position through an operation.
pub fn transform_cursor(cursor: usize, op: &Operation) -> usize {
    let mut new_cursor = cursor;
    let mut pos = 0;

    for comp in &op.components {
        if pos > cursor {
            break;
        }
        match comp {
            OpComponent::Retain(n) => {
                pos += n;
            }
            OpComponent::Insert(s) => {
                let len = s.chars().count();
                if pos <= cursor {
                    new_cursor += len;
                }
            }
            OpComponent::Delete(n) => {
                let end = pos + n;
                if cursor <= pos {
                    // Cursor before deletion — no change.
                } else if cursor < end {
                    // Cursor within deletion — move to deletion start.
                    new_cursor -= cursor - pos;
                } else {
                    // Cursor after deletion.
                    new_cursor -= n;
                }
                pos = end;
            }
        }
    }

    new_cursor
}

// ── History ────────────────────────────────────────────────────────

/// A history of operations for undo/redo.
#[derive(Debug, Clone)]
pub struct History {
    /// Past operations and the document state before each.
    undo_stack: Vec<(Operation, String)>,
    /// Redo stack.
    redo_stack: Vec<(Operation, String)>,
    /// Current document state.
    current: String,
    /// Maximum history depth.
    max_depth: usize,
}

impl History {
    /// Create a new history with the initial document.
    pub fn new(doc: &str, max_depth: usize) -> Self {
        Self {
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            current: doc.to_string(),
            max_depth,
        }
    }

    /// Apply an operation and record it in history.
    pub fn apply(&mut self, op: &Operation) -> Result<String, OtError> {
        let old_doc = self.current.clone();
        let new_doc = apply(&old_doc, op)?;
        self.undo_stack.push((op.clone(), old_doc));
        if self.undo_stack.len() > self.max_depth {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
        self.current = new_doc.clone();
        Ok(new_doc)
    }

    /// Undo the last operation.
    pub fn undo(&mut self) -> Option<String> {
        let (op, old_doc) = self.undo_stack.pop()?;
        let inv = invert(&op, &old_doc);
        self.redo_stack.push((op, self.current.clone()));
        self.current = old_doc;
        Some(self.current.clone())
    }

    /// Redo the last undone operation.
    pub fn redo(&mut self) -> Option<String> {
        let (op, old_doc) = self.redo_stack.pop()?;
        let result = apply(&self.current, &op).ok()?;
        self.undo_stack.push((op, self.current.clone()));
        self.current = result.clone();
        Some(result)
    }

    /// Get the current document state.
    pub fn current(&self) -> &str {
        &self.current
    }

    /// Number of undoable operations.
    pub fn undo_count(&self) -> usize {
        self.undo_stack.len()
    }

    /// Number of redoable operations.
    pub fn redo_count(&self) -> usize {
        self.redo_stack.len()
    }
}

// ── Convenience builders ───────────────────────────────────────────

/// Build a simple insert operation at a given position.
pub fn insert_at(pos: usize, text: &str, doc_len: usize) -> Operation {
    let mut op = Operation::new();
    if pos > 0 {
        op.retain(pos);
    }
    op.insert(text);
    let remaining = doc_len - pos;
    if remaining > 0 {
        op.retain(remaining);
    }
    op
}

/// Build a simple delete operation.
pub fn delete_at(pos: usize, count: usize, doc_len: usize) -> Operation {
    let mut op = Operation::new();
    if pos > 0 {
        op.retain(pos);
    }
    op.delete(count);
    let remaining = doc_len - pos - count;
    if remaining > 0 {
        op.retain(remaining);
    }
    op
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_insert() {
        let op = insert_at(5, "world", 5);
        let result = apply("hello", &op).unwrap();
        assert_eq!(result, "helloworld");
    }

    #[test]
    fn apply_delete() {
        let op = delete_at(5, 5, 10);
        let result = apply("helloworld", &op).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn apply_retain_only() {
        let mut op = Operation::new();
        op.retain(5);
        let result = apply("hello", &op).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn apply_length_mismatch() {
        let op = insert_at(0, "x", 10);
        let result = apply("hi", &op);
        assert!(result.is_err());
    }

    #[test]
    fn compose_basic() {
        let a = insert_at(0, "hello", 0);
        let b = insert_at(5, " world", 5);
        let c = compose(&a, &b).unwrap();
        let result = apply("", &c).unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn compose_length_mismatch() {
        let a = insert_at(0, "hi", 0);
        let mut b = Operation::new();
        b.retain(100);
        let result = compose(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn transform_concurrent_inserts() {
        let doc = "hello";
        let a = insert_at(5, "A", 5);
        let b = insert_at(0, "B", 5);
        let (a_prime, b_prime) = transform(&a, &b).unwrap();

        let via_a = apply(doc, &a).unwrap();
        let via_a_then_b = apply(&via_a, &b_prime).unwrap();

        let via_b = apply(doc, &b).unwrap();
        let via_b_then_a = apply(&via_b, &a_prime).unwrap();

        assert_eq!(via_a_then_b, via_b_then_a);
    }

    #[test]
    fn transform_base_mismatch() {
        let a = insert_at(0, "x", 5);
        let b = insert_at(0, "y", 10);
        let result = transform(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn invert_insert() {
        let doc = "hello";
        let op = insert_at(5, " world", 5);
        let result = apply(doc, &op).unwrap();
        assert_eq!(result, "hello world");

        let inv = invert(&op, doc);
        let back = apply(&result, &inv).unwrap();
        assert_eq!(back, doc);
    }

    #[test]
    fn invert_delete() {
        let doc = "hello world";
        let op = delete_at(5, 6, 11);
        let result = apply(doc, &op).unwrap();
        assert_eq!(result, "hello");

        let inv = invert(&op, doc);
        let back = apply(&result, &inv).unwrap();
        assert_eq!(back, doc);
    }

    #[test]
    fn cursor_transform_insert_before() {
        let op = insert_at(0, "xxx", 5);
        let new_cursor = transform_cursor(3, &op);
        assert_eq!(new_cursor, 6); // shifted right by 3
    }

    #[test]
    fn cursor_transform_delete_before() {
        let op = delete_at(0, 2, 5);
        let new_cursor = transform_cursor(4, &op);
        assert_eq!(new_cursor, 2); // shifted left by 2
    }

    #[test]
    fn cursor_in_deleted_region() {
        let op = delete_at(2, 3, 7);
        let new_cursor = transform_cursor(3, &op);
        assert_eq!(new_cursor, 2); // moved to deletion start
    }

    #[test]
    fn history_undo_redo() {
        let mut hist = History::new("hello", 10);
        let op = insert_at(5, " world", 5);
        hist.apply(&op).unwrap();
        assert_eq!(hist.current(), "hello world");

        hist.undo();
        assert_eq!(hist.current(), "hello");

        hist.redo();
        assert_eq!(hist.current(), "hello world");
    }

    #[test]
    fn history_max_depth() {
        let mut hist = History::new("", 2);
        for i in 0..5 {
            let op = insert_at(i, "a", i);
            hist.apply(&op).unwrap();
        }
        assert!(hist.undo_count() <= 2);
    }

    #[test]
    fn history_redo_cleared_on_new_edit() {
        let mut hist = History::new("ab", 10);
        let op1 = insert_at(2, "c", 2);
        hist.apply(&op1).unwrap();
        hist.undo();
        assert_eq!(hist.redo_count(), 1);

        let op2 = insert_at(2, "d", 2);
        hist.apply(&op2).unwrap();
        assert_eq!(hist.redo_count(), 0);
    }

    #[test]
    fn operation_display() {
        let op = insert_at(3, "hi", 5);
        let s = format!("{}", op);
        assert!(s.contains("retain(3)"));
        assert!(s.contains("insert(\"hi\")"));
    }

    #[test]
    fn operation_is_noop() {
        let mut op = Operation::new();
        op.retain(5);
        assert!(op.is_noop());

        let op2 = insert_at(0, "x", 5);
        assert!(!op2.is_noop());
    }

    #[test]
    fn error_display() {
        let e = OtError::LengthMismatch {
            expected: 5,
            actual: 3,
        };
        assert!(e.to_string().contains("5"));
    }

    #[test]
    fn component_merging() {
        let mut op = Operation::new();
        op.retain(3);
        op.retain(2);
        assert_eq!(op.components.len(), 1);
        assert_eq!(op.components[0], OpComponent::Retain(5));
    }

    #[test]
    fn empty_insert_ignored() {
        let mut op = Operation::new();
        op.insert("");
        assert!(op.components.is_empty());
    }

    #[test]
    fn zero_retain_ignored() {
        let mut op = Operation::new();
        op.retain(0);
        assert!(op.components.is_empty());
    }
}
