//! CRDT for collaborative text editing.
//!
//! Implements a Replicated Growable Array (RGA) for conflict-free replicated
//! text. Each character has a unique ID (site, sequence), supporting
//! concurrent inserts with deterministic resolution, delete via tombstones,
//! local/remote operation application, causal ordering, and state merging.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;

// ── Types ──────────────────────────────────────────────────────────

/// A unique site identifier for a replica.
pub type SiteId = u32;

/// A unique identifier for a character in the RGA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CharId {
    /// The site that created this character.
    pub site: SiteId,
    /// Monotonically increasing sequence number per site.
    pub seq: u64,
}

impl CharId {
    /// The root ID (precedes all characters).
    pub fn root() -> Self {
        Self { site: 0, seq: 0 }
    }

    /// Whether this is the root sentinel.
    pub fn is_root(&self) -> bool {
        self.site == 0 && self.seq == 0
    }
}

impl Ord for CharId {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher sequence wins, then higher site wins (for tie-breaking).
        self.seq
            .cmp(&other.seq)
            .then(self.site.cmp(&other.site))
    }
}

impl PartialOrd for CharId {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Display for CharId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "({},{})", self.site, self.seq)
    }
}

/// A character node in the RGA.
#[derive(Debug, Clone)]
struct RgaNode {
    id: CharId,
    value: char,
    deleted: bool,
    /// ID of the character this was inserted after.
    parent: CharId,
}

/// An operation on the CRDT.
#[derive(Debug, Clone)]
pub enum CrdtOp {
    /// Insert a character after the given parent.
    Insert {
        id: CharId,
        parent: CharId,
        value: char,
    },
    /// Delete the character with the given ID.
    Delete { id: CharId },
}

/// Error type for CRDT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrdtError {
    /// The referenced character was not found.
    NotFound(CharId),
    /// A duplicate insert was detected.
    DuplicateId(CharId),
    /// The causal dependency is not satisfied.
    CausalityViolation(String),
}

impl fmt::Display for CrdtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound(id) => write!(f, "character not found: {}", id),
            Self::DuplicateId(id) => write!(f, "duplicate id: {}", id),
            Self::CausalityViolation(s) => write!(f, "causality violation: {}", s),
        }
    }
}

/// The RGA document.
#[derive(Debug, Clone)]
pub struct RgaDocument {
    /// Linear sequence of nodes (including tombstones).
    nodes: Vec<RgaNode>,
    /// Index: CharId -> position in `nodes`.
    index: HashMap<CharId, usize>,
    /// Per-site sequence counters for generating IDs.
    clocks: HashMap<SiteId, u64>,
    /// The local site ID.
    site_id: SiteId,
    /// Pending operations waiting for causal dependencies.
    pending: Vec<CrdtOp>,
}

impl RgaDocument {
    /// Create a new empty document for the given site.
    pub fn new(site_id: SiteId) -> Self {
        let root = RgaNode {
            id: CharId::root(),
            value: '\0',
            deleted: true, // Sentinel, never visible.
            parent: CharId::root(),
        };
        let mut index = HashMap::new();
        index.insert(CharId::root(), 0);

        Self {
            nodes: vec![root],
            index,
            clocks: HashMap::new(),
            site_id,
            pending: Vec::new(),
        }
    }

    /// Get the current visible text.
    pub fn text(&self) -> String {
        self.nodes
            .iter()
            .filter(|n| !n.deleted && !n.id.is_root())
            .map(|n| n.value)
            .collect()
    }

    /// Get the number of visible characters.
    pub fn len(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| !n.deleted && !n.id.is_root())
            .count()
    }

    /// Whether the document is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the total number of nodes (including tombstones).
    pub fn total_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Generate the next CharId for this site.
    fn next_id(&mut self) -> CharId {
        let seq = self.clocks.entry(self.site_id).or_insert(0);
        *seq += 1;
        CharId {
            site: self.site_id,
            seq: *seq,
        }
    }

    /// Insert a character at the given visible position (0-based).
    /// Returns the operation to broadcast.
    pub fn insert_at(&mut self, pos: usize, value: char) -> Result<CrdtOp, CrdtError> {
        // Find the parent: the character at position `pos - 1` (or root for pos 0).
        let parent_id = if pos == 0 {
            CharId::root()
        } else {
            self.visible_id_at(pos - 1)?
        };

        let id = self.next_id();
        let op = CrdtOp::Insert {
            id,
            parent: parent_id,
            value,
        };

        self.apply_local(&op)?;
        Ok(op)
    }

    /// Delete the character at the given visible position.
    /// Returns the operation to broadcast.
    pub fn delete_at(&mut self, pos: usize) -> Result<CrdtOp, CrdtError> {
        let id = self.visible_id_at(pos)?;
        let op = CrdtOp::Delete { id };
        self.apply_local(&op)?;
        Ok(op)
    }

    /// Get the CharId of the visible character at position `pos`.
    fn visible_id_at(&self, pos: usize) -> Result<CharId, CrdtError> {
        let mut count = 0;
        for node in &self.nodes {
            if node.deleted || node.id.is_root() {
                continue;
            }
            if count == pos {
                return Ok(node.id);
            }
            count += 1;
        }
        Err(CrdtError::NotFound(CharId {
            site: 0,
            seq: pos as u64,
        }))
    }

    /// Apply a locally generated operation.
    fn apply_local(&mut self, op: &CrdtOp) -> Result<(), CrdtError> {
        match op {
            CrdtOp::Insert { id, parent, value } => {
                self.integrate_insert(*id, *parent, *value)
            }
            CrdtOp::Delete { id } => self.integrate_delete(*id),
        }
    }

    /// Apply a remote operation.
    pub fn apply_remote(&mut self, op: &CrdtOp) -> Result<(), CrdtError> {
        // Update our clock.
        match op {
            CrdtOp::Insert { id, parent, value } => {
                // Check causal dependency: parent must exist.
                if !parent.is_root() && !self.index.contains_key(parent) {
                    self.pending.push(op.clone());
                    return Ok(());
                }
                self.integrate_insert(*id, *parent, *value)?;
                // Update clock.
                let entry = self.clocks.entry(id.site).or_insert(0);
                if id.seq > *entry {
                    *entry = id.seq;
                }
            }
            CrdtOp::Delete { id } => {
                if !self.index.contains_key(id) {
                    self.pending.push(op.clone());
                    return Ok(());
                }
                self.integrate_delete(*id)?;
            }
        }

        // Try to apply pending operations.
        self.flush_pending();
        Ok(())
    }

    fn flush_pending(&mut self) {
        let mut made_progress = true;
        while made_progress {
            made_progress = false;
            let pending = std::mem::take(&mut self.pending);
            for op in pending {
                let can_apply = match &op {
                    CrdtOp::Insert { parent, .. } => {
                        parent.is_root() || self.index.contains_key(parent)
                    }
                    CrdtOp::Delete { id } => self.index.contains_key(id),
                };
                if can_apply {
                    let _ = self.apply_local(&op);
                    made_progress = true;
                } else {
                    self.pending.push(op);
                }
            }
        }
    }

    /// Integrate an insert into the node list.
    fn integrate_insert(
        &mut self,
        id: CharId,
        parent: CharId,
        value: char,
    ) -> Result<(), CrdtError> {
        if self.index.contains_key(&id) {
            // Idempotent: duplicate received.
            return Ok(());
        }

        let parent_pos = self
            .index
            .get(&parent)
            .copied()
            .ok_or(CrdtError::NotFound(parent))?;

        // Find the insertion position: right after parent, but before any
        // existing children with higher IDs (higher IDs go first for
        // deterministic concurrent resolution).
        let mut insert_pos = parent_pos + 1;
        while insert_pos < self.nodes.len() {
            let existing = &self.nodes[insert_pos];
            // If this node's parent is also our parent, we need to compare IDs.
            if existing.parent == parent {
                if existing.id < id {
                    // Existing has lower priority — insert before it.
                    break;
                }
                insert_pos += 1;
            } else if self.is_descendant_of(insert_pos, parent_pos) {
                // Skip descendants of earlier siblings.
                insert_pos += 1;
            } else {
                break;
            }
        }

        let node = RgaNode {
            id,
            value,
            deleted: false,
            parent,
        };

        self.nodes.insert(insert_pos, node);

        // Rebuild index from insert_pos onward.
        for i in insert_pos..self.nodes.len() {
            self.index.insert(self.nodes[i].id, i);
        }

        Ok(())
    }

    /// Check if the node at `pos` is a descendant of the node at `ancestor_pos`.
    fn is_descendant_of(&self, pos: usize, ancestor_pos: usize) -> bool {
        let ancestor_id = self.nodes[ancestor_pos].id;
        let mut current = pos;
        let mut depth = 0;
        while depth < self.nodes.len() {
            let parent = self.nodes[current].parent;
            if parent == ancestor_id {
                return true;
            }
            if parent.is_root() {
                return false;
            }
            match self.index.get(&parent) {
                Some(&p) => current = p,
                None => return false,
            }
            depth += 1;
        }
        false
    }

    /// Integrate a delete (tombstone).
    fn integrate_delete(&mut self, id: CharId) -> Result<(), CrdtError> {
        let pos = self
            .index
            .get(&id)
            .copied()
            .ok_or(CrdtError::NotFound(id))?;
        self.nodes[pos].deleted = true;
        Ok(())
    }

    /// Merge another document's state into this one.
    pub fn merge(&mut self, other: &RgaDocument) -> Result<(), CrdtError> {
        // Replay all non-root nodes from `other` that we don't have.
        for node in &other.nodes {
            if node.id.is_root() {
                continue;
            }
            if !self.index.contains_key(&node.id) {
                self.integrate_insert(node.id, node.parent, node.value)?;
            }
            if node.deleted {
                if let Some(&pos) = self.index.get(&node.id) {
                    self.nodes[pos].deleted = true;
                }
            }
        }
        // Update clocks.
        for (site, seq) in &other.clocks {
            let entry = self.clocks.entry(*site).or_insert(0);
            if *seq > *entry {
                *entry = *seq;
            }
        }
        Ok(())
    }

    /// Get the causal clock for this document (vector clock).
    pub fn clock(&self) -> &HashMap<SiteId, u64> {
        &self.clocks
    }

    /// Insert a string at a position (convenience).
    pub fn insert_string_at(
        &mut self,
        pos: usize,
        s: &str,
    ) -> Result<Vec<CrdtOp>, CrdtError> {
        let mut ops = Vec::new();
        for (i, ch) in s.chars().enumerate() {
            let op = self.insert_at(pos + i, ch)?;
            ops.push(op);
        }
        Ok(ops)
    }
}

impl fmt::Display for RgaDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text())
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_document_is_empty() {
        let doc = RgaDocument::new(1);
        assert!(doc.is_empty());
        assert_eq!(doc.len(), 0);
        assert_eq!(doc.text(), "");
    }

    #[test]
    fn insert_single_char() {
        let mut doc = RgaDocument::new(1);
        doc.insert_at(0, 'a').unwrap();
        assert_eq!(doc.text(), "a");
        assert_eq!(doc.len(), 1);
    }

    #[test]
    fn insert_multiple_chars() {
        let mut doc = RgaDocument::new(1);
        doc.insert_at(0, 'h').unwrap();
        doc.insert_at(1, 'i').unwrap();
        assert_eq!(doc.text(), "hi");
    }

    #[test]
    fn insert_at_beginning() {
        let mut doc = RgaDocument::new(1);
        doc.insert_at(0, 'b').unwrap();
        doc.insert_at(0, 'a').unwrap();
        assert_eq!(doc.text(), "ab");
    }

    #[test]
    fn delete_char() {
        let mut doc = RgaDocument::new(1);
        doc.insert_at(0, 'a').unwrap();
        doc.insert_at(1, 'b').unwrap();
        doc.insert_at(2, 'c').unwrap();
        doc.delete_at(1).unwrap();
        assert_eq!(doc.text(), "ac");
    }

    #[test]
    fn delete_all() {
        let mut doc = RgaDocument::new(1);
        doc.insert_at(0, 'x').unwrap();
        doc.delete_at(0).unwrap();
        assert!(doc.is_empty());
    }

    #[test]
    fn concurrent_insert_same_position() {
        let mut doc1 = RgaDocument::new(1);
        let mut doc2 = RgaDocument::new(2);

        // Both start with "ac".
        let ops: Vec<CrdtOp> = doc1.insert_string_at(0, "ac").unwrap();
        for op in &ops {
            doc2.apply_remote(op).unwrap();
        }
        assert_eq!(doc1.text(), "ac");
        assert_eq!(doc2.text(), "ac");

        // doc1 inserts 'b' after 'a' (position 1).
        let op1 = doc1.insert_at(1, 'X').unwrap();
        // doc2 inserts 'Y' after 'a' (position 1).
        let op2 = doc2.insert_at(1, 'Y').unwrap();

        // Apply remote ops.
        doc1.apply_remote(&op2).unwrap();
        doc2.apply_remote(&op1).unwrap();

        // Both documents should converge to the same text.
        assert_eq!(doc1.text(), doc2.text());
    }

    #[test]
    fn concurrent_insert_different_positions() {
        let mut doc1 = RgaDocument::new(1);
        let mut doc2 = RgaDocument::new(2);

        let ops = doc1.insert_string_at(0, "abc").unwrap();
        for op in &ops {
            doc2.apply_remote(op).unwrap();
        }

        let op1 = doc1.insert_at(1, 'X').unwrap(); // after 'a'
        let op2 = doc2.insert_at(2, 'Y').unwrap(); // after 'b'

        doc1.apply_remote(&op2).unwrap();
        doc2.apply_remote(&op1).unwrap();

        assert_eq!(doc1.text(), doc2.text());
    }

    #[test]
    fn delete_tombstone_preserves_order() {
        let mut doc = RgaDocument::new(1);
        doc.insert_string_at(0, "abcde").unwrap();
        doc.delete_at(2).unwrap(); // delete 'c'
        assert_eq!(doc.text(), "abde");
        // Total nodes should include the tombstone.
        assert!(doc.total_nodes() > doc.len());
    }

    #[test]
    fn merge_two_documents() {
        let mut doc1 = RgaDocument::new(1);
        let mut doc2 = RgaDocument::new(2);

        doc1.insert_string_at(0, "hello").unwrap();
        doc2.insert_string_at(0, "world").unwrap();

        doc1.merge(&doc2).unwrap();
        doc2.merge(&doc1).unwrap();

        assert_eq!(doc1.text(), doc2.text());
    }

    #[test]
    fn idempotent_insert() {
        let mut doc1 = RgaDocument::new(1);
        let op = doc1.insert_at(0, 'a').unwrap();
        // Apply the same op again — should be idempotent.
        doc1.apply_remote(&op).unwrap();
        assert_eq!(doc1.text(), "a");
    }

    #[test]
    fn char_id_ordering() {
        let a = CharId { site: 1, seq: 5 };
        let b = CharId { site: 2, seq: 5 };
        let c = CharId { site: 1, seq: 6 };
        assert!(a < b); // same seq, higher site wins
        assert!(a < c); // higher seq wins
    }

    #[test]
    fn char_id_display() {
        let id = CharId { site: 3, seq: 7 };
        assert_eq!(format!("{}", id), "(3,7)");
    }

    #[test]
    fn document_display() {
        let mut doc = RgaDocument::new(1);
        doc.insert_string_at(0, "test").unwrap();
        assert_eq!(format!("{}", doc), "test");
    }

    #[test]
    fn error_display() {
        let e = CrdtError::NotFound(CharId { site: 1, seq: 1 });
        assert!(e.to_string().contains("not found"));
    }

    #[test]
    fn clock_updates() {
        let mut doc = RgaDocument::new(1);
        doc.insert_at(0, 'a').unwrap();
        doc.insert_at(1, 'b').unwrap();
        assert_eq!(*doc.clock().get(&1).unwrap(), 2);
    }

    #[test]
    fn insert_string_at_convenience() {
        let mut doc = RgaDocument::new(1);
        let ops = doc.insert_string_at(0, "hello").unwrap();
        assert_eq!(ops.len(), 5);
        assert_eq!(doc.text(), "hello");
    }

    #[test]
    fn delete_out_of_bounds() {
        let mut doc = RgaDocument::new(1);
        doc.insert_at(0, 'a').unwrap();
        let result = doc.delete_at(5);
        assert!(result.is_err());
    }

    #[test]
    fn root_id_properties() {
        let root = CharId::root();
        assert!(root.is_root());
        assert_eq!(root.site, 0);
        assert_eq!(root.seq, 0);
    }
}
