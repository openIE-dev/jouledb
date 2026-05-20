//! Collaborative editing — operational transform (OT) basics with insert,
//! delete, and retain operations, transform/compose functions, and a
//! client-server synchronization model.
//!
//! Pure-Rust OT engine with no I/O. Callers wire it to their own transport.

use std::fmt;

// ── Operations ─────────────────────────────────────────────────────

/// A single component of an OT operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpComponent {
    /// Retain `n` characters unchanged.
    Retain(usize),
    /// Insert a string at the current position.
    Insert(String),
    /// Delete `n` characters at the current position.
    Delete(usize),
}

/// A compound operation consisting of a sequence of components.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Operation {
    pub components: Vec<OpComponent>,
    /// Length of the document this operation applies to (base length).
    pub base_len: usize,
    /// Length of the document after this operation is applied (target length).
    pub target_len: usize,
}

impl Operation {
    /// Create a new empty operation for a document of the given length.
    pub fn new(base_len: usize) -> Self {
        Self {
            components: Vec::new(),
            base_len,
            target_len: base_len,
        }
    }

    /// Add a retain component.
    pub fn retain(mut self, n: usize) -> Self {
        if n == 0 {
            return self;
        }
        // Merge with previous retain if possible
        if let Some(OpComponent::Retain(prev)) = self.components.last_mut() {
            *prev += n;
        } else {
            self.components.push(OpComponent::Retain(n));
        }
        self
    }

    /// Add an insert component.
    pub fn insert(mut self, s: impl Into<String>) -> Self {
        let text: String = s.into();
        if text.is_empty() {
            return self;
        }
        let added = text.len();
        // Merge with previous insert if possible
        if let Some(OpComponent::Insert(prev)) = self.components.last_mut() {
            prev.push_str(&text);
        } else {
            self.components.push(OpComponent::Insert(text));
        }
        self.target_len += added;
        self
    }

    /// Add a delete component.
    pub fn delete(mut self, n: usize) -> Self {
        if n == 0 {
            return self;
        }
        // Merge with previous delete if possible
        if let Some(OpComponent::Delete(prev)) = self.components.last_mut() {
            *prev += n;
        } else {
            self.components.push(OpComponent::Delete(n));
        }
        self.target_len -= n;
        self
    }

    /// Check that the operation's component lengths are consistent.
    pub fn is_valid(&self) -> bool {
        let mut consumed = 0usize;
        let mut produced = 0usize;
        for c in &self.components {
            match c {
                OpComponent::Retain(n) => {
                    consumed += n;
                    produced += n;
                }
                OpComponent::Insert(s) => {
                    produced += s.len();
                }
                OpComponent::Delete(n) => {
                    consumed += n;
                }
            }
        }
        consumed == self.base_len && produced == self.target_len
    }

    /// Apply this operation to a document string.
    pub fn apply(&self, doc: &str) -> Result<String, OtError> {
        if doc.len() != self.base_len {
            return Err(OtError::LengthMismatch {
                expected: self.base_len,
                actual: doc.len(),
            });
        }
        let mut result = String::with_capacity(self.target_len);
        let mut cursor = 0;
        for comp in &self.components {
            match comp {
                OpComponent::Retain(n) => {
                    if cursor + n > doc.len() {
                        return Err(OtError::RetainPastEnd);
                    }
                    result.push_str(&doc[cursor..cursor + n]);
                    cursor += n;
                }
                OpComponent::Insert(s) => {
                    result.push_str(s);
                }
                OpComponent::Delete(n) => {
                    if cursor + n > doc.len() {
                        return Err(OtError::DeletePastEnd);
                    }
                    cursor += n;
                }
            }
        }
        // Consume any remaining characters
        if cursor < doc.len() {
            result.push_str(&doc[cursor..]);
        }
        Ok(result)
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Op[")?;
        for (i, c) in self.components.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
            match c {
                OpComponent::Retain(n) => write!(f, "retain({})", n)?,
                OpComponent::Insert(s) => write!(f, "insert({:?})", s)?,
                OpComponent::Delete(n) => write!(f, "delete({})", n)?,
            }
        }
        write!(f, "]")
    }
}

// ── OT errors ──────────────────────────────────────────────────────

/// Errors that can occur during OT operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OtError {
    LengthMismatch { expected: usize, actual: usize },
    RetainPastEnd,
    DeletePastEnd,
    IncompatibleOperations,
    InvalidState(String),
}

impl fmt::Display for OtError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } =>
                write!(f, "length mismatch: expected {}, got {}", expected, actual),
            Self::RetainPastEnd => write!(f, "retain past end of document"),
            Self::DeletePastEnd => write!(f, "delete past end of document"),
            Self::IncompatibleOperations => write!(f, "operations have incompatible base lengths"),
            Self::InvalidState(msg) => write!(f, "invalid state: {}", msg),
        }
    }
}

// ── Compose ────────────────────────────────────────────────────────

/// Compose two operations into a single operation: apply(a) then apply(b) = apply(compose(a, b)).
pub fn compose(a: &Operation, b: &Operation) -> Result<Operation, OtError> {
    if a.target_len != b.base_len {
        return Err(OtError::IncompatibleOperations);
    }

    let mut result = Operation::new(a.base_len);
    result.target_len = a.base_len; // will be adjusted by builders

    let mut ai = ComponentIter::new(&a.components);
    let mut bi = ComponentIter::new(&b.components);

    // We need to manually build the result to track lengths properly
    let mut components = Vec::new();
    let mut result_target_len = 0usize;

    loop {
        let a_comp = ai.peek();
        let b_comp = bi.peek();

        if a_comp.is_none() && b_comp.is_none() {
            break;
        }

        // If b is an insert, just add it
        if let Some(OpComponent::Insert(s)) = b_comp {
            let s = s.clone();
            result_target_len += s.len();
            components.push(OpComponent::Insert(s));
            bi.advance();
            continue;
        }

        // If a is a delete, just add it
        if let Some(OpComponent::Delete(n)) = a_comp {
            let n = *n;
            components.push(OpComponent::Delete(n));
            ai.advance();
            continue;
        }

        match (a_comp, b_comp) {
            (Some(OpComponent::Retain(an)), Some(OpComponent::Retain(bn))) => {
                let an = *an;
                let bn = *bn;
                let min = an.min(bn);
                result_target_len += min;
                components.push(OpComponent::Retain(min));
                if an > bn {
                    ai.consume(bn);
                    bi.advance();
                } else if an < bn {
                    ai.advance();
                    bi.consume(an);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            (Some(OpComponent::Insert(s)), Some(OpComponent::Retain(bn))) => {
                let s = s.clone();
                let bn = *bn;
                let slen = s.len();
                let min = slen.min(bn);
                result_target_len += min;
                components.push(OpComponent::Insert(s[..min].to_string()));
                if slen > bn {
                    ai.consume_insert(bn);
                    bi.advance();
                } else if slen < bn {
                    ai.advance();
                    bi.consume(slen);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            (Some(OpComponent::Insert(s)), Some(OpComponent::Delete(bn))) => {
                let s = s.clone();
                let bn = *bn;
                let slen = s.len();
                let min = slen.min(bn);
                // Insert then delete cancels out
                if slen > bn {
                    ai.consume_insert(bn);
                    bi.advance();
                } else if slen < bn {
                    ai.advance();
                    bi.consume(slen);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            (Some(OpComponent::Retain(an)), Some(OpComponent::Delete(bn))) => {
                let an = *an;
                let bn = *bn;
                let min = an.min(bn);
                components.push(OpComponent::Delete(min));
                if an > bn {
                    ai.consume(bn);
                    bi.advance();
                } else if an < bn {
                    ai.advance();
                    bi.consume(an);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            _ => break,
        }
    }

    let mut op = Operation::new(a.base_len);
    op.components = components;
    op.target_len = result_target_len;
    Ok(op)
}

// ── Transform ──────────────────────────────────────────────────────

/// Transform two concurrent operations so they can both be applied.
/// Returns (a', b') such that apply(a) then apply(b') = apply(b) then apply(a').
pub fn transform(a: &Operation, b: &Operation) -> Result<(Operation, Operation), OtError> {
    if a.base_len != b.base_len {
        return Err(OtError::IncompatibleOperations);
    }

    let mut a_prime_comps: Vec<OpComponent> = Vec::new();
    let mut b_prime_comps: Vec<OpComponent> = Vec::new();

    let mut ai = ComponentIter::new(&a.components);
    let mut bi = ComponentIter::new(&b.components);

    loop {
        let ac = ai.peek();
        let bc = bi.peek();

        if ac.is_none() && bc.is_none() {
            break;
        }

        // a inserts first (tie-break: a wins)
        if let Some(OpComponent::Insert(s)) = ac {
            let s = s.clone();
            let len = s.len();
            a_prime_comps.push(OpComponent::Insert(s));
            b_prime_comps.push(OpComponent::Retain(len));
            ai.advance();
            continue;
        }

        // b inserts
        if let Some(OpComponent::Insert(s)) = bc {
            let s = s.clone();
            let len = s.len();
            b_prime_comps.push(OpComponent::Insert(s));
            a_prime_comps.push(OpComponent::Retain(len));
            bi.advance();
            continue;
        }

        match (ac, bc) {
            (Some(OpComponent::Retain(an)), Some(OpComponent::Retain(bn))) => {
                let an = *an;
                let bn = *bn;
                let min = an.min(bn);
                a_prime_comps.push(OpComponent::Retain(min));
                b_prime_comps.push(OpComponent::Retain(min));
                if an > bn {
                    ai.consume(bn);
                    bi.advance();
                } else if an < bn {
                    ai.advance();
                    bi.consume(an);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            (Some(OpComponent::Delete(an)), Some(OpComponent::Delete(bn))) => {
                let an = *an;
                let bn = *bn;
                let min = an.min(bn);
                // Both delete the same range — cancel
                if an > bn {
                    ai.consume(bn);
                    bi.advance();
                } else if an < bn {
                    ai.advance();
                    bi.consume(an);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            (Some(OpComponent::Delete(an)), Some(OpComponent::Retain(bn))) => {
                let an = *an;
                let bn = *bn;
                let min = an.min(bn);
                a_prime_comps.push(OpComponent::Delete(min));
                if an > bn {
                    ai.consume(bn);
                    bi.advance();
                } else if an < bn {
                    ai.advance();
                    bi.consume(an);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            (Some(OpComponent::Retain(an)), Some(OpComponent::Delete(bn))) => {
                let an = *an;
                let bn = *bn;
                let min = an.min(bn);
                b_prime_comps.push(OpComponent::Delete(min));
                if an > bn {
                    ai.consume(bn);
                    bi.advance();
                } else if an < bn {
                    ai.advance();
                    bi.consume(an);
                } else {
                    ai.advance();
                    bi.advance();
                }
            }
            _ => break,
        }
    }

    let a_prime = build_op_from_comps(a_prime_comps, b.target_len);
    let b_prime = build_op_from_comps(b_prime_comps, a.target_len);

    Ok((a_prime, b_prime))
}

fn build_op_from_comps(comps: Vec<OpComponent>, base_len: usize) -> Operation {
    let mut target_len = base_len;
    // Recalculate: base_len is the length after the *other* op was applied
    // We need to figure out what this op does to it
    let mut consumed = 0usize;
    let mut produced = 0usize;
    for c in &comps {
        match c {
            OpComponent::Retain(n) => { consumed += n; produced += n; }
            OpComponent::Insert(s) => { produced += s.len(); }
            OpComponent::Delete(n) => { consumed += n; }
        }
    }
    Operation {
        components: comps,
        base_len: consumed,
        target_len: produced,
    }
}

// ── Component iterator helper ──────────────────────────────────────

struct ComponentIter {
    components: Vec<OpComponent>,
    index: usize,
    offset: usize, // partial consumption within current component
}

impl ComponentIter {
    fn new(components: &[OpComponent]) -> Self {
        Self { components: components.to_vec(), index: 0, offset: 0 }
    }

    fn peek(&self) -> Option<&OpComponent> {
        if self.index >= self.components.len() {
            return None;
        }
        Some(&self.components[self.index])
    }

    fn advance(&mut self) {
        self.index += 1;
        self.offset = 0;
    }

    /// Consume `n` units from a Retain or Delete, leaving the remainder.
    fn consume(&mut self, n: usize) {
        match &self.components[self.index] {
            OpComponent::Retain(total) => {
                let remaining = total - self.offset;
                if n >= remaining {
                    self.advance();
                } else {
                    self.offset += n;
                    let new_val = *total - self.offset;
                    self.components[self.index] = OpComponent::Retain(new_val);
                    self.offset = 0;
                }
            }
            OpComponent::Delete(total) => {
                let remaining = total - self.offset;
                if n >= remaining {
                    self.advance();
                } else {
                    self.offset += n;
                    let new_val = *total - self.offset;
                    self.components[self.index] = OpComponent::Delete(new_val);
                    self.offset = 0;
                }
            }
            OpComponent::Insert(_) => {
                self.consume_insert(n);
            }
        }
    }

    fn consume_insert(&mut self, n: usize) {
        if let OpComponent::Insert(s) = &self.components[self.index] {
            let remaining = s.len() - self.offset;
            if n >= remaining {
                self.advance();
            } else {
                self.offset += n;
                let new_s = s[self.offset..].to_string();
                self.components[self.index] = OpComponent::Insert(new_s);
                self.offset = 0;
            }
        }
    }
}

// ── Client-server sync model ───────────────────────────────────────

/// Client synchronization state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClientState {
    /// No pending operations — in sync with server.
    Synchronized,
    /// One operation sent to server, awaiting acknowledgment.
    AwaitingConfirm,
    /// Awaiting confirm with additional buffered operations.
    AwaitingWithBuffer,
}

/// A collaborative editing client that tracks sync state.
#[derive(Debug)]
pub struct OtClient {
    pub state: ClientState,
    pub revision: u64,
    pub document: String,
    pending: Option<Operation>,
    buffer: Option<Operation>,
}

impl OtClient {
    pub fn new(document: impl Into<String>, revision: u64) -> Self {
        Self {
            state: ClientState::Synchronized,
            revision,
            document: document.into(),
            pending: None,
            buffer: None,
        }
    }

    /// Apply a local operation. Returns the operation to send to server (if any).
    pub fn apply_local(&mut self, op: Operation) -> Result<Option<Operation>, OtError> {
        self.document = op.apply(&self.document)?;
        match self.state {
            ClientState::Synchronized => {
                self.pending = Some(op.clone());
                self.state = ClientState::AwaitingConfirm;
                Ok(Some(op))
            }
            ClientState::AwaitingConfirm => {
                self.buffer = Some(op);
                self.state = ClientState::AwaitingWithBuffer;
                Ok(None)
            }
            ClientState::AwaitingWithBuffer => {
                let buf = self.buffer.take().unwrap();
                self.buffer = Some(compose(&buf, &op)?);
                Ok(None)
            }
        }
    }

    /// Server acknowledged our pending operation.
    pub fn server_ack(&mut self) -> Result<Option<Operation>, OtError> {
        self.revision += 1;
        match self.state {
            ClientState::AwaitingConfirm => {
                self.pending = None;
                self.state = ClientState::Synchronized;
                Ok(None)
            }
            ClientState::AwaitingWithBuffer => {
                let buf = self.buffer.take().unwrap();
                self.pending = Some(buf.clone());
                self.state = ClientState::AwaitingConfirm;
                Ok(Some(buf))
            }
            ClientState::Synchronized => {
                Err(OtError::InvalidState("ack in synchronized state".into()))
            }
        }
    }

    /// Apply a server-originated operation from another client.
    pub fn apply_server(&mut self, server_op: Operation) -> Result<(), OtError> {
        match self.state {
            ClientState::Synchronized => {
                self.document = server_op.apply(&self.document)?;
                self.revision += 1;
            }
            ClientState::AwaitingConfirm => {
                let pending = self.pending.take().unwrap();
                let (pending_prime, server_prime) = transform(&pending, &server_op)?;
                self.pending = Some(pending_prime);
                self.document = server_prime.apply(&self.document)?;
                self.revision += 1;
            }
            ClientState::AwaitingWithBuffer => {
                let pending = self.pending.take().unwrap();
                let (pending_prime, server_prime_1) = transform(&pending, &server_op)?;
                let buffer = self.buffer.take().unwrap();
                let (buffer_prime, server_prime_2) = transform(&buffer, &server_prime_1)?;
                self.pending = Some(pending_prime);
                self.buffer = Some(buffer_prime);
                self.document = server_prime_2.apply(&self.document)?;
                self.revision += 1;
            }
        }
        Ok(())
    }
}

/// A simple OT server that maintains the document and revision history.
#[derive(Debug)]
pub struct OtServer {
    pub document: String,
    pub revision: u64,
    history: Vec<Operation>,
}

impl OtServer {
    pub fn new(document: impl Into<String>) -> Self {
        Self {
            document: document.into(),
            revision: 0,
            history: Vec::new(),
        }
    }

    /// Receive an operation from a client at a given revision.
    /// Returns the transformed operation to broadcast to other clients.
    pub fn receive(&mut self, client_rev: u64, mut op: Operation) -> Result<Operation, OtError> {
        if client_rev > self.revision {
            return Err(OtError::InvalidState("client revision ahead of server".into()));
        }

        // Transform against all operations since client's revision
        let start = client_rev as usize;
        for server_op in &self.history[start..] {
            let (_, op_prime) = transform(server_op, &op)?;
            op = op_prime;
        }

        self.document = op.apply(&self.document)?;
        self.history.push(op.clone());
        self.revision += 1;
        Ok(op)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Operation building ─────────────────────────────────────────

    #[test]
    fn operation_retain_merge() {
        let op = Operation::new(10).retain(3).retain(4);
        assert_eq!(op.components.len(), 1);
        assert_eq!(op.components[0], OpComponent::Retain(7));
    }

    #[test]
    fn operation_insert_merge() {
        let op = Operation::new(0).insert("hello").insert(" world");
        assert_eq!(op.components.len(), 1);
        assert_eq!(op.components[0], OpComponent::Insert("hello world".into()));
    }

    #[test]
    fn operation_delete_merge() {
        let op = Operation::new(10).delete(3).delete(2);
        assert_eq!(op.components.len(), 1);
        assert_eq!(op.components[0], OpComponent::Delete(5));
        assert_eq!(op.target_len, 5);
    }

    #[test]
    fn operation_zero_length_ignored() {
        let op = Operation::new(5).retain(0).insert("").delete(0);
        assert!(op.components.is_empty());
    }

    #[test]
    fn operation_display() {
        let op = Operation::new(5).retain(2).insert("x").delete(1);
        let s = op.to_string();
        assert!(s.contains("retain(2)"));
        assert!(s.contains("insert(\"x\")"));
        assert!(s.contains("delete(1)"));
    }

    // ── Apply ──────────────────────────────────────────────────────

    #[test]
    fn apply_insert_at_start() {
        let op = Operation::new(5).insert("hi ").retain(5);
        let result = op.apply("world").unwrap();
        assert_eq!(result, "hi world");
    }

    #[test]
    fn apply_delete_at_start() {
        let op = Operation::new(5).delete(2).retain(3);
        let result = op.apply("hello").unwrap();
        assert_eq!(result, "llo");
    }

    #[test]
    fn apply_insert_in_middle() {
        let op = Operation::new(5).retain(2).insert("XY").retain(3);
        let result = op.apply("abcde").unwrap();
        assert_eq!(result, "abXYcde");
    }

    #[test]
    fn apply_replace() {
        let op = Operation::new(5).retain(1).delete(3).insert("!!").retain(1);
        let result = op.apply("abcde").unwrap();
        assert_eq!(result, "a!!e");
    }

    #[test]
    fn apply_length_mismatch() {
        let op = Operation::new(5).retain(5);
        let err = op.apply("hi").unwrap_err();
        assert!(matches!(err, OtError::LengthMismatch { expected: 5, actual: 2 }));
    }

    #[test]
    fn apply_retain_past_end() {
        let op = Operation::new(3);
        let mut bad_op = op;
        bad_op.components.push(OpComponent::Retain(10));
        let err = bad_op.apply("abc").unwrap_err();
        assert!(matches!(err, OtError::RetainPastEnd));
    }

    #[test]
    fn apply_delete_past_end() {
        let op = Operation::new(3);
        let mut bad_op = op;
        bad_op.components.push(OpComponent::Delete(10));
        let err = bad_op.apply("abc").unwrap_err();
        assert!(matches!(err, OtError::DeletePastEnd));
    }

    // ── Validity ───────────────────────────────────────────────────

    #[test]
    fn operation_valid() {
        let op = Operation::new(5).retain(2).insert("x").delete(1).retain(2);
        assert!(op.is_valid());
    }

    #[test]
    fn operation_invalid() {
        let mut op = Operation::new(5);
        op.components.push(OpComponent::Retain(99));
        assert!(!op.is_valid());
    }

    // ── Compose ────────────────────────────────────────────────────

    #[test]
    fn compose_two_inserts() {
        let doc = "abc";
        let a = Operation::new(3).retain(1).insert("X").retain(2);
        let b = Operation::new(4).retain(2).insert("Y").retain(2);
        let composed = compose(&a, &b).unwrap();
        let direct = composed.apply(doc).unwrap();
        let step = a.apply(doc).unwrap();
        let sequential = b.apply(&step).unwrap();
        assert_eq!(direct, sequential);
    }

    #[test]
    fn compose_incompatible() {
        let a = Operation::new(5).retain(5);
        let b = Operation::new(3).retain(3); // base_len doesn't match a.target_len
        assert!(compose(&a, &b).is_err());
    }

    // ── Transform ──────────────────────────────────────────────────

    #[test]
    fn transform_two_inserts_at_same_position() {
        let doc = "abc";
        let a = Operation::new(3).insert("X").retain(3);
        let b = Operation::new(3).insert("Y").retain(3);
        let (a_prime, b_prime) = transform(&a, &b).unwrap();

        let after_a = a.apply(doc).unwrap();
        let after_a_then_b = b_prime.apply(&after_a).unwrap();

        let after_b = b.apply(doc).unwrap();
        let after_b_then_a = a_prime.apply(&after_b).unwrap();

        assert_eq!(after_a_then_b, after_b_then_a);
    }

    #[test]
    fn transform_insert_vs_delete() {
        let doc = "abcde";
        let a = Operation::new(5).retain(2).insert("X").retain(3);
        let b = Operation::new(5).retain(1).delete(2).retain(2);
        let (a_prime, b_prime) = transform(&a, &b).unwrap();

        let after_a = a.apply(doc).unwrap();
        let after_a_then_b = b_prime.apply(&after_a).unwrap();

        let after_b = b.apply(doc).unwrap();
        let after_b_then_a = a_prime.apply(&after_b).unwrap();

        assert_eq!(after_a_then_b, after_b_then_a);
    }

    #[test]
    fn transform_incompatible_base() {
        let a = Operation::new(3).retain(3);
        let b = Operation::new(5).retain(5);
        assert!(transform(&a, &b).is_err());
    }

    // ── OT client ──────────────────────────────────────────────────

    #[test]
    fn client_synchronized_send() {
        let mut client = OtClient::new("hello", 0);
        let op = Operation::new(5).retain(5).insert("!");
        let to_send = client.apply_local(op).unwrap();
        assert!(to_send.is_some());
        assert_eq!(client.state, ClientState::AwaitingConfirm);
        assert_eq!(client.document, "hello!");
    }

    #[test]
    fn client_buffer_while_awaiting() {
        let mut client = OtClient::new("hello", 0);
        let op1 = Operation::new(5).retain(5).insert("!");
        client.apply_local(op1).unwrap();
        assert_eq!(client.state, ClientState::AwaitingConfirm);

        let op2 = Operation::new(6).retain(6).insert("?");
        let to_send = client.apply_local(op2).unwrap();
        assert!(to_send.is_none()); // buffered
        assert_eq!(client.state, ClientState::AwaitingWithBuffer);
        assert_eq!(client.document, "hello!?");
    }

    #[test]
    fn client_ack_synchronized() {
        let mut client = OtClient::new("hello", 0);
        let op = Operation::new(5).retain(5).insert("!");
        client.apply_local(op).unwrap();
        let flush = client.server_ack().unwrap();
        assert!(flush.is_none());
        assert_eq!(client.state, ClientState::Synchronized);
        assert_eq!(client.revision, 1);
    }

    #[test]
    fn client_ack_flushes_buffer() {
        let mut client = OtClient::new("ab", 0);
        let op1 = Operation::new(2).retain(2).insert("c");
        client.apply_local(op1).unwrap();
        let op2 = Operation::new(3).retain(3).insert("d");
        client.apply_local(op2).unwrap();
        assert_eq!(client.state, ClientState::AwaitingWithBuffer);

        let flush = client.server_ack().unwrap();
        assert!(flush.is_some()); // buffer sent
        assert_eq!(client.state, ClientState::AwaitingConfirm);
    }

    #[test]
    fn client_ack_in_synchronized_is_error() {
        let mut client = OtClient::new("ab", 0);
        assert!(client.server_ack().is_err());
    }

    #[test]
    fn client_apply_server_in_synchronized() {
        let mut client = OtClient::new("ab", 0);
        let server_op = Operation::new(2).retain(2).insert("c");
        client.apply_server(server_op).unwrap();
        assert_eq!(client.document, "abc");
        assert_eq!(client.revision, 1);
    }

    // ── OT server ──────────────────────────────────────────────────

    #[test]
    fn server_receive_at_current_revision() {
        let mut server = OtServer::new("abc");
        let op = Operation::new(3).retain(3).insert("d");
        let broadcast = server.receive(0, op).unwrap();
        assert_eq!(server.document, "abcd");
        assert_eq!(server.revision, 1);
        assert!(!broadcast.components.is_empty());
    }

    #[test]
    fn server_receive_stale_revision() {
        let mut server = OtServer::new("abc");
        // First op
        let op1 = Operation::new(3).retain(3).insert("X");
        server.receive(0, op1).unwrap();
        // Second op at revision 0 (stale)
        let op2 = Operation::new(3).retain(3).insert("Y");
        let broadcast = server.receive(0, op2).unwrap();
        // Should transform against op1 and still apply
        assert_eq!(server.revision, 2);
        assert!(server.document.contains('X'));
        assert!(server.document.contains('Y'));
        assert!(!broadcast.components.is_empty());
    }

    #[test]
    fn server_reject_future_revision() {
        let mut server = OtServer::new("abc");
        let op = Operation::new(3).retain(3).insert("d");
        assert!(server.receive(99, op).is_err());
    }

    // ── Error display ──────────────────────────────────────────────

    #[test]
    fn error_display() {
        assert!(OtError::RetainPastEnd.to_string().contains("retain past end"));
        assert!(OtError::DeletePastEnd.to_string().contains("delete past end"));
        assert!(OtError::IncompatibleOperations.to_string().contains("incompatible"));
        let lm = OtError::LengthMismatch { expected: 5, actual: 3 };
        assert!(lm.to_string().contains("5"));
    }
}
