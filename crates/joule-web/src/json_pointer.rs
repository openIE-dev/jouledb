// SPDX-License-Identifier: MIT
//! JSON Pointer — RFC 6901 implementation.
//!
//! Parsing, value resolution, escape sequences (`~0` = `~`, `~1` = `/`),
//! relative pointers (RFC draft), and pointer construction from path segments.

use serde_json::Value;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum PointerError {
    InvalidSyntax(String),
    NotFound(String),
    IndexOutOfBounds { index: usize, len: usize },
    InvalidIndex(String),
}

impl fmt::Display for PointerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSyntax(msg) => write!(f, "invalid pointer syntax: {msg}"),
            Self::NotFound(seg) => write!(f, "segment not found: {seg}"),
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds (len {len})")
            }
            Self::InvalidIndex(seg) => write!(f, "invalid array index: {seg}"),
        }
    }
}

// ── JsonPointer ─────────────────────────────────────────────────────────────

/// A parsed JSON Pointer (RFC 6901).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct JsonPointer {
    segments: Vec<String>,
}

impl JsonPointer {
    /// Parse a JSON Pointer string. Must be empty or start with `/`.
    pub fn parse(input: &str) -> Result<Self, PointerError> {
        if input.is_empty() {
            return Ok(Self { segments: vec![] });
        }
        if !input.starts_with('/') {
            return Err(PointerError::InvalidSyntax(
                "must start with '/' or be empty".into(),
            ));
        }
        // Validate: no incomplete escape sequences
        let bytes = input.as_bytes();
        for i in 0..bytes.len() {
            if bytes[i] == b'~' {
                if i + 1 >= bytes.len() || (bytes[i + 1] != b'0' && bytes[i + 1] != b'1') {
                    return Err(PointerError::InvalidSyntax(
                        format!("invalid escape at position {i}"),
                    ));
                }
            }
        }
        let segments = input[1..]
            .split('/')
            .map(|seg| unescape(seg))
            .collect();
        Ok(Self { segments })
    }

    /// Build a pointer from raw (unescaped) path segments.
    pub fn from_segments(segments: Vec<String>) -> Self {
        Self { segments }
    }

    /// Build a pointer from an iterator of segments.
    pub fn from_parts<I, S>(parts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            segments: parts.into_iter().map(Into::into).collect(),
        }
    }

    /// The root pointer (empty string).
    pub fn root() -> Self {
        Self { segments: vec![] }
    }

    /// Number of segments.
    pub fn len(&self) -> usize {
        self.segments.len()
    }

    /// Whether this is the root pointer.
    pub fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    /// Access individual segments.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Append a segment, returning a new pointer.
    pub fn push(&self, segment: &str) -> Self {
        let mut segs = self.segments.clone();
        segs.push(segment.to_string());
        Self { segments: segs }
    }

    /// Remove the last segment, returning the parent pointer and the segment.
    pub fn pop(&self) -> Option<(Self, String)> {
        if self.segments.is_empty() {
            return None;
        }
        let mut segs = self.segments.clone();
        let last = segs.pop().unwrap();
        Some((Self { segments: segs }, last))
    }

    /// Return the parent pointer (all but last segment).
    pub fn parent(&self) -> Option<Self> {
        self.pop().map(|(p, _)| p)
    }

    /// Return the last segment.
    pub fn last(&self) -> Option<&str> {
        self.segments.last().map(|s| s.as_str())
    }

    /// Serialize to the RFC 6901 string representation.
    pub fn to_string_repr(&self) -> String {
        if self.segments.is_empty() {
            return String::new();
        }
        let mut out = String::new();
        for seg in &self.segments {
            out.push('/');
            out.push_str(&escape(seg));
        }
        out
    }

    /// Resolve this pointer against a JSON value.
    pub fn resolve<'a>(&self, root: &'a Value) -> Result<&'a Value, PointerError> {
        let mut cur = root;
        for seg in &self.segments {
            cur = descend(cur, seg)?;
        }
        Ok(cur)
    }

    /// Resolve this pointer mutably.
    pub fn resolve_mut<'a>(&self, root: &'a mut Value) -> Result<&'a mut Value, PointerError> {
        let mut cur = root;
        for seg in &self.segments {
            cur = descend_mut(cur, seg)?;
        }
        Ok(cur)
    }

    /// Check whether this pointer is a prefix of another.
    pub fn is_prefix_of(&self, other: &JsonPointer) -> bool {
        if self.segments.len() > other.segments.len() {
            return false;
        }
        self.segments.iter().zip(&other.segments).all(|(a, b)| a == b)
    }

    /// Return the relative path from self to descendant.
    pub fn relative_to(&self, descendant: &JsonPointer) -> Option<JsonPointer> {
        if !self.is_prefix_of(descendant) {
            return None;
        }
        Some(JsonPointer {
            segments: descendant.segments[self.segments.len()..].to_vec(),
        })
    }

    /// Concatenate two pointers.
    pub fn join(&self, other: &JsonPointer) -> JsonPointer {
        let mut segs = self.segments.clone();
        segs.extend_from_slice(&other.segments);
        JsonPointer { segments: segs }
    }
}

impl fmt::Display for JsonPointer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

// ── Relative JSON Pointer ───────────────────────────────────────────────────

/// A Relative JSON Pointer (draft spec).
/// Format: `<non-negative-integer><json-pointer-or-#>`
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePointer {
    pub up: usize,
    pub kind: RelativeKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RelativeKind {
    /// Resolves to a value at the given sub-pointer.
    Pointer(JsonPointer),
    /// Resolves to the key/index name (the `#` form).
    Key,
}

impl RelativePointer {
    pub fn parse(input: &str) -> Result<Self, PointerError> {
        // Find where digits end
        let digit_end = input
            .find(|c: char| !c.is_ascii_digit())
            .unwrap_or(input.len());
        if digit_end == 0 {
            return Err(PointerError::InvalidSyntax(
                "relative pointer must start with a number".into(),
            ));
        }
        let up: usize = input[..digit_end]
            .parse()
            .map_err(|_| PointerError::InvalidSyntax("bad integer".into()))?;
        let rest = &input[digit_end..];
        if rest == "#" {
            Ok(Self { up, kind: RelativeKind::Key })
        } else {
            let ptr = if rest.is_empty() {
                JsonPointer::root()
            } else {
                JsonPointer::parse(rest)?
            };
            Ok(Self { up, kind: RelativeKind::Pointer(ptr) })
        }
    }

    /// Resolve a relative pointer given the current path in the document.
    pub fn resolve<'a>(
        &self,
        root: &'a Value,
        current_path: &JsonPointer,
    ) -> Result<RelativeResult<'a>, PointerError> {
        if self.up > current_path.len() {
            return Err(PointerError::InvalidSyntax(
                "cannot go above root".into(),
            ));
        }
        let base_segs = &current_path.segments()[..current_path.len() - self.up];
        match &self.kind {
            RelativeKind::Key => {
                if base_segs.len() == current_path.len() - self.up && self.up > 0 {
                    let idx = current_path.len() - self.up;
                    Ok(RelativeResult::Key(current_path.segments()[idx].clone()))
                } else {
                    Err(PointerError::InvalidSyntax("# requires up > 0".into()))
                }
            }
            RelativeKind::Pointer(sub) => {
                let base = JsonPointer::from_segments(base_segs.to_vec());
                let full = base.join(sub);
                let val = full.resolve(root)?;
                Ok(RelativeResult::Value(val))
            }
        }
    }
}

#[derive(Debug)]
pub enum RelativeResult<'a> {
    Value(&'a Value),
    Key(String),
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn escape(seg: &str) -> String {
    seg.replace('~', "~0").replace('/', "~1")
}

fn unescape(seg: &str) -> String {
    seg.replace("~1", "/").replace("~0", "~")
}

fn descend<'a>(val: &'a Value, seg: &str) -> Result<&'a Value, PointerError> {
    match val {
        Value::Object(map) => map
            .get(seg)
            .ok_or_else(|| PointerError::NotFound(seg.into())),
        Value::Array(arr) => {
            let idx: usize = seg
                .parse()
                .map_err(|_| PointerError::InvalidIndex(seg.into()))?;
            arr.get(idx).ok_or(PointerError::IndexOutOfBounds {
                index: idx,
                len: arr.len(),
            })
        }
        _ => Err(PointerError::NotFound(seg.into())),
    }
}

fn descend_mut<'a>(val: &'a mut Value, seg: &str) -> Result<&'a mut Value, PointerError> {
    match val {
        Value::Object(map) => map
            .get_mut(seg)
            .ok_or_else(|| PointerError::NotFound(seg.into())),
        Value::Array(arr) => {
            let len = arr.len();
            let idx: usize = seg
                .parse()
                .map_err(|_| PointerError::InvalidIndex(seg.into()))?;
            arr.get_mut(idx).ok_or(PointerError::IndexOutOfBounds {
                index: idx,
                len,
            })
        }
        _ => Err(PointerError::NotFound(seg.into())),
    }
}

/// Convenience: resolve a pointer string against a value.
pub fn resolve<'a>(pointer: &str, root: &'a Value) -> Result<&'a Value, PointerError> {
    JsonPointer::parse(pointer)?.resolve(root)
}

/// Convenience: resolve mutably.
pub fn resolve_mut<'a>(pointer: &str, root: &'a mut Value) -> Result<&'a mut Value, PointerError> {
    JsonPointer::parse(pointer)?.resolve_mut(root)
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_empty() {
        let p = JsonPointer::parse("").unwrap();
        assert!(p.is_empty());
        assert_eq!(p.to_string_repr(), "");
    }

    #[test]
    fn parse_segments() {
        let p = JsonPointer::parse("/a/b/c").unwrap();
        assert_eq!(p.segments(), &["a", "b", "c"]);
        assert_eq!(p.len(), 3);
    }

    #[test]
    fn parse_escape() {
        let p = JsonPointer::parse("/a~1b/c~0d").unwrap();
        assert_eq!(p.segments(), &["a/b", "c~d"]);
    }

    #[test]
    fn parse_invalid_no_slash() {
        assert!(JsonPointer::parse("abc").is_err());
    }

    #[test]
    fn parse_invalid_escape() {
        assert!(JsonPointer::parse("/a~").is_err());
        assert!(JsonPointer::parse("/a~2").is_err());
    }

    #[test]
    fn roundtrip_string() {
        let cases = vec!["", "/a", "/a/b", "/a~1b", "/~0~1"];
        for c in cases {
            let p = JsonPointer::parse(c).unwrap();
            assert_eq!(p.to_string_repr(), c);
        }
    }

    #[test]
    fn from_segments_roundtrip() {
        let p = JsonPointer::from_segments(vec!["a/b".into(), "c~d".into()]);
        assert_eq!(p.to_string_repr(), "/a~1b/c~0d");
        let parsed = JsonPointer::parse(&p.to_string_repr()).unwrap();
        assert_eq!(parsed, p);
    }

    #[test]
    fn from_parts() {
        let p = JsonPointer::from_parts(["x", "y", "z"]);
        assert_eq!(p.segments(), &["x", "y", "z"]);
    }

    #[test]
    fn resolve_object() {
        let doc = json!({"a": {"b": {"c": 42}}});
        let val = resolve("/a/b/c", &doc).unwrap();
        assert_eq!(val, &json!(42));
    }

    #[test]
    fn resolve_array() {
        let doc = json!({"items": [10, 20, 30]});
        assert_eq!(resolve("/items/1", &doc).unwrap(), &json!(20));
    }

    #[test]
    fn resolve_root() {
        let doc = json!(42);
        assert_eq!(resolve("", &doc).unwrap(), &json!(42));
    }

    #[test]
    fn resolve_not_found() {
        let doc = json!({"a": 1});
        assert!(resolve("/b", &doc).is_err());
    }

    #[test]
    fn resolve_index_oob() {
        let doc = json!([1, 2]);
        let err = resolve("/5", &doc).unwrap_err();
        assert!(matches!(err, PointerError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn resolve_invalid_index() {
        let doc = json!([1, 2]);
        let err = resolve("/abc", &doc).unwrap_err();
        assert!(matches!(err, PointerError::InvalidIndex(_)));
    }

    #[test]
    fn resolve_mut_works() {
        let mut doc = json!({"a": {"b": 1}});
        let val = resolve_mut("/a/b", &mut doc).unwrap();
        *val = json!(99);
        assert_eq!(doc, json!({"a": {"b": 99}}));
    }

    #[test]
    fn push_and_pop() {
        let p = JsonPointer::parse("/a/b").unwrap();
        let extended = p.push("c");
        assert_eq!(extended.to_string_repr(), "/a/b/c");
        let (parent, last) = extended.pop().unwrap();
        assert_eq!(parent, p);
        assert_eq!(last, "c");
    }

    #[test]
    fn pop_root_returns_none() {
        assert!(JsonPointer::root().pop().is_none());
    }

    #[test]
    fn parent_and_last() {
        let p = JsonPointer::parse("/a/b/c").unwrap();
        assert_eq!(p.last(), Some("c"));
        let parent = p.parent().unwrap();
        assert_eq!(parent.to_string_repr(), "/a/b");
    }

    #[test]
    fn is_prefix_of() {
        let a = JsonPointer::parse("/a/b").unwrap();
        let b = JsonPointer::parse("/a/b/c").unwrap();
        assert!(a.is_prefix_of(&b));
        assert!(!b.is_prefix_of(&a));
        assert!(a.is_prefix_of(&a));
    }

    #[test]
    fn relative_to() {
        let base = JsonPointer::parse("/a").unwrap();
        let child = JsonPointer::parse("/a/b/c").unwrap();
        let rel = base.relative_to(&child).unwrap();
        assert_eq!(rel.to_string_repr(), "/b/c");
    }

    #[test]
    fn relative_to_non_prefix() {
        let a = JsonPointer::parse("/a").unwrap();
        let b = JsonPointer::parse("/b").unwrap();
        assert!(a.relative_to(&b).is_none());
    }

    #[test]
    fn join_pointers() {
        let a = JsonPointer::parse("/a/b").unwrap();
        let b = JsonPointer::parse("/c/d").unwrap();
        let c = a.join(&b);
        assert_eq!(c.to_string_repr(), "/a/b/c/d");
    }

    #[test]
    fn display_trait() {
        let p = JsonPointer::parse("/a/b").unwrap();
        assert_eq!(format!("{p}"), "/a/b");
    }

    #[test]
    fn rfc_6901_examples() {
        // From the RFC
        let doc = json!({
            "foo": ["bar", "baz"],
            "": 0,
            "a/b": 1,
            "c%d": 2,
            "e^f": 3,
            "g|h": 4,
            "i\\j": 5,
            "k\"l": 6,
            " ": 7,
            "m~n": 8
        });
        assert_eq!(resolve("", &doc).unwrap(), &doc);
        assert_eq!(resolve("/foo", &doc).unwrap(), &json!(["bar", "baz"]));
        assert_eq!(resolve("/foo/0", &doc).unwrap(), &json!("bar"));
        assert_eq!(resolve("/", &doc).unwrap(), &json!(0));
        assert_eq!(resolve("/a~1b", &doc).unwrap(), &json!(1));
        assert_eq!(resolve("/c%d", &doc).unwrap(), &json!(2));
        assert_eq!(resolve("/e^f", &doc).unwrap(), &json!(3));
        assert_eq!(resolve("/g|h", &doc).unwrap(), &json!(4));
        assert_eq!(resolve("/i\\j", &doc).unwrap(), &json!(5));
        assert_eq!(resolve("/k\"l", &doc).unwrap(), &json!(6));
        assert_eq!(resolve("/ ", &doc).unwrap(), &json!(7));
        assert_eq!(resolve("/m~0n", &doc).unwrap(), &json!(8));
    }

    #[test]
    fn relative_pointer_parse() {
        let rp = RelativePointer::parse("0/foo").unwrap();
        assert_eq!(rp.up, 0);
        assert!(matches!(rp.kind, RelativeKind::Pointer(_)));

        let rp = RelativePointer::parse("2#").unwrap();
        assert_eq!(rp.up, 2);
        assert_eq!(rp.kind, RelativeKind::Key);

        let rp = RelativePointer::parse("1").unwrap();
        assert_eq!(rp.up, 1);
    }

    #[test]
    fn relative_pointer_parse_invalid() {
        assert!(RelativePointer::parse("abc").is_err());
    }

    #[test]
    fn relative_pointer_resolve_value() {
        let doc = json!({"a": {"b": 42, "c": 99}});
        let current = JsonPointer::parse("/a/b").unwrap();
        let rp = RelativePointer::parse("1/c").unwrap();
        match rp.resolve(&doc, &current).unwrap() {
            RelativeResult::Value(v) => assert_eq!(v, &json!(99)),
            _ => panic!("expected value"),
        }
    }

    #[test]
    fn relative_pointer_resolve_key() {
        let doc = json!({"a": {"b": 42}});
        let current = JsonPointer::parse("/a/b").unwrap();
        let rp = RelativePointer::parse("1#").unwrap();
        match rp.resolve(&doc, &current).unwrap() {
            RelativeResult::Key(k) => assert_eq!(k, "b"),
            _ => panic!("expected key"),
        }
    }

    #[test]
    fn relative_pointer_above_root() {
        let doc = json!({"a": 1});
        let current = JsonPointer::parse("/a").unwrap();
        let rp = RelativePointer::parse("5").unwrap();
        assert!(rp.resolve(&doc, &current).is_err());
    }

    #[test]
    fn error_display() {
        assert_eq!(
            format!("{}", PointerError::NotFound("x".into())),
            "segment not found: x"
        );
        assert_eq!(
            format!("{}", PointerError::IndexOutOfBounds { index: 5, len: 3 }),
            "index 5 out of bounds (len 3)"
        );
    }

    #[test]
    fn empty_key_segment() {
        // "//" means key = ""
        let p = JsonPointer::parse("//").unwrap();
        assert_eq!(p.segments(), &["", ""]);
        let doc = json!({"": {"": 42}});
        assert_eq!(resolve("//", &doc).unwrap(), &json!(42));
    }

    #[test]
    fn deeply_nested() {
        let doc = json!({"a": {"b": {"c": {"d": {"e": 100}}}}});
        assert_eq!(resolve("/a/b/c/d/e", &doc).unwrap(), &json!(100));
    }
}
