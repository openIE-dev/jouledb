//! Source map (V3 format concepts) — original/generated position mapping,
//! VLQ encoding/decoding, mapping segments, source file references,
//! name references, reverse lookup (generated to original), and source map merging.

use std::collections::HashMap;

// ── Base64 VLQ ──────────────────────────────────────────────────

const BASE64_CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn base64_decode_char(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some(u32::from(c - b'A')),
        b'a'..=b'z' => Some(u32::from(c - b'a') + 26),
        b'0'..=b'9' => Some(u32::from(c - b'0') + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// Encode a signed integer as Base64-VLQ.
pub fn vlq_encode(value: i64) -> String {
    let mut vlq = if value < 0 {
        ((-value) as u64) << 1 | 1
    } else {
        (value as u64) << 1
    };

    let mut out = String::new();
    loop {
        let mut digit = (vlq & 0x1f) as u8;
        vlq >>= 5;
        if vlq > 0 {
            digit |= 0x20; // continuation bit
        }
        out.push(char::from(BASE64_CHARS[digit as usize]));
        if vlq == 0 {
            break;
        }
    }
    out
}

/// Decode one VLQ value from the byte slice, returning (value, bytes_consumed).
pub fn vlq_decode(bytes: &[u8]) -> Option<(i64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0u32;
    let mut consumed = 0;

    for &b in bytes {
        let digit = base64_decode_char(b)?;
        consumed += 1;
        result |= u64::from(digit & 0x1f) << shift;
        shift += 5;
        if digit & 0x20 == 0 {
            break;
        }
    }

    let is_negative = result & 1 == 1;
    let magnitude = result >> 1;
    let value = if is_negative {
        -(magnitude as i64)
    } else {
        magnitude as i64
    };

    Some((value, consumed))
}

/// Encode a sequence of signed integers as VLQ.
pub fn vlq_encode_values(values: &[i64]) -> String {
    values.iter().map(|v| vlq_encode(*v)).collect()
}

/// Decode a VLQ string into a sequence of signed integers.
pub fn vlq_decode_all(encoded: &str) -> Vec<i64> {
    let bytes = encoded.as_bytes();
    let mut values = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        if let Some((val, consumed)) = vlq_decode(&bytes[offset..]) {
            values.push(val);
            offset += consumed;
        } else {
            break;
        }
    }
    values
}

// ── Mapping ─────────────────────────────────────────────────────

/// A single mapping from generated position to original position.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mapping {
    pub generated_line: u32,
    pub generated_col: u32,
    pub source_idx: Option<u32>,
    pub original_line: Option<u32>,
    pub original_col: Option<u32>,
    pub name_idx: Option<u32>,
}

impl Mapping {
    pub fn new(gen_line: u32, gen_col: u32) -> Self {
        Self {
            generated_line: gen_line,
            generated_col: gen_col,
            source_idx: None,
            original_line: None,
            original_col: None,
            name_idx: None,
        }
    }

    pub fn with_source(mut self, source_idx: u32, orig_line: u32, orig_col: u32) -> Self {
        self.source_idx = Some(source_idx);
        self.original_line = Some(orig_line);
        self.original_col = Some(orig_col);
        self
    }

    pub fn with_name(mut self, name_idx: u32) -> Self {
        self.name_idx = Some(name_idx);
        self
    }

    /// Is this mapping fully resolved (has original position)?
    pub fn is_resolved(&self) -> bool {
        self.source_idx.is_some() && self.original_line.is_some() && self.original_col.is_some()
    }
}

// ── MappingSegment ──────────────────────────────────────────────

/// A parsed segment from the mappings string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MappingSegment {
    /// The VLQ-decoded fields for this segment.
    pub fields: Vec<i64>,
}

impl MappingSegment {
    /// Number of fields (1, 4, or 5 in valid source maps).
    pub fn field_count(&self) -> usize {
        self.fields.len()
    }

    /// Does this segment have original position information?
    pub fn has_original(&self) -> bool {
        self.fields.len() >= 4
    }

    /// Does this segment have a name reference?
    pub fn has_name(&self) -> bool {
        self.fields.len() >= 5
    }
}

/// Parse a mappings string into groups of segments (one group per generated line).
pub fn parse_segments(mappings_str: &str) -> Vec<Vec<MappingSegment>> {
    mappings_str
        .split(';')
        .map(|group| {
            if group.is_empty() {
                Vec::new()
            } else {
                group
                    .split(',')
                    .filter(|s| !s.is_empty())
                    .map(|s| MappingSegment {
                        fields: vlq_decode_all(s),
                    })
                    .collect()
            }
        })
        .collect()
}

// ── SourceMap ───────────────────────────────────────────────────

/// A Source Map v3.
#[derive(Debug, Clone)]
pub struct SourceMap {
    pub version: u32,
    pub file: Option<String>,
    pub source_root: Option<String>,
    pub sources: Vec<String>,
    pub sources_content: Vec<Option<String>>,
    pub names: Vec<String>,
    pub mappings: Vec<Mapping>,
    /// Reverse index: (source_idx, orig_line, orig_col) -> Vec of generated positions.
    reverse_index: HashMap<(u32, u32, u32), Vec<(u32, u32)>>,
}

impl Default for SourceMap {
    fn default() -> Self {
        Self {
            version: 3,
            file: None,
            source_root: None,
            sources: Vec::new(),
            sources_content: Vec::new(),
            names: Vec::new(),
            mappings: Vec::new(),
            reverse_index: HashMap::new(),
        }
    }
}

impl SourceMap {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a source file, returning its index.
    pub fn add_source(&mut self, path: impl Into<String>, content: Option<String>) -> u32 {
        let idx = self.sources.len() as u32;
        self.sources.push(path.into());
        self.sources_content.push(content);
        idx
    }

    /// Add a name, returning its index.
    pub fn add_name(&mut self, name: impl Into<String>) -> u32 {
        let idx = self.names.len() as u32;
        self.names.push(name.into());
        idx
    }

    /// Add a mapping and update the reverse index.
    pub fn add_mapping(&mut self, mapping: Mapping) {
        if let (Some(si), Some(ol), Some(oc)) =
            (mapping.source_idx, mapping.original_line, mapping.original_col)
        {
            self.reverse_index
                .entry((si, ol, oc))
                .or_default()
                .push((mapping.generated_line, mapping.generated_col));
        }
        self.mappings.push(mapping);
    }

    /// Resolve a source path with the source root.
    pub fn resolve_source(&self, source_idx: u32) -> Option<String> {
        let source = self.sources.get(source_idx as usize)?;
        match &self.source_root {
            Some(root) => {
                if root.ends_with('/') {
                    Some(format!("{}{}", root, source))
                } else {
                    Some(format!("{}/{}", root, source))
                }
            }
            None => Some(source.clone()),
        }
    }

    /// Get the source content for a given source index.
    pub fn source_content(&self, source_idx: u32) -> Option<&str> {
        self.sources_content
            .get(source_idx as usize)?
            .as_deref()
    }

    /// Generate the `mappings` string in VLQ format.
    pub fn generate_mappings_string(&self) -> String {
        let mut out = String::new();
        let mut prev_gen_line: u32 = 0;
        let mut prev_gen_col: i64 = 0;
        let mut prev_source: i64 = 0;
        let mut prev_orig_line: i64 = 0;
        let mut prev_orig_col: i64 = 0;
        let mut prev_name: i64 = 0;
        let mut first_in_line = true;

        for m in &self.mappings {
            while prev_gen_line < m.generated_line {
                out.push(';');
                prev_gen_line += 1;
                prev_gen_col = 0;
                first_in_line = true;
            }

            if !first_in_line {
                out.push(',');
            }
            first_in_line = false;

            let gen_col = i64::from(m.generated_col);
            out.push_str(&vlq_encode(gen_col - prev_gen_col));
            prev_gen_col = gen_col;

            if let (Some(si), Some(ol), Some(oc)) =
                (m.source_idx, m.original_line, m.original_col)
            {
                let si = i64::from(si);
                let ol = i64::from(ol);
                let oc = i64::from(oc);
                out.push_str(&vlq_encode(si - prev_source));
                out.push_str(&vlq_encode(ol - prev_orig_line));
                out.push_str(&vlq_encode(oc - prev_orig_col));
                prev_source = si;
                prev_orig_line = ol;
                prev_orig_col = oc;

                if let Some(ni) = m.name_idx {
                    let ni = i64::from(ni);
                    out.push_str(&vlq_encode(ni - prev_name));
                    prev_name = ni;
                }
            }
        }

        out
    }

    /// Parse a `mappings` string and populate self.mappings.
    pub fn parse_mappings_string(&mut self, mappings_str: &str) {
        self.mappings.clear();
        self.reverse_index.clear();
        let mut gen_line: u32 = 0;
        let mut prev_source: i64 = 0;
        let mut prev_orig_line: i64 = 0;
        let mut prev_orig_col: i64 = 0;
        let mut prev_name: i64 = 0;

        for group in mappings_str.split(';') {
            if !group.is_empty() {
                let mut prev_col: i64 = 0;
                for segment in group.split(',') {
                    if segment.is_empty() {
                        continue;
                    }
                    let values = vlq_decode_all(segment);
                    if values.is_empty() {
                        continue;
                    }

                    prev_col += values[0];
                    let mut mapping = Mapping::new(gen_line, prev_col as u32);

                    if values.len() >= 4 {
                        prev_source += values[1];
                        prev_orig_line += values[2];
                        prev_orig_col += values[3];
                        mapping.source_idx = Some(prev_source as u32);
                        mapping.original_line = Some(prev_orig_line as u32);
                        mapping.original_col = Some(prev_orig_col as u32);
                    }

                    if values.len() >= 5 {
                        prev_name += values[4];
                        mapping.name_idx = Some(prev_name as u32);
                    }

                    self.add_mapping(mapping);
                }
            }
            gen_line += 1;
        }
    }

    /// Look up the original position for a generated position (forward lookup).
    pub fn lookup(&self, gen_line: u32, gen_col: u32) -> Option<&Mapping> {
        let mut best: Option<&Mapping> = None;
        for m in &self.mappings {
            if m.generated_line == gen_line && m.generated_col <= gen_col {
                match best {
                    Some(b) if b.generated_col > m.generated_col => {}
                    _ => best = Some(m),
                }
            }
        }
        best
    }

    /// Reverse lookup: find all generated positions for an original position.
    pub fn reverse_lookup(&self, source_idx: u32, orig_line: u32, orig_col: u32) -> Vec<(u32, u32)> {
        self.reverse_index
            .get(&(source_idx, orig_line, orig_col))
            .cloned()
            .unwrap_or_default()
    }

    /// Find all mappings that reference a given source index.
    pub fn mappings_for_source(&self, source_idx: u32) -> Vec<&Mapping> {
        self.mappings
            .iter()
            .filter(|m| m.source_idx == Some(source_idx))
            .collect()
    }

    /// Find all mappings on a given generated line.
    pub fn mappings_on_line(&self, gen_line: u32) -> Vec<&Mapping> {
        self.mappings
            .iter()
            .filter(|m| m.generated_line == gen_line)
            .collect()
    }

    /// Compose two source maps: `self` maps A->B, `other` maps B->C, result maps A->C.
    pub fn compose(&self, other: &SourceMap) -> SourceMap {
        let mut result = SourceMap::new();
        result.sources = self.sources.clone();
        result.sources_content = self.sources_content.clone();
        result.names = self.names.clone();

        for m in &other.mappings {
            if let (Some(orig_line), Some(orig_col)) = (m.original_line, m.original_col) {
                if let Some(orig) = self.lookup(orig_line, orig_col) {
                    let mut composed = Mapping::new(m.generated_line, m.generated_col);
                    composed.source_idx = orig.source_idx;
                    composed.original_line = orig.original_line;
                    composed.original_col = orig.original_col;
                    composed.name_idx = orig.name_idx;
                    result.add_mapping(composed);
                } else {
                    result.add_mapping(Mapping::new(m.generated_line, m.generated_col));
                }
            } else {
                result.add_mapping(Mapping::new(m.generated_line, m.generated_col));
            }
        }

        result
    }

    /// Merge another source map into this one, offsetting source indices.
    pub fn merge(&mut self, other: &SourceMap) {
        let source_offset = self.sources.len() as u32;
        let name_offset = self.names.len() as u32;

        for source in &other.sources {
            self.sources.push(source.clone());
        }
        for content in &other.sources_content {
            self.sources_content.push(content.clone());
        }
        for name in &other.names {
            self.names.push(name.clone());
        }

        for m in &other.mappings {
            let mut new_mapping = m.clone();
            if let Some(si) = new_mapping.source_idx {
                new_mapping.source_idx = Some(si + source_offset);
            }
            if let Some(ni) = new_mapping.name_idx {
                new_mapping.name_idx = Some(ni + name_offset);
            }
            self.add_mapping(new_mapping);
        }
    }

    /// Total number of mappings.
    pub fn mapping_count(&self) -> usize {
        self.mappings.len()
    }

    /// Count of resolved mappings (those with original position info).
    pub fn resolved_count(&self) -> usize {
        self.mappings.iter().filter(|m| m.is_resolved()).count()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vlq_encode_zero() {
        assert_eq!(vlq_encode(0), "A");
    }

    #[test]
    fn vlq_encode_positive() {
        assert_eq!(vlq_encode(1), "C");
    }

    #[test]
    fn vlq_encode_negative() {
        assert_eq!(vlq_encode(-1), "D");
    }

    #[test]
    fn vlq_roundtrip() {
        for v in [-100, -1, 0, 1, 50, 1000] {
            let encoded = vlq_encode(v);
            let (decoded, _) = vlq_decode(encoded.as_bytes()).unwrap();
            assert_eq!(decoded, v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn vlq_decode_all_multiple() {
        let encoded = vlq_encode_values(&[0, 5, -3, 10]);
        let decoded = vlq_decode_all(&encoded);
        assert_eq!(decoded, vec![0, 5, -3, 10]);
    }

    #[test]
    fn source_map_add_source() {
        let mut sm = SourceMap::new();
        let idx = sm.add_source("foo.js", Some("var x;".into()));
        assert_eq!(idx, 0);
        assert_eq!(sm.sources[0], "foo.js");
        assert_eq!(sm.sources_content[0].as_deref(), Some("var x;"));
    }

    #[test]
    fn generate_and_parse_mappings() {
        let mut sm = SourceMap::new();
        sm.add_source("a.js", None);

        sm.add_mapping(Mapping::new(0, 0).with_source(0, 0, 0));
        sm.add_mapping(Mapping::new(0, 5).with_source(0, 0, 5));
        sm.add_mapping(Mapping::new(1, 0).with_source(0, 1, 0));

        let mappings_str = sm.generate_mappings_string();
        assert!(mappings_str.contains(';'));

        let mut parsed = SourceMap::new();
        parsed.parse_mappings_string(&mappings_str);

        assert_eq!(parsed.mappings.len(), 3);
        assert_eq!(parsed.mappings[0].generated_line, 0);
        assert_eq!(parsed.mappings[0].generated_col, 0);
        assert_eq!(parsed.mappings[2].generated_line, 1);
    }

    #[test]
    fn lookup_finds_best_match() {
        let mut sm = SourceMap::new();
        sm.add_source("a.js", None);
        sm.add_mapping(Mapping::new(0, 0).with_source(0, 0, 0));
        sm.add_mapping(Mapping::new(0, 10).with_source(0, 0, 10));

        let m = sm.lookup(0, 5).unwrap();
        assert_eq!(m.generated_col, 0);

        let m = sm.lookup(0, 15).unwrap();
        assert_eq!(m.generated_col, 10);
    }

    #[test]
    fn lookup_no_match() {
        let sm = SourceMap::new();
        assert!(sm.lookup(0, 0).is_none());
    }

    #[test]
    fn compose_source_maps() {
        let mut ab = SourceMap::new();
        ab.add_source("original.js", None);
        ab.add_mapping(Mapping::new(0, 0).with_source(0, 5, 0));

        let mut bc = SourceMap::new();
        bc.add_source("intermediate.js", None);
        bc.add_mapping(Mapping::new(0, 0).with_source(0, 0, 0));

        let ac = ab.compose(&bc);
        assert_eq!(ac.mappings.len(), 1);
        assert_eq!(ac.mappings[0].original_line, Some(5));
    }

    #[test]
    fn vlq_large_values() {
        let large = 123456;
        let encoded = vlq_encode(large);
        let (decoded, _) = vlq_decode(encoded.as_bytes()).unwrap();
        assert_eq!(decoded, large);
    }

    #[test]
    fn mapping_with_name() {
        let mut sm = SourceMap::new();
        sm.add_source("a.js", None);
        let ni = sm.add_name("myFunc");
        sm.add_mapping(Mapping::new(0, 0).with_source(0, 0, 0).with_name(ni));
        let s = sm.generate_mappings_string();
        let mut parsed = SourceMap::new();
        parsed.parse_mappings_string(&s);
        assert_eq!(parsed.mappings[0].name_idx, Some(0));
    }

    #[test]
    fn reverse_lookup() {
        let mut sm = SourceMap::new();
        sm.add_source("a.js", None);
        sm.add_mapping(Mapping::new(0, 0).with_source(0, 5, 10));
        sm.add_mapping(Mapping::new(3, 7).with_source(0, 5, 10));

        let positions = sm.reverse_lookup(0, 5, 10);
        assert_eq!(positions.len(), 2);
        assert!(positions.contains(&(0, 0)));
        assert!(positions.contains(&(3, 7)));
    }

    #[test]
    fn reverse_lookup_no_match() {
        let sm = SourceMap::new();
        let positions = sm.reverse_lookup(0, 0, 0);
        assert!(positions.is_empty());
    }

    #[test]
    fn merge_source_maps() {
        let mut sm1 = SourceMap::new();
        let s0 = sm1.add_source("a.js", None);
        sm1.add_mapping(Mapping::new(0, 0).with_source(s0, 0, 0));

        let mut sm2 = SourceMap::new();
        let s1 = sm2.add_source("b.js", None);
        sm2.add_mapping(Mapping::new(1, 0).with_source(s1, 0, 0));

        sm1.merge(&sm2);
        assert_eq!(sm1.sources.len(), 2);
        assert_eq!(sm1.mapping_count(), 2);
        // The merged mapping should reference source index 1 (offset)
        assert_eq!(sm1.mappings[1].source_idx, Some(1));
    }

    #[test]
    fn mappings_for_source() {
        let mut sm = SourceMap::new();
        let s0 = sm.add_source("a.js", None);
        let s1 = sm.add_source("b.js", None);
        sm.add_mapping(Mapping::new(0, 0).with_source(s0, 0, 0));
        sm.add_mapping(Mapping::new(1, 0).with_source(s1, 0, 0));
        sm.add_mapping(Mapping::new(2, 0).with_source(s0, 1, 0));

        let for_a = sm.mappings_for_source(s0);
        assert_eq!(for_a.len(), 2);
    }

    #[test]
    fn mappings_on_line() {
        let mut sm = SourceMap::new();
        sm.add_source("a.js", None);
        sm.add_mapping(Mapping::new(0, 0).with_source(0, 0, 0));
        sm.add_mapping(Mapping::new(0, 5).with_source(0, 0, 5));
        sm.add_mapping(Mapping::new(1, 0).with_source(0, 1, 0));

        assert_eq!(sm.mappings_on_line(0).len(), 2);
        assert_eq!(sm.mappings_on_line(1).len(), 1);
    }

    #[test]
    fn mapping_is_resolved() {
        let resolved = Mapping::new(0, 0).with_source(0, 1, 2);
        assert!(resolved.is_resolved());
        let unresolved = Mapping::new(0, 0);
        assert!(!unresolved.is_resolved());
    }

    #[test]
    fn resolved_count() {
        let mut sm = SourceMap::new();
        sm.add_source("a.js", None);
        sm.add_mapping(Mapping::new(0, 0).with_source(0, 0, 0));
        sm.add_mapping(Mapping::new(1, 0));
        assert_eq!(sm.resolved_count(), 1);
    }

    #[test]
    fn resolve_source_with_root() {
        let mut sm = SourceMap::new();
        sm.source_root = Some("src/".to_string());
        sm.add_source("app.js", None);
        assert_eq!(sm.resolve_source(0).as_deref(), Some("src/app.js"));
    }

    #[test]
    fn resolve_source_no_root() {
        let mut sm = SourceMap::new();
        sm.add_source("app.js", None);
        assert_eq!(sm.resolve_source(0).as_deref(), Some("app.js"));
    }

    #[test]
    fn source_content_lookup() {
        let mut sm = SourceMap::new();
        sm.add_source("a.js", Some("var x = 1;".to_string()));
        sm.add_source("b.js", None);
        assert_eq!(sm.source_content(0), Some("var x = 1;"));
        assert_eq!(sm.source_content(1), None);
    }

    #[test]
    fn test_parse_segments() {
        let segments = super::parse_segments("AAAA,KACG;AACA");
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].len(), 2);
        assert_eq!(segments[1].len(), 1);
        assert!(segments[0][0].has_original());
    }

    #[test]
    fn segment_field_count() {
        let seg = MappingSegment {
            fields: vec![0, 0, 0, 0, 0],
        };
        assert_eq!(seg.field_count(), 5);
        assert!(seg.has_original());
        assert!(seg.has_name());
    }

    #[test]
    fn merge_with_names() {
        let mut sm1 = SourceMap::new();
        sm1.add_source("a.js", None);
        let n0 = sm1.add_name("foo");
        sm1.add_mapping(Mapping::new(0, 0).with_source(0, 0, 0).with_name(n0));

        let mut sm2 = SourceMap::new();
        sm2.add_source("b.js", None);
        let n1 = sm2.add_name("bar");
        sm2.add_mapping(Mapping::new(1, 0).with_source(n1, 0, 0).with_name(0));

        sm1.merge(&sm2);
        assert_eq!(sm1.names.len(), 2);
        assert_eq!(sm1.names[1], "bar");
        // Name index should be offset
        assert_eq!(sm1.mappings[1].name_idx, Some(1));
    }
}
