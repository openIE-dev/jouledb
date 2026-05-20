//! INI file parser and writer.
//!
//! Handles sections, key-value pairs, comments (# and ;), multiline values
//! (trailing backslash), section inheritance, and `${section:key}` interpolation.

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum IniError {
    #[error("duplicate section '{0}' at line {1}")]
    DuplicateSection(String, usize),
    #[error("key '{key}' outside of any section at line {line}")]
    KeyOutsideSection { key: String, line: usize },
    #[error("invalid line at {0}: {1}")]
    InvalidLine(usize, String),
    #[error("circular interpolation detected for '${{{0}:{1}}}'")]
    CircularInterpolation(String, String),
    #[error("unresolved reference '${{{section}:{key}}}'")]
    UnresolvedReference { section: String, key: String },
    #[error("section '{0}' not found")]
    SectionNotFound(String),
    #[error("key '{key}' not found in section '{section}'")]
    KeyNotFound { section: String, key: String },
}

// ── Config ──────────────────────────────────────────────────────

/// Configuration for INI parsing.
#[derive(Debug, Clone)]
pub struct IniConfig {
    /// Allow duplicate sections (merge keys). Default: true.
    pub allow_duplicate_sections: bool,
    /// Allow keys without a section (placed in a default section). Default: true.
    pub allow_global_keys: bool,
    /// Name for the default (global) section. Default: "".
    pub default_section: String,
    /// Delimiter between key and value. Default: '='.
    pub delimiter: char,
    /// Enable `${section:key}` interpolation. Default: true.
    pub interpolation: bool,
    /// Support multiline values (trailing backslash). Default: true.
    pub multiline: bool,
    /// Support section inheritance via `[child : parent]`. Default: true.
    pub inheritance: bool,
}

impl Default for IniConfig {
    fn default() -> Self {
        Self {
            allow_duplicate_sections: true,
            allow_global_keys: true,
            default_section: String::new(),
            delimiter: '=',
            interpolation: true,
            multiline: true,
            inheritance: true,
        }
    }
}

// ── Section ─────────────────────────────────────────────────────

/// A section in an INI file.
#[derive(Debug, Clone)]
pub struct IniSection {
    /// Section name.
    pub name: String,
    /// Parent section name for inheritance.
    pub parent: Option<String>,
    /// Ordered key-value pairs.
    entries: Vec<(String, String)>,
    /// Fast lookup by key.
    map: HashMap<String, String>,
    /// Comments associated with this section header.
    pub comments: Vec<String>,
}

impl IniSection {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            parent: None,
            entries: Vec::new(),
            map: HashMap::new(),
            comments: Vec::new(),
        }
    }

    /// Get a value by key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.map.get(key).map(|s| s.as_str())
    }

    /// Set a key-value pair.
    pub fn set(&mut self, key: &str, value: &str) {
        if self.map.contains_key(key) {
            self.map.insert(key.to_string(), value.to_string());
            for entry in &mut self.entries {
                if entry.0 == key {
                    entry.1 = value.to_string();
                    break;
                }
            }
        } else {
            self.map.insert(key.to_string(), value.to_string());
            self.entries.push((key.to_string(), value.to_string()));
        }
    }

    /// Remove a key.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        let val = self.map.remove(key);
        if val.is_some() {
            self.entries.retain(|(k, _)| k != key);
        }
        val
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    /// Number of key-value pairs.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the section has no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over key-value pairs in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// All keys.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(k, _)| k.as_str())
    }
}

// ── Document ────────────────────────────────────────────────────

/// A parsed INI document.
#[derive(Debug, Clone)]
pub struct IniDocument {
    /// Section names in order of appearance.
    section_order: Vec<String>,
    /// Sections by name.
    sections: HashMap<String, IniSection>,
    /// Config used for parsing.
    config: IniConfig,
}

impl IniDocument {
    /// Create an empty INI document.
    pub fn new() -> Self {
        Self {
            section_order: Vec::new(),
            sections: HashMap::new(),
            config: IniConfig::default(),
        }
    }

    /// Get a section by name.
    pub fn section(&self, name: &str) -> Option<&IniSection> {
        self.sections.get(name)
    }

    /// Get a mutable section by name.
    pub fn section_mut(&mut self, name: &str) -> Option<&mut IniSection> {
        self.sections.get_mut(name)
    }

    /// Get or create a section.
    pub fn section_or_create(&mut self, name: &str) -> &mut IniSection {
        if !self.sections.contains_key(name) {
            self.section_order.push(name.to_string());
            self.sections.insert(name.to_string(), IniSection::new(name));
        }
        self.sections.get_mut(name).unwrap()
    }

    /// List all section names.
    pub fn section_names(&self) -> &[String] {
        &self.section_order
    }

    /// Number of sections.
    pub fn len(&self) -> usize {
        self.sections.len()
    }

    /// Whether there are no sections.
    pub fn is_empty(&self) -> bool {
        self.sections.is_empty()
    }

    /// Get a value using `section.key` notation.
    pub fn get(&self, section: &str, key: &str) -> Option<&str> {
        self.sections.get(section).and_then(|s| s.get(key))
    }

    /// Set a value in a section.
    pub fn set(&mut self, section: &str, key: &str, value: &str) {
        self.section_or_create(section).set(key, value);
    }

    /// Get a value with inheritance — if not found in the section,
    /// look in the parent section.
    pub fn get_inherited(&self, section: &str, key: &str) -> Option<&str> {
        let sec = self.sections.get(section)?;
        if let Some(val) = sec.get(key) {
            return Some(val);
        }
        if let Some(parent_name) = &sec.parent {
            return self.get_inherited(parent_name, key);
        }
        None
    }

    /// Perform `${section:key}` interpolation on all values.
    pub fn interpolate(&mut self) -> Result<(), IniError> {
        // Collect all (section, key, raw_value) triples.
        let entries: Vec<(String, String, String)> = self.sections.iter()
            .flat_map(|(sname, sec)| {
                sec.entries.iter().map(move |(k, v)| (sname.clone(), k.clone(), v.clone()))
            })
            .collect();

        for (sname, key, raw) in &entries {
            let resolved = self.resolve_value(raw, sname, &mut Vec::new())?;
            if let Some(sec) = self.sections.get_mut(sname.as_str()) {
                sec.set(key, &resolved);
            }
        }
        Ok(())
    }

    fn resolve_value(&self, value: &str, current_section: &str, visited: &mut Vec<(String, String)>) -> Result<String, IniError> {
        let mut result = String::new();
        let chars: Vec<char> = value.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if i + 1 < chars.len() && chars[i] == '$' && chars[i + 1] == '{' {
                // Find closing brace
                if let Some(end) = chars[i + 2..].iter().position(|c| *c == '}') {
                    let ref_str: String = chars[i + 2..i + 2 + end].iter().collect();
                    let (ref_sec, ref_key) = if let Some(colon_pos) = ref_str.find(':') {
                        (&ref_str[..colon_pos], &ref_str[colon_pos + 1..])
                    } else {
                        (current_section, ref_str.as_str())
                    };
                    let pair = (ref_sec.to_string(), ref_key.to_string());
                    if visited.contains(&pair) {
                        return Err(IniError::CircularInterpolation(ref_sec.to_string(), ref_key.to_string()));
                    }
                    visited.push(pair);
                    let ref_val = self.get(ref_sec, ref_key)
                        .ok_or_else(|| IniError::UnresolvedReference {
                            section: ref_sec.to_string(),
                            key: ref_key.to_string(),
                        })?;
                    let resolved = self.resolve_value(ref_val, ref_sec, visited)?;
                    visited.pop();
                    result.push_str(&resolved);
                    i += 2 + end + 1;
                    continue;
                }
            }
            result.push(chars[i]);
            i += 1;
        }
        Ok(result)
    }

    /// Remove a section.
    pub fn remove_section(&mut self, name: &str) -> Option<IniSection> {
        self.section_order.retain(|n| n != name);
        self.sections.remove(name)
    }
}

impl Default for IniDocument {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// Parse an INI string with default config.
pub fn parse(input: &str) -> Result<IniDocument, IniError> {
    parse_with(input, &IniConfig::default())
}

/// Parse an INI string with custom config.
pub fn parse_with(input: &str, config: &IniConfig) -> Result<IniDocument, IniError> {
    let mut doc = IniDocument {
        section_order: Vec::new(),
        sections: HashMap::new(),
        config: config.clone(),
    };

    let mut current_section: Option<String> = None;
    let mut pending_comments: Vec<String> = Vec::new();
    let lines: Vec<&str> = input.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line_num = i + 1;
        let raw = lines[i];
        let trimmed = raw.trim();

        // Empty line
        if trimmed.is_empty() {
            i += 1;
            continue;
        }

        // Comment
        if trimmed.starts_with('#') || trimmed.starts_with(';') {
            pending_comments.push(trimmed.to_string());
            i += 1;
            continue;
        }

        // Section header
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let inner = trimmed[1..trimmed.len() - 1].trim();
            let (name, parent) = if config.inheritance {
                if let Some(colon_pos) = inner.find(':') {
                    let n = inner[..colon_pos].trim();
                    let p = inner[colon_pos + 1..].trim();
                    (n.to_string(), Some(p.to_string()))
                } else {
                    (inner.to_string(), None)
                }
            } else {
                (inner.to_string(), None)
            };

            if doc.sections.contains_key(&name) && !config.allow_duplicate_sections {
                return Err(IniError::DuplicateSection(name, line_num));
            }

            if !doc.sections.contains_key(&name) {
                doc.section_order.push(name.clone());
                let mut sec = IniSection::new(&name);
                sec.parent = parent;
                sec.comments = std::mem::take(&mut pending_comments);
                doc.sections.insert(name.clone(), sec);
            }
            current_section = Some(name);
            i += 1;
            continue;
        }

        // Key-value pair
        if let Some(delim_pos) = trimmed.find(config.delimiter) {
            let key = trimmed[..delim_pos].trim().to_string();
            let mut value = trimmed[delim_pos + 1..].trim().to_string();

            // Multiline continuation
            if config.multiline {
                while value.ends_with('\\') && i + 1 < lines.len() {
                    value.pop(); // remove trailing backslash
                    i += 1;
                    let cont = lines[i].trim();
                    value.push_str(cont);
                }
            }

            let sec_name = if let Some(sn) = &current_section {
                sn.clone()
            } else if config.allow_global_keys {
                let sn = config.default_section.clone();
                if !doc.sections.contains_key(&sn) {
                    doc.section_order.push(sn.clone());
                    doc.sections.insert(sn.clone(), IniSection::new(&sn));
                }
                sn
            } else {
                return Err(IniError::KeyOutsideSection { key, line: line_num });
            };

            if let Some(sec) = doc.sections.get_mut(&sec_name) {
                sec.set(&key, &value);
            }
            i += 1;
            continue;
        }

        i += 1;
    }

    if config.interpolation {
        doc.interpolate()?;
    }

    Ok(doc)
}

// ── Writer ──────────────────────────────────────────────────────

/// Write an INI document to a string.
pub fn write(doc: &IniDocument) -> String {
    let mut output = String::new();
    for (idx, sec_name) in doc.section_order.iter().enumerate() {
        let sec = match doc.sections.get(sec_name) {
            Some(s) => s,
            None => continue,
        };
        if idx > 0 {
            output.push('\n');
        }
        for comment in &sec.comments {
            output.push_str(comment);
            output.push('\n');
        }
        if !sec_name.is_empty() {
            if let Some(parent) = &sec.parent {
                output.push_str(&format!("[{} : {}]\n", sec_name, parent));
            } else {
                output.push_str(&format!("[{}]\n", sec_name));
            }
        }
        for (key, value) in &sec.entries {
            output.push_str(&format!("{} = {}\n", key, value));
        }
    }
    output
}

impl fmt::Display for IniDocument {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", write(self))
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parse() {
        let ini = "[server]\nhost = localhost\nport = 8080";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("server", "host"), Some("localhost"));
        assert_eq!(doc.get("server", "port"), Some("8080"));
    }

    #[test]
    fn test_multiple_sections() {
        let ini = "[db]\nhost = db.local\n\n[cache]\nhost = cache.local";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("db", "host"), Some("db.local"));
        assert_eq!(doc.get("cache", "host"), Some("cache.local"));
    }

    #[test]
    fn test_comments_hash() {
        let ini = "# this is a comment\n[main]\nkey = val";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("main", "key"), Some("val"));
        assert_eq!(doc.section("main").unwrap().comments.len(), 1);
    }

    #[test]
    fn test_comments_semicolon() {
        let ini = "; semicolon comment\n[main]\nkey = val";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("main", "key"), Some("val"));
    }

    #[test]
    fn test_multiline_value() {
        let ini = "[main]\npath = /usr/\\\nlocal/bin";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("main", "path"), Some("/usr/local/bin"));
    }

    #[test]
    fn test_global_keys() {
        let ini = "key = global_value\n[section]\nother = val";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("", "key"), Some("global_value"));
    }

    #[test]
    fn test_global_keys_disallowed() {
        let config = IniConfig { allow_global_keys: false, ..Default::default() };
        let ini = "key = value";
        let result = parse_with(ini, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_inheritance() {
        let ini = "[base]\ncolor = red\nsize = 10\n\n[child : base]\nsize = 20";
        let config = IniConfig { interpolation: false, ..Default::default() };
        let doc = parse_with(ini, &config).unwrap();
        assert_eq!(doc.get_inherited("child", "color"), Some("red"));
        assert_eq!(doc.get_inherited("child", "size"), Some("20"));
    }

    #[test]
    fn test_interpolation() {
        let ini = "[paths]\nbase = /opt\nbin = ${paths:base}/bin";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("paths", "bin"), Some("/opt/bin"));
    }

    #[test]
    fn test_circular_interpolation() {
        let ini = "[a]\nx = ${a:y}\ny = ${a:x}";
        let result = parse(ini);
        assert!(result.is_err());
    }

    #[test]
    fn test_duplicate_section_merge() {
        let ini = "[server]\nhost = a\n\n[server]\nport = 80";
        let doc = parse(ini).unwrap();
        assert_eq!(doc.get("server", "host"), Some("a"));
        assert_eq!(doc.get("server", "port"), Some("80"));
    }

    #[test]
    fn test_duplicate_section_error() {
        let config = IniConfig { allow_duplicate_sections: false, interpolation: false, ..Default::default() };
        let ini = "[server]\nhost = a\n\n[server]\nport = 80";
        let result = parse_with(ini, &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_write_roundtrip() {
        let ini = "[database]\nhost = localhost\nport = 5432\n";
        let config = IniConfig { interpolation: false, ..Default::default() };
        let doc = parse_with(ini, &config).unwrap();
        let output = write(&doc);
        assert!(output.contains("[database]"));
        assert!(output.contains("host = localhost"));
        assert!(output.contains("port = 5432"));
    }

    #[test]
    fn test_section_mutation() {
        let mut doc = IniDocument::new();
        doc.set("app", "name", "test");
        doc.set("app", "version", "1.0");
        assert_eq!(doc.get("app", "name"), Some("test"));
        assert_eq!(doc.section("app").unwrap().len(), 2);
    }

    #[test]
    fn test_remove_key() {
        let mut doc = IniDocument::new();
        doc.set("a", "x", "1");
        doc.set("a", "y", "2");
        doc.section_mut("a").unwrap().remove("x");
        assert!(doc.get("a", "x").is_none());
        assert_eq!(doc.get("a", "y"), Some("2"));
    }

    #[test]
    fn test_remove_section() {
        let mut doc = IniDocument::new();
        doc.set("a", "x", "1");
        doc.set("b", "y", "2");
        doc.remove_section("a");
        assert!(doc.section("a").is_none());
        assert_eq!(doc.len(), 1);
    }

    #[test]
    fn test_section_keys_iter() {
        let ini = "[s]\na = 1\nb = 2\nc = 3";
        let config = IniConfig { interpolation: false, ..Default::default() };
        let doc = parse_with(ini, &config).unwrap();
        let keys: Vec<&str> = doc.section("s").unwrap().keys().collect();
        assert_eq!(keys, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_empty_input() {
        let doc = parse("").unwrap();
        assert!(doc.is_empty());
    }

    #[test]
    fn test_whitespace_around_delimiter() {
        let ini = "[s]\n  key  =  value  ";
        let config = IniConfig { interpolation: false, ..Default::default() };
        let doc = parse_with(ini, &config).unwrap();
        assert_eq!(doc.get("s", "key"), Some("value"));
    }

    #[test]
    fn test_contains_key() {
        let mut doc = IniDocument::new();
        doc.set("s", "k", "v");
        assert!(doc.section("s").unwrap().contains_key("k"));
        assert!(!doc.section("s").unwrap().contains_key("missing"));
    }
}
