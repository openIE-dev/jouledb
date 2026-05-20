// SPDX-License-Identifier: MIT
//! YAML subset parser -- mappings, sequences, scalars, multi-line strings,
//! anchors/aliases, flow/block styles. Outputs `serde_json::Value`.

use serde_json::{Map, Value};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YamlError {
    UnexpectedEof,
    UnexpectedChar(char, usize),
    InvalidIndentation(usize),
    UndefinedAlias(String),
    InvalidScalar(String),
    UnterminatedString,
}

impl std::fmt::Display for YamlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::UnexpectedChar(c, p) => write!(f, "unexpected char '{c}' at pos {p}"),
            Self::InvalidIndentation(line) => write!(f, "invalid indentation at line {line}"),
            Self::UndefinedAlias(n) => write!(f, "undefined alias: {n}"),
            Self::InvalidScalar(s) => write!(f, "invalid scalar: {s}"),
            Self::UnterminatedString => write!(f, "unterminated string"),
        }
    }
}

// ── Line-based parser ───────────────────────────────────────────────────────

struct Line {
    indent: usize,
    content: String,
}

struct Parser {
    lines: Vec<Line>,
    pos: usize,
    anchors: HashMap<String, Value>,
}

impl Parser {
    fn new(input: &str) -> Self {
        let lines: Vec<Line> = input
            .lines()
            .map(|raw| {
                let trimmed = raw.trim_start();
                let indent = raw.len() - trimmed.len();
                let content = strip_inline_comment(trimmed).to_string();
                Line { indent, content }
            })
            .collect();
        Self { lines, pos: 0, anchors: HashMap::new() }
    }

    fn skip_empty(&mut self) {
        while self.pos < self.lines.len() {
            let c = &self.lines[self.pos].content;
            if c.is_empty() || c.starts_with('#') { self.pos += 1; } else { break; }
        }
    }

    fn peek(&self) -> Option<(&str, usize)> {
        let mut p = self.pos;
        while p < self.lines.len() {
            let ln = &self.lines[p];
            if !ln.content.is_empty() && !ln.content.starts_with('#') {
                return Some((&ln.content, ln.indent));
            }
            p += 1;
        }
        None
    }

    // ── Document ────────────────────────────────────────────────────────

    fn parse_document(&mut self) -> Result<Value, YamlError> {
        self.skip_empty();
        if let Some((c, _)) = self.peek() {
            if c == "---" { self.pos += 1; self.skip_empty(); }
        }
        if self.pos >= self.lines.len() { return Ok(Value::Null); }
        self.parse_node(0)
    }

    fn parse_node(&mut self, min_indent: usize) -> Result<Value, YamlError> {
        self.skip_empty();
        let (content, indent) = match self.peek() {
            Some((c, i)) => (c.to_string(), i),
            None => return Ok(Value::Null),
        };
        if indent < min_indent { return Ok(Value::Null); }

        // Anchor
        if content.starts_with('&') { return self.parse_anchor(min_indent); }
        // Alias
        if content.starts_with('*') { return self.parse_alias(); }
        // Block sequence
        if content.starts_with("- ") || content == "-" { return self.parse_block_seq(indent); }
        // Flow sequence
        if content.starts_with('[') { return self.parse_flow_value(); }
        // Flow mapping
        if content.starts_with('{') { return self.parse_flow_value(); }
        // Literal block
        if content == "|" || content.starts_with("| ") {
            self.pos += 1;
            return self.parse_literal_block(indent);
        }
        // Folded block
        if content == ">" || content.starts_with("> ") {
            self.pos += 1;
            return self.parse_folded_block(indent);
        }
        // Block mapping
        if find_colon(&content).is_some() { return self.parse_block_map(indent); }
        // Plain scalar
        self.pos += 1;
        Ok(parse_scalar(&content))
    }

    // ── Block sequence ──────────────────────────────────────────────────

    fn parse_block_seq(&mut self, seq_indent: usize) -> Result<Value, YamlError> {
        let mut items = Vec::new();
        loop {
            self.skip_empty();
            let (content, indent) = match self.peek() {
                Some((c, i)) => (c.to_string(), i),
                None => break,
            };
            if indent != seq_indent || !content.starts_with('-') { break; }
            self.pos += 1;
            let rest = content[1..].trim_start();
            if rest.is_empty() {
                items.push(self.parse_node(seq_indent + 1)?);
            } else if find_colon(rest).is_some() {
                // Inline mapping as sequence element
                items.push(self.parse_inline_map_in_seq(rest, seq_indent)?);
            } else if rest.starts_with('[') || rest.starts_with('{') {
                items.push(parse_flow_value(rest)?);
            } else {
                items.push(parse_scalar(rest));
            }
        }
        Ok(Value::Array(items))
    }

    fn parse_inline_map_in_seq(&mut self, first_line: &str, seq_indent: usize) -> Result<Value, YamlError> {
        let mut map = Map::new();
        // Parse first k:v from the dash line
        if let Some(cp) = find_colon(first_line) {
            let k = first_line[..cp].trim();
            let v_str = first_line[cp + 1..].trim();
            let val = if v_str.is_empty() {
                self.parse_node(seq_indent + 2)?
            } else if v_str.starts_with('[') || v_str.starts_with('{') {
                parse_flow_value(v_str)?
            } else {
                parse_scalar(v_str)
            };
            map.insert(k.to_string(), val);
        }
        // Continue reading deeper indented mapping entries
        loop {
            self.skip_empty();
            let (c, i) = match self.peek() {
                Some((c, i)) => (c.to_string(), i),
                None => break,
            };
            if i <= seq_indent || c.starts_with('-') { break; }
            if let Some(cp) = find_colon(&c) {
                let k = c[..cp].trim().to_string();
                let v_str = c[cp + 1..].trim().to_string();
                self.pos += 1;
                let val = if v_str.is_empty() {
                    self.parse_node(i + 1)?
                } else if v_str.starts_with('[') || v_str.starts_with('{') {
                    parse_flow_value(&v_str)?
                } else if v_str == "|" {
                    self.parse_literal_block(i)?
                } else if v_str == ">" {
                    self.parse_folded_block(i)?
                } else {
                    parse_scalar(&v_str)
                };
                map.insert(k, val);
            } else {
                break;
            }
        }
        Ok(Value::Object(map))
    }

    // ── Block mapping ───────────────────────────────────────────────────

    fn parse_block_map(&mut self, map_indent: usize) -> Result<Value, YamlError> {
        let mut map = Map::new();
        loop {
            self.skip_empty();
            let (content, indent) = match self.peek() {
                Some((c, i)) => (c.to_string(), i),
                None => break,
            };
            if indent != map_indent { break; }
            let cp = match find_colon(&content) {
                Some(p) => p,
                None => break,
            };
            self.pos += 1;
            let key = content[..cp].trim().to_string();

            // Merge key
            if key == "<<" {
                let v_str = content[cp + 1..].trim();
                if v_str.starts_with('*') {
                    let alias = v_str[1..].trim();
                    if let Some(anchor_val) = self.anchors.get(alias) {
                        if let Value::Object(merged) = anchor_val.clone() {
                            for (mk, mv) in merged {
                                map.entry(mk).or_insert(mv);
                            }
                        }
                    }
                }
                continue;
            }

            let v_str = content[cp + 1..].trim().to_string();
            let value = if v_str.is_empty() {
                self.parse_node(map_indent + 1)?
            } else if v_str == "|" {
                self.parse_literal_block(map_indent)?
            } else if v_str == ">" {
                self.parse_folded_block(map_indent)?
            } else if v_str.starts_with('*') {
                let alias = v_str[1..].trim();
                self.anchors.get(alias)
                    .cloned()
                    .ok_or_else(|| YamlError::UndefinedAlias(alias.into()))?
            } else if v_str.starts_with('&') {
                let rest = &v_str[1..];
                let (anchor, scalar) = rest.split_once(' ').unwrap_or((rest, ""));
                let val = if scalar.is_empty() {
                    // Anchor with no inline value — parse the nested block
                    self.parse_node(map_indent + 1)?
                } else {
                    parse_scalar(scalar)
                };
                self.anchors.insert(anchor.to_string(), val.clone());
                val
            } else if v_str.starts_with('[') || v_str.starts_with('{') {
                parse_flow_value(&v_str)?
            } else {
                parse_scalar(&v_str)
            };
            map.insert(key, value);
        }
        Ok(Value::Object(map))
    }

    // ── Literal / Folded blocks ─────────────────────────────────────────

    fn parse_literal_block(&mut self, base_indent: usize) -> Result<Value, YamlError> {
        let mut lines = Vec::new();
        let mut block_indent: Option<usize> = None;
        while self.pos < self.lines.len() {
            let ln = &self.lines[self.pos];
            if ln.content.is_empty() {
                lines.push(String::new());
                self.pos += 1;
                continue;
            }
            if ln.indent <= base_indent { break; }
            if block_indent.is_none() { block_indent = Some(ln.indent); }
            let bi = block_indent.unwrap();
            let extra = ln.indent.saturating_sub(bi);
            lines.push(format!("{}{}", " ".repeat(extra), ln.content));
            self.pos += 1;
        }
        while lines.last().map_or(false, |l| l.is_empty()) { lines.pop(); }
        Ok(Value::String(lines.join("\n") + "\n"))
    }

    fn parse_folded_block(&mut self, base_indent: usize) -> Result<Value, YamlError> {
        let mut paragraphs: Vec<String> = Vec::new();
        let mut current = String::new();
        let mut _block_indent: Option<usize> = None;
        while self.pos < self.lines.len() {
            let ln = &self.lines[self.pos];
            if ln.content.is_empty() {
                if !current.is_empty() { paragraphs.push(current.clone()); current.clear(); }
                paragraphs.push(String::new());
                self.pos += 1;
                continue;
            }
            if ln.indent <= base_indent { break; }
            if _block_indent.is_none() { _block_indent = Some(ln.indent); }
            if !current.is_empty() { current.push(' '); }
            current.push_str(&ln.content);
            self.pos += 1;
        }
        if !current.is_empty() { paragraphs.push(current); }
        while paragraphs.last().map_or(false, |l| l.is_empty()) { paragraphs.pop(); }
        Ok(Value::String(paragraphs.join("\n") + "\n"))
    }

    // ── Anchor / Alias ──────────────────────────────────────────────────

    fn parse_anchor(&mut self, min_indent: usize) -> Result<Value, YamlError> {
        let content = self.lines[self.pos].content.clone();
        let rest = &content[1..];
        let (anchor_name, remainder) = rest.split_once(' ').unwrap_or((rest, ""));
        let anchor_name = anchor_name.to_string();
        if remainder.is_empty() {
            self.pos += 1;
            let val = self.parse_node(min_indent)?;
            self.anchors.insert(anchor_name, val.clone());
            Ok(val)
        } else {
            let indent = self.lines[self.pos].indent;
            self.lines[self.pos] = Line { indent, content: remainder.to_string() };
            let val = self.parse_node(min_indent)?;
            self.anchors.insert(anchor_name, val.clone());
            Ok(val)
        }
    }

    fn parse_alias(&mut self) -> Result<Value, YamlError> {
        let content = &self.lines[self.pos].content;
        let name = content[1..].trim().to_string();
        self.pos += 1;
        self.anchors.get(&name).cloned().ok_or_else(|| YamlError::UndefinedAlias(name))
    }

    fn parse_flow_value(&mut self) -> Result<Value, YamlError> {
        let content = self.lines[self.pos].content.clone();
        self.pos += 1;
        parse_flow_value(&content)
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────────

fn strip_inline_comment(s: &str) -> &str {
    let mut in_sq = false;
    let mut in_dq = false;
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'\'' if !in_dq => in_sq = !in_sq,
            b'"' if !in_sq => in_dq = !in_dq,
            b'#' if !in_sq && !in_dq && (i == 0 || bytes[i - 1] == b' ') => {
                return s[..i].trim_end();
            }
            _ => {}
        }
    }
    s
}

fn find_colon(s: &str) -> Option<usize> {
    let mut in_sq = false;
    let mut in_dq = false;
    let bytes = s.as_bytes();
    for i in 0..bytes.len() {
        match bytes[i] {
            b'\'' if !in_dq => in_sq = !in_sq,
            b'"' if !in_sq => in_dq = !in_dq,
            b':' if !in_sq && !in_dq => {
                if i + 1 >= bytes.len() || bytes[i + 1] == b' ' {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_scalar(s: &str) -> Value {
    let s = s.trim();
    if s.is_empty() || s == "~" || s == "null" || s == "Null" || s == "NULL" {
        return Value::Null;
    }
    if s == "true" || s == "True" || s == "TRUE" { return Value::Bool(true); }
    if s == "false" || s == "False" || s == "FALSE" { return Value::Bool(false); }

    // Quoted strings
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return Value::String(s[1..s.len() - 1].to_string());
    }
    // Integer
    if let Ok(n) = s.parse::<i64>() { return serde_json::json!(n); }
    // Hex
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        if let Ok(n) = i64::from_str_radix(hex, 16) { return serde_json::json!(n); }
    }
    // Float
    if let Ok(f) = s.parse::<f64>() { return serde_json::json!(f); }
    // Special floats
    if s == ".inf" || s == ".Inf" || s == ".INF" { return serde_json::json!(f64::INFINITY); }
    if s == "-.inf" || s == "-.Inf" || s == "-.INF" { return serde_json::json!(f64::NEG_INFINITY); }
    if s == ".nan" || s == ".NaN" || s == ".NAN" { return Value::Null; }

    Value::String(s.to_string())
}

fn parse_flow_value(s: &str) -> Result<Value, YamlError> {
    let s = s.trim();
    if s.starts_with('[') && s.ends_with(']') {
        let inner = &s[1..s.len() - 1];
        let items = split_flow(inner);
        let values: Result<Vec<_>, _> = items.iter()
            .filter(|i| !i.trim().is_empty())
            .map(|item| {
                let t = item.trim();
                if t.starts_with('[') || t.starts_with('{') {
                    parse_flow_value(t)
                } else {
                    Ok(parse_scalar(t))
                }
            })
            .collect();
        Ok(Value::Array(values?))
    } else if s.starts_with('{') && s.ends_with('}') {
        let inner = &s[1..s.len() - 1];
        let items = split_flow(inner);
        let mut map = Map::new();
        for item in &items {
            let t = item.trim();
            if t.is_empty() { continue; }
            if let Some(cp) = find_colon(t) {
                let k = t[..cp].trim();
                let v = t[cp + 1..].trim();
                let key = match parse_scalar(k) {
                    Value::String(s) => s,
                    other => format!("{other}"),
                };
                let val = if v.starts_with('[') || v.starts_with('{') {
                    parse_flow_value(v)?
                } else {
                    parse_scalar(v)
                };
                map.insert(key, val);
            }
        }
        Ok(Value::Object(map))
    } else {
        Ok(parse_scalar(s))
    }
}

fn split_flow(s: &str) -> Vec<String> {
    let mut items = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in s.chars() {
        match c {
            '[' | '{' => { depth += 1; current.push(c); }
            ']' | '}' => { depth -= 1; current.push(c); }
            ',' if depth == 0 => { items.push(current.clone()); current.clear(); }
            _ => current.push(c),
        }
    }
    if !current.is_empty() { items.push(current); }
    items
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Parse a YAML string into a `serde_json::Value`.
pub fn parse(input: &str) -> Result<Value, YamlError> {
    let mut parser = Parser::new(input);
    parser.parse_document()
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn scalar_types() {
        assert_eq!(parse_scalar("42"), json!(42));
        assert_eq!(parse_scalar("-7"), json!(-7));
        assert_eq!(parse_scalar("3.14"), json!(3.14));
        assert_eq!(parse_scalar("true"), json!(true));
        assert_eq!(parse_scalar("false"), json!(false));
        assert_eq!(parse_scalar("null"), Value::Null);
        assert_eq!(parse_scalar("~"), Value::Null);
        assert_eq!(parse_scalar("hello"), json!("hello"));
    }

    #[test]
    fn block_sequence() {
        let v = parse("- apple\n- banana\n- cherry").unwrap();
        assert_eq!(v, json!(["apple", "banana", "cherry"]));
    }

    #[test]
    fn block_mapping() {
        let v = parse("name: Alice\nage: 30\nactive: true").unwrap();
        assert_eq!(v["name"], json!("Alice"));
        assert_eq!(v["age"], json!(30));
        assert_eq!(v["active"], json!(true));
    }

    #[test]
    fn nested_mapping() {
        let v = parse("server:\n  host: localhost\n  port: 8080").unwrap();
        assert_eq!(v["server"]["host"], json!("localhost"));
        assert_eq!(v["server"]["port"], json!(8080));
    }

    #[test]
    fn flow_sequence() {
        let v = parse("[1, 2, 3]").unwrap();
        assert_eq!(v, json!([1, 2, 3]));
    }

    #[test]
    fn flow_mapping() {
        let v = parse("{name: Alice, age: 30}").unwrap();
        assert_eq!(v["name"], json!("Alice"));
        assert_eq!(v["age"], json!(30));
    }

    #[test]
    fn literal_block() {
        let input = "text: |\n  line one\n  line two";
        let v = parse(input).unwrap();
        let text = v["text"].as_str().unwrap();
        assert!(text.contains("line one"));
        assert!(text.contains("line two"));
        assert!(text.ends_with('\n'));
    }

    #[test]
    fn folded_block() {
        let input = "text: >\n  this is a\n  long line";
        let v = parse(input).unwrap();
        let text = v["text"].as_str().unwrap();
        assert!(text.contains("this is a long line"));
    }

    #[test]
    fn null_values() {
        assert_eq!(parse_scalar("null"), Value::Null);
        assert_eq!(parse_scalar("~"), Value::Null);
        assert_eq!(parse_scalar("NULL"), Value::Null);
    }

    #[test]
    fn document_start_marker() {
        let v = parse("---\nkey: value").unwrap();
        assert_eq!(v["key"], json!("value"));
    }

    #[test]
    fn sequence_of_mappings() {
        let input = "- name: Alice\n  age: 30\n- name: Bob\n  age: 25";
        let v = parse(input).unwrap();
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["name"], json!("Alice"));
        assert_eq!(arr[1]["age"], json!(25));
    }

    #[test]
    fn comments_stripped() {
        let v = parse("key: value # comment\n# full line\nother: data").unwrap();
        assert_eq!(v["key"], json!("value"));
        assert_eq!(v["other"], json!("data"));
    }

    #[test]
    fn quoted_strings() {
        assert_eq!(parse_scalar("\"hello world\""), json!("hello world"));
        assert_eq!(parse_scalar("'hello world'"), json!("hello world"));
    }

    #[test]
    fn empty_document() {
        assert_eq!(parse("").unwrap(), Value::Null);
    }

    #[test]
    fn nested_flow() {
        let v = parse("data: [1, [2, 3], 4]").unwrap();
        assert_eq!(v["data"], json!([1, [2, 3], 4]));
    }

    #[test]
    fn anchor_and_alias() {
        let input = "defaults: &defaults\n  adapter: postgres\n  host: localhost\nproduction:\n  <<: *defaults\n  database: prod_db";
        let v = parse(input).unwrap();
        assert_eq!(v["production"]["adapter"], json!("postgres"));
        assert_eq!(v["production"]["database"], json!("prod_db"));
    }

    #[test]
    fn inline_anchor() {
        let input = "color: &primary blue\nheading: *primary";
        let v = parse(input).unwrap();
        assert_eq!(v["color"], json!("blue"));
        assert_eq!(v["heading"], json!("blue"));
    }

    #[test]
    fn deeply_nested() {
        let input = "a:\n  b:\n    c:\n      d: deep";
        let v = parse(input).unwrap();
        assert_eq!(v["a"]["b"]["c"]["d"], json!("deep"));
    }

    #[test]
    fn error_display() {
        let e = YamlError::UndefinedAlias("foo".into());
        assert_eq!(format!("{e}"), "undefined alias: foo");
    }

    #[test]
    fn hex_integer() {
        assert_eq!(parse_scalar("0xFF"), json!(255));
    }

    #[test]
    fn boolean_variants() {
        assert_eq!(parse_scalar("True"), json!(true));
        assert_eq!(parse_scalar("TRUE"), json!(true));
        assert_eq!(parse_scalar("False"), json!(false));
        assert_eq!(parse_scalar("FALSE"), json!(false));
    }
}
