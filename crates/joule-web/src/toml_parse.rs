// SPDX-License-Identifier: MIT
//! TOML parser -- tables, arrays, inline tables, all value types, dotted keys.
//!
//! Pure-Rust TOML v1.0 subset: basic/literal/multiline strings, integers
//! (dec/hex/oct/bin), floats (inf/nan), datetime, inline tables, array of
//! tables, dotted keys, comments. Outputs `serde_json::Value`.

use serde_json::{Map, Value};

// ── Errors ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TomlError {
    UnexpectedChar(char, usize),
    UnexpectedEof,
    InvalidEscape(String),
    InvalidNumber(String),
    DuplicateKey(String),
    ExpectedEquals(usize),
    UnterminatedString,
    InvalidKey(String),
}

impl std::fmt::Display for TomlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedChar(c, pos) => write!(f, "unexpected char '{c}' at position {pos}"),
            Self::UnexpectedEof => write!(f, "unexpected end of input"),
            Self::InvalidEscape(s) => write!(f, "invalid escape sequence: {s}"),
            Self::InvalidNumber(s) => write!(f, "invalid number: {s}"),
            Self::DuplicateKey(k) => write!(f, "duplicate key: {k}"),
            Self::ExpectedEquals(pos) => write!(f, "expected '=' at position {pos}"),
            Self::UnterminatedString => write!(f, "unterminated string"),
            Self::InvalidKey(s) => write!(f, "invalid key: {s}"),
        }
    }
}

// ── Parser internals ────────────────────────────────────────────────────────

struct Parser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn remaining(&self) -> &'a str {
        &self.input[self.pos..]
    }

    fn peek(&self) -> Option<char> {
        self.remaining().chars().next()
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' { self.advance(1); } else { break; }
        }
    }

    fn skip_ws_and_nl(&mut self) {
        while let Some(c) = self.peek() {
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.advance(c.len_utf8());
            } else if c == '#' {
                self.skip_comment();
            } else {
                break;
            }
        }
    }

    fn skip_comment(&mut self) {
        while let Some(c) = self.peek() {
            self.advance(c.len_utf8());
            if c == '\n' { break; }
        }
    }

    fn skip_to_newline(&mut self) {
        self.skip_ws();
        if let Some('#') = self.peek() { self.skip_comment(); }
        else if let Some('\n') = self.peek() { self.advance(1); }
        else if let Some('\r') = self.peek() {
            self.advance(1);
            if let Some('\n') = self.peek() { self.advance(1); }
        }
    }

    // ── Keys ────────────────────────────────────────────────────────────

    fn parse_bare_key(&mut self) -> Result<String, TomlError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                self.advance(1);
            } else {
                break;
            }
        }
        if self.pos == start { return Err(TomlError::InvalidKey("empty bare key".into())); }
        Ok(self.input[start..self.pos].to_string())
    }

    fn parse_key(&mut self) -> Result<String, TomlError> {
        match self.peek() {
            Some('"') => self.parse_basic_string(),
            Some('\'') => self.parse_literal_string(),
            _ => self.parse_bare_key(),
        }
    }

    fn parse_dotted_key(&mut self) -> Result<Vec<String>, TomlError> {
        let mut parts = vec![self.parse_key()?];
        loop {
            self.skip_ws();
            if let Some('.') = self.peek() {
                self.advance(1);
                self.skip_ws();
                parts.push(self.parse_key()?);
            } else {
                break;
            }
        }
        Ok(parts)
    }

    // ── Strings ─────────────────────────────────────────────────────────

    fn parse_basic_string(&mut self) -> Result<String, TomlError> {
        self.advance(1); // skip "
        if self.remaining().starts_with("\"\"") {
            self.advance(2);
            return self.parse_ml_basic();
        }
        let mut out = String::new();
        loop {
            match self.peek() {
                None => return Err(TomlError::UnterminatedString),
                Some('"') => { self.advance(1); return Ok(out); }
                Some('\\') => { self.advance(1); out.push(self.parse_escape()?); }
                Some(c) => { self.advance(c.len_utf8()); out.push(c); }
            }
        }
    }

    fn parse_ml_basic(&mut self) -> Result<String, TomlError> {
        if let Some('\n') = self.peek() { self.advance(1); }
        else if self.remaining().starts_with("\r\n") { self.advance(2); }
        let mut out = String::new();
        loop {
            if self.remaining().starts_with("\"\"\"") { self.advance(3); return Ok(out); }
            match self.peek() {
                None => return Err(TomlError::UnterminatedString),
                Some('\\') => {
                    self.advance(1);
                    if matches!(self.peek(), Some('\n') | Some('\r')) {
                        while matches!(self.peek(), Some('\n') | Some('\r') | Some(' ') | Some('\t')) {
                            self.advance(1);
                        }
                    } else {
                        out.push(self.parse_escape()?);
                    }
                }
                Some(c) => { self.advance(c.len_utf8()); out.push(c); }
            }
        }
    }

    fn parse_literal_string(&mut self) -> Result<String, TomlError> {
        self.advance(1); // skip '
        if self.remaining().starts_with("''") {
            self.advance(2);
            return self.parse_ml_literal();
        }
        let start = self.pos;
        loop {
            match self.peek() {
                None => return Err(TomlError::UnterminatedString),
                Some('\'') => {
                    let s = self.input[start..self.pos].to_string();
                    self.advance(1);
                    return Ok(s);
                }
                Some(c) => self.advance(c.len_utf8()),
            }
        }
    }

    fn parse_ml_literal(&mut self) -> Result<String, TomlError> {
        if let Some('\n') = self.peek() { self.advance(1); }
        else if self.remaining().starts_with("\r\n") { self.advance(2); }
        let start = self.pos;
        loop {
            if self.remaining().starts_with("'''") {
                let s = self.input[start..self.pos].to_string();
                self.advance(3);
                return Ok(s);
            }
            match self.peek() {
                None => return Err(TomlError::UnterminatedString),
                Some(c) => self.advance(c.len_utf8()),
            }
        }
    }

    fn parse_escape(&mut self) -> Result<char, TomlError> {
        match self.peek() {
            Some('b') => { self.advance(1); Ok('\u{0008}') }
            Some('t') => { self.advance(1); Ok('\t') }
            Some('n') => { self.advance(1); Ok('\n') }
            Some('f') => { self.advance(1); Ok('\u{000C}') }
            Some('r') => { self.advance(1); Ok('\r') }
            Some('"') => { self.advance(1); Ok('"') }
            Some('\\') => { self.advance(1); Ok('\\') }
            Some('u') => { self.advance(1); self.parse_unicode(4) }
            Some('U') => { self.advance(1); self.parse_unicode(8) }
            Some(c) => Err(TomlError::InvalidEscape(format!("\\{c}"))),
            None => Err(TomlError::UnexpectedEof),
        }
    }

    fn parse_unicode(&mut self, digits: usize) -> Result<char, TomlError> {
        let start = self.pos;
        for _ in 0..digits {
            if self.peek().map_or(false, |c| c.is_ascii_hexdigit()) { self.advance(1); }
            else { return Err(TomlError::InvalidEscape(format!("unicode at {start}"))); }
        }
        let hex = &self.input[start..self.pos];
        let code = u32::from_str_radix(hex, 16).map_err(|_| TomlError::InvalidEscape(hex.into()))?;
        char::from_u32(code).ok_or_else(|| TomlError::InvalidEscape(hex.into()))
    }

    // ── Values ──────────────────────────────────────────────────────────

    fn parse_value(&mut self) -> Result<Value, TomlError> {
        self.skip_ws();
        match self.peek() {
            Some('"') => Ok(Value::String(self.parse_basic_string()?)),
            Some('\'') => Ok(Value::String(self.parse_literal_string()?)),
            Some('t') if self.remaining().starts_with("true") => {
                self.advance(4); Ok(Value::Bool(true))
            }
            Some('f') if self.remaining().starts_with("false") => {
                self.advance(5); Ok(Value::Bool(false))
            }
            Some('[') => self.parse_array_value(),
            Some('{') => self.parse_inline_table(),
            Some('i') if self.remaining().starts_with("inf") => {
                self.advance(3); Ok(serde_json::json!(f64::INFINITY))
            }
            Some('+') if self.remaining().starts_with("+inf") => {
                self.advance(4); Ok(serde_json::json!(f64::INFINITY))
            }
            Some('-') if self.remaining().starts_with("-inf") => {
                self.advance(4); Ok(serde_json::json!(f64::NEG_INFINITY))
            }
            Some('n') if self.remaining().starts_with("nan") => {
                self.advance(3); Ok(Value::Null) // JSON has no NaN, use null
            }
            Some('+') if self.remaining().starts_with("+nan") => {
                self.advance(4); Ok(Value::Null)
            }
            Some('-') if self.remaining().starts_with("-nan") => {
                self.advance(4); Ok(Value::Null)
            }
            Some(c) if c.is_ascii_digit() || c == '+' || c == '-' => self.parse_number_or_dt(),
            Some(c) => Err(TomlError::UnexpectedChar(c, self.pos)),
            None => Err(TomlError::UnexpectedEof),
        }
    }

    fn parse_number_or_dt(&mut self) -> Result<Value, TomlError> {
        let start = self.pos;
        if matches!(self.peek(), Some('+') | Some('-')) { self.advance(1); }

        // 0x / 0o / 0b
        if self.remaining().starts_with("0x") || self.remaining().starts_with("0X") {
            self.advance(2);
            let hs = self.pos;
            while self.peek().map_or(false, |c| c.is_ascii_hexdigit() || c == '_') { self.advance(1); }
            let clean: String = self.input[hs..self.pos].chars().filter(|c| *c != '_').collect();
            let sign: i64 = if self.input[start..].starts_with('-') { -1 } else { 1 };
            let val = i64::from_str_radix(&clean, 16).map_err(|_| TomlError::InvalidNumber(self.input[start..self.pos].into()))?;
            return Ok(serde_json::json!(sign * val));
        }
        if self.remaining().starts_with("0o") || self.remaining().starts_with("0O") {
            self.advance(2);
            let os = self.pos;
            while self.peek().map_or(false, |c| ('0'..='7').contains(&c) || c == '_') { self.advance(1); }
            let clean: String = self.input[os..self.pos].chars().filter(|c| *c != '_').collect();
            let sign: i64 = if self.input[start..].starts_with('-') { -1 } else { 1 };
            let val = i64::from_str_radix(&clean, 8).map_err(|_| TomlError::InvalidNumber(self.input[start..self.pos].into()))?;
            return Ok(serde_json::json!(sign * val));
        }
        if self.remaining().starts_with("0b") || self.remaining().starts_with("0B") {
            self.advance(2);
            let bs = self.pos;
            while self.peek().map_or(false, |c| c == '0' || c == '1' || c == '_') { self.advance(1); }
            let clean: String = self.input[bs..self.pos].chars().filter(|c| *c != '_').collect();
            let sign: i64 = if self.input[start..].starts_with('-') { -1 } else { 1 };
            let val = i64::from_str_radix(&clean, 2).map_err(|_| TomlError::InvalidNumber(self.input[start..self.pos].into()))?;
            return Ok(serde_json::json!(sign * val));
        }

        let mut is_float = false;
        let mut could_be_dt = false;
        while let Some(c) = self.peek() {
            match c {
                '0'..='9' | '_' => self.advance(1),
                '.' => { is_float = true; self.advance(1); }
                'e' | 'E' => {
                    is_float = true;
                    self.advance(1);
                    if matches!(self.peek(), Some('+') | Some('-')) { self.advance(1); }
                }
                '-' if !is_float && (self.pos - start == 4 || self.pos - start == 5) => {
                    could_be_dt = true; break;
                }
                'T' | 't' | ':' | 'Z' | 'z' => { could_be_dt = true; break; }
                _ => break,
            }
        }

        if could_be_dt {
            while self.peek().map_or(false, |c| c.is_ascii_digit() || "T:Zz.+-t".contains(c)) {
                self.advance(1);
            }
            return Ok(Value::String(self.input[start..self.pos].to_string()));
        }

        let raw = &self.input[start..self.pos];
        let clean: String = raw.chars().filter(|c| *c != '_').collect();
        if is_float {
            let val: f64 = clean.parse().map_err(|_| TomlError::InvalidNumber(raw.into()))?;
            Ok(serde_json::json!(val))
        } else {
            let val: i64 = clean.parse().map_err(|_| TomlError::InvalidNumber(raw.into()))?;
            Ok(serde_json::json!(val))
        }
    }

    fn parse_array_value(&mut self) -> Result<Value, TomlError> {
        self.advance(1); // [
        let mut items = Vec::new();
        loop {
            self.skip_ws_and_nl();
            if let Some(']') = self.peek() { self.advance(1); return Ok(Value::Array(items)); }
            items.push(self.parse_value()?);
            self.skip_ws_and_nl();
            if let Some(',') = self.peek() { self.advance(1); }
        }
    }

    fn parse_inline_table(&mut self) -> Result<Value, TomlError> {
        self.advance(1); // {
        let mut map = Map::new();
        loop {
            self.skip_ws();
            if let Some('}') = self.peek() { self.advance(1); return Ok(Value::Object(map)); }
            let keys = self.parse_dotted_key()?;
            self.skip_ws();
            match self.peek() {
                Some('=') => self.advance(1),
                _ => return Err(TomlError::ExpectedEquals(self.pos)),
            }
            let val = self.parse_value()?;
            insert_dotted(&mut map, &keys, val)?;
            self.skip_ws();
            if let Some(',') = self.peek() { self.advance(1); }
        }
    }
}

// ── Table navigation ────────────────────────────────────────────────────────

fn ensure_table(root: &mut Map<String, Value>, path: &[String]) {
    let mut cur = root;
    for key in path {
        cur = cur
            .entry(key.clone())
            .or_insert_with(|| Value::Object(Map::new()))
            .as_object_mut()
            .expect("ensure_table: expected object");
    }
}

fn ensure_aot(root: &mut Map<String, Value>, path: &[String]) {
    if path.is_empty() { return; }
    let mut cur = root;
    // Navigate to parent tables
    for key in &path[..path.len() - 1] {
        let val = cur
            .entry(key.clone())
            .or_insert_with(|| Value::Object(Map::new()));
        match val {
            Value::Object(m) => cur = m,
            Value::Array(arr) => {
                if let Some(Value::Object(m)) = arr.last_mut() {
                    cur = m;
                } else {
                    return;
                }
            }
            _ => return,
        }
    }
    // Handle the final key as array-of-tables
    let last_key = &path[path.len() - 1];
    let entry = cur
        .entry(last_key.clone())
        .or_insert_with(|| Value::Array(Vec::new()));
    if let Value::Array(arr) = entry {
        arr.push(Value::Object(Map::new()));
    }
}

fn get_table_mut<'a>(root: &'a mut Map<String, Value>, path: &[String]) -> &'a mut Map<String, Value> {
    if path.is_empty() {
        return root;
    }
    // Recursive approach to satisfy the borrow checker
    get_table_step(root, path, 0)
}

fn get_table_step<'a>(cur: &'a mut Map<String, Value>, path: &[String], idx: usize) -> &'a mut Map<String, Value> {
    if idx >= path.len() {
        return cur;
    }
    let key = &path[idx];
    // Check if we can descend before borrowing through entry
    let should_descend = match cur.get(key) {
        Some(Value::Object(_)) => true,
        Some(Value::Array(arr)) => matches!(arr.last(), Some(Value::Object(_))),
        _ => false,
    };
    if !should_descend && !cur.contains_key(key) {
        cur.insert(key.clone(), Value::Object(Map::new()));
    }
    if !should_descend && !matches!(cur.get(key), Some(Value::Object(_)) | Some(Value::Array(_))) {
        return cur;
    }
    let val = cur.get_mut(key).unwrap();
    match val {
        Value::Object(m) => get_table_step(m, path, idx + 1),
        Value::Array(arr) => {
            if let Some(Value::Object(m)) = arr.last_mut() {
                get_table_step(m, path, idx + 1)
            } else {
                // This shouldn't happen due to the check above, but return root-level
                panic!("get_table_step: unreachable");
            }
        }
        _ => panic!("get_table_step: unreachable"),
    }
}

fn insert_dotted(table: &mut Map<String, Value>, keys: &[String], val: Value) -> Result<(), TomlError> {
    if keys.len() == 1 {
        table.insert(keys[0].clone(), val);
        Ok(())
    } else {
        if !table.contains_key(&keys[0]) {
            table.insert(keys[0].clone(), Value::Object(Map::new()));
        }
        match table.get_mut(&keys[0]) {
            Some(Value::Object(m)) => insert_dotted(m, &keys[1..], val),
            _ => Err(TomlError::DuplicateKey(keys[0].clone())),
        }
    }
}

// ── Public API ──────────────────────────────────────────────────────────────

/// Parse a TOML string into a `serde_json::Value::Object`.
pub fn parse(input: &str) -> Result<Value, TomlError> {
    let mut parser = Parser::new(input);
    let mut root = Map::new();
    let mut current_path: Vec<String> = Vec::new();

    loop {
        parser.skip_ws_and_nl();
        if parser.peek().is_none() { break; }

        match parser.peek() {
            Some('[') => {
                parser.advance(1);
                let is_aot = if let Some('[') = parser.peek() { parser.advance(1); true } else { false };
                parser.skip_ws();
                let path = parser.parse_dotted_key()?;
                parser.skip_ws();
                if is_aot {
                    if let Some(']') = parser.peek() { parser.advance(1); }
                    if let Some(']') = parser.peek() { parser.advance(1); }
                    ensure_aot(&mut root, &path);
                } else {
                    if let Some(']') = parser.peek() { parser.advance(1); }
                    ensure_table(&mut root, &path);
                }
                current_path = path;
                parser.skip_to_newline();
            }
            Some('#') => { parser.skip_comment(); }
            Some(_) => {
                let keys = parser.parse_dotted_key()?;
                parser.skip_ws();
                match parser.peek() {
                    Some('=') => parser.advance(1),
                    _ => return Err(TomlError::ExpectedEquals(parser.pos)),
                }
                let val = parser.parse_value()?;
                parser.skip_to_newline();
                let table = get_table_mut(&mut root, &current_path);
                insert_dotted(table, &keys, val)?;
            }
            None => break,
        }
    }
    Ok(Value::Object(root))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn basic_kv() {
        let v = parse("name = \"TOML\"\nversion = 1").unwrap();
        assert_eq!(v["name"], json!("TOML"));
        assert_eq!(v["version"], json!(1));
    }

    #[test]
    fn booleans() {
        let v = parse("a = true\nb = false").unwrap();
        assert_eq!(v["a"], json!(true));
        assert_eq!(v["b"], json!(false));
    }

    #[test]
    fn integer_bases() {
        let v = parse("dec = 42\nhex = 0xDEAD\noct = 0o755\nbin = 0b11010110").unwrap();
        assert_eq!(v["dec"], json!(42));
        assert_eq!(v["hex"], json!(0xDEAD));
        assert_eq!(v["oct"], json!(0o755));
        assert_eq!(v["bin"], json!(0b11010110));
    }

    #[test]
    fn float_values() {
        let v = parse("pi = 3.14\nsci = 5e+22").unwrap();
        let pi = v["pi"].as_f64().unwrap();
        assert!((pi - 3.14).abs() < 0.001);
    }

    #[test]
    fn string_types() {
        let input = "basic = \"hello\\nworld\"\nliteral = 'no\\\\escape'";
        let v = parse(input).unwrap();
        assert_eq!(v["basic"], json!("hello\nworld"));
        assert_eq!(v["literal"], json!("no\\\\escape"));
    }

    #[test]
    fn multiline_string() {
        let input = "ml = \"\"\"\nhello\nworld\"\"\"";
        let v = parse(input).unwrap();
        assert_eq!(v["ml"], json!("hello\nworld"));
    }

    #[test]
    fn table_section() {
        let v = parse("[server]\nhost = \"localhost\"\nport = 8080").unwrap();
        assert_eq!(v["server"]["host"], json!("localhost"));
        assert_eq!(v["server"]["port"], json!(8080));
    }

    #[test]
    fn inline_table() {
        let v = parse("point = {x = 1, y = 2}").unwrap();
        assert_eq!(v["point"]["x"], json!(1));
        assert_eq!(v["point"]["y"], json!(2));
    }

    #[test]
    fn array_value() {
        let v = parse("ports = [8001, 8001, 8002]").unwrap();
        assert_eq!(v["ports"], json!([8001, 8001, 8002]));
    }

    #[test]
    fn array_of_tables() {
        let input = "[[products]]\nname = \"Hammer\"\n\n[[products]]\nname = \"Nail\"";
        let v = parse(input).unwrap();
        let products = v["products"].as_array().unwrap();
        assert_eq!(products.len(), 2);
        assert_eq!(products[0]["name"], json!("Hammer"));
        assert_eq!(products[1]["name"], json!("Nail"));
    }

    #[test]
    fn dotted_keys() {
        let v = parse("fruit.apple.color = \"red\"").unwrap();
        assert_eq!(v["fruit"]["apple"]["color"], json!("red"));
    }

    #[test]
    fn comments_ignored() {
        let v = parse("# comment\nkey = \"value\" # inline").unwrap();
        assert_eq!(v["key"], json!("value"));
    }

    #[test]
    fn datetime() {
        let v = parse("dt = 1979-05-27T07:32:00Z").unwrap();
        let dt = v["dt"].as_str().unwrap();
        assert!(dt.starts_with("1979-05-27"));
    }

    #[test]
    fn underscore_in_numbers() {
        let v = parse("big = 1_000_000").unwrap();
        assert_eq!(v["big"], json!(1_000_000));
    }

    #[test]
    fn nan_maps_to_null() {
        let v = parse("x = nan").unwrap();
        assert_eq!(v["x"], json!(null));
    }

    #[test]
    fn negative_integer() {
        let v = parse("x = -42").unwrap();
        assert_eq!(v["x"], json!(-42));
    }

    #[test]
    fn positive_sign_integer() {
        let v = parse("x = +99").unwrap();
        assert_eq!(v["x"], json!(99));
    }

    #[test]
    fn empty_input() {
        let v = parse("").unwrap();
        assert_eq!(v, json!({}));
    }

    #[test]
    fn nested_tables() {
        let input = "[a.b]\nc = 1\n[a.d]\ne = 2";
        let v = parse(input).unwrap();
        assert_eq!(v["a"]["b"]["c"], json!(1));
        assert_eq!(v["a"]["d"]["e"], json!(2));
    }

    #[test]
    fn mixed_array_types() {
        let v = parse("arr = [1, \"two\", true]").unwrap();
        assert_eq!(v["arr"], json!([1, "two", true]));
    }

    #[test]
    fn multiline_array() {
        let input = "arr = [\n  1,\n  2,\n  3,\n]";
        let v = parse(input).unwrap();
        assert_eq!(v["arr"], json!([1, 2, 3]));
    }

    #[test]
    fn escape_sequences() {
        let v = parse("s = \"tab\\there\"").unwrap();
        assert_eq!(v["s"], json!("tab\there"));
    }

    #[test]
    fn unicode_escape() {
        let v = parse("s = \"\\u0041\"").unwrap();
        assert_eq!(v["s"], json!("A"));
    }

    #[test]
    fn error_display() {
        let e = TomlError::UnexpectedChar('x', 5);
        assert_eq!(format!("{e}"), "unexpected char 'x' at position 5");
    }
}
