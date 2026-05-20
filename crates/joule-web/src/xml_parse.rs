//! XML parser — SAX-style event parser and simple tree builder.
//!
//! Handles elements, attributes, text, CDATA, comments, namespaces,
//! entity references (&amp; etc), and XPath-lite queries.

use std::fmt;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum XmlParseError {
    #[error("unexpected end of input at position {0}")]
    UnexpectedEof(usize),
    #[error("expected '{expected}' at position {pos}")]
    Expected { expected: String, pos: usize },
    #[error("mismatched tag: expected '</{expected}>', found '</{found}>'")]
    MismatchedTag { expected: String, found: String },
    #[error("invalid XML at position {pos}: {msg}")]
    Invalid { pos: usize, msg: String },
    #[error("unknown entity reference '&{0};'")]
    UnknownEntity(String),
}

// ── SAX Events ──────────────────────────────────────────────────

/// SAX-style XML event.
#[derive(Debug, Clone, PartialEq)]
pub enum XmlEvent {
    /// Start of an element: `<tag attr="val">`.
    StartElement {
        name: String,
        namespace: Option<String>,
        attributes: Vec<(String, String)>,
    },
    /// End of an element: `</tag>`.
    EndElement { name: String },
    /// Text content.
    Text(String),
    /// CDATA section: `<![CDATA[...]]>`.
    CData(String),
    /// Comment: `<!-- ... -->`.
    Comment(String),
    /// Processing instruction: `<?target data?>`.
    ProcessingInstruction { target: String, data: String },
    /// XML declaration: `<?xml version="1.0"?>`.
    Declaration {
        version: String,
        encoding: Option<String>,
    },
}

// ── Tree Node ───────────────────────────────────────────────────

/// A node in the XML tree.
#[derive(Debug, Clone, PartialEq)]
pub enum XmlNode {
    /// An element with optional namespace, attributes, and children.
    Element {
        name: String,
        namespace: Option<String>,
        attributes: Vec<(String, String)>,
        children: Vec<XmlNode>,
    },
    /// Text content.
    Text(String),
    /// CDATA section.
    CData(String),
    /// Comment.
    Comment(String),
}

impl XmlNode {
    /// Get the element name, or None for non-element nodes.
    pub fn name(&self) -> Option<&str> {
        match self {
            Self::Element { name, .. } => Some(name.as_str()),
            _ => None,
        }
    }

    /// Get an attribute value by name.
    pub fn attr(&self, attr_name: &str) -> Option<&str> {
        match self {
            Self::Element { attributes, .. } => {
                attributes.iter()
                    .find(|(k, _)| k == attr_name)
                    .map(|(_, v)| v.as_str())
            }
            _ => None,
        }
    }

    /// Get all child elements.
    pub fn children(&self) -> &[XmlNode] {
        match self {
            Self::Element { children, .. } => children,
            _ => &[],
        }
    }

    /// Get child elements with a specific tag name.
    pub fn children_by_name(&self, tag: &str) -> Vec<&XmlNode> {
        self.children().iter()
            .filter(|c| c.name() == Some(tag))
            .collect()
    }

    /// Get the text content of this node (concatenated).
    pub fn text_content(&self) -> String {
        match self {
            Self::Element { children, .. } => {
                children.iter().map(|c| c.text_content()).collect()
            }
            Self::Text(t) => t.clone(),
            Self::CData(t) => t.clone(),
            Self::Comment(_) => String::new(),
        }
    }

    /// Find the first child element with the given name.
    pub fn first_child(&self, tag: &str) -> Option<&XmlNode> {
        self.children().iter().find(|c| c.name() == Some(tag))
    }

    /// XPath-lite query: supports `/root/child/grandchild`, `//tag`, `tag[@attr='val']`.
    pub fn query(&self, xpath: &str) -> Vec<&XmlNode> {
        let parts = parse_xpath(xpath);
        let mut results = vec![self];
        for part in &parts {
            let mut next = Vec::new();
            for node in &results {
                match part {
                    XPathStep::Child(name, pred) => {
                        for child in node.children() {
                            if child.name() == Some(name.as_str()) && matches_predicate(child, pred) {
                                next.push(child);
                            }
                        }
                    }
                    XPathStep::Descendant(name, pred) => {
                        collect_descendants(*node, name, pred, &mut next);
                    }
                }
            }
            results = next;
        }
        results
    }
}

#[derive(Debug)]
enum XPathStep {
    Child(String, Option<XPathPredicate>),
    Descendant(String, Option<XPathPredicate>),
}

#[derive(Debug)]
enum XPathPredicate {
    AttrEquals(String, String),
}

fn parse_xpath(xpath: &str) -> Vec<XPathStep> {
    let mut steps = Vec::new();
    let trimmed = xpath.strip_prefix('/').unwrap_or(xpath);
    let parts: Vec<&str> = split_xpath_parts(trimmed);
    let mut i = 0;
    while i < parts.len() {
        let part = parts[i];
        if part.is_empty() {
            // Double slash -> descendant
            if i + 1 < parts.len() {
                let (name, pred) = parse_step(parts[i + 1]);
                steps.push(XPathStep::Descendant(name, pred));
                i += 2;
            } else {
                i += 1;
            }
        } else {
            let (name, pred) = parse_step(part);
            steps.push(XPathStep::Child(name, pred));
            i += 1;
        }
    }
    steps
}

fn split_xpath_parts(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut in_bracket = false;
    let bytes = s.as_bytes();
    for (i, b) in bytes.iter().enumerate() {
        if *b == b'[' { in_bracket = true; }
        if *b == b']' { in_bracket = false; }
        if *b == b'/' && !in_bracket {
            parts.push(&s[start..i]);
            start = i + 1;
        }
    }
    if start <= s.len() {
        parts.push(&s[start..]);
    }
    parts
}

fn parse_step(s: &str) -> (String, Option<XPathPredicate>) {
    if let Some(bracket_start) = s.find('[') {
        let name = s[..bracket_start].to_string();
        let pred_str = &s[bracket_start + 1..s.len().saturating_sub(1)];
        let pred = parse_predicate(pred_str);
        (name, pred)
    } else {
        (s.to_string(), None)
    }
}

fn parse_predicate(s: &str) -> Option<XPathPredicate> {
    // @attr='value'
    let s = s.trim();
    if s.starts_with('@') {
        let rest = &s[1..];
        if let Some(eq_pos) = rest.find('=') {
            let attr = rest[..eq_pos].trim().to_string();
            let val_raw = rest[eq_pos + 1..].trim();
            let val = val_raw.trim_matches('\'').trim_matches('"').to_string();
            return Some(XPathPredicate::AttrEquals(attr, val));
        }
    }
    None
}

fn matches_predicate(node: &XmlNode, pred: &Option<XPathPredicate>) -> bool {
    match pred {
        None => true,
        Some(XPathPredicate::AttrEquals(attr, val)) => {
            node.attr(attr) == Some(val.as_str())
        }
    }
}

fn collect_descendants<'a>(
    node: &'a XmlNode,
    name: &str,
    pred: &Option<XPathPredicate>,
    results: &mut Vec<&'a XmlNode>,
) {
    for child in node.children() {
        if child.name() == Some(name) && matches_predicate(child, pred) {
            results.push(child);
        }
        collect_descendants(child, name, pred, results);
    }
}

// ── Entity Handling ─────────────────────────────────────────────

fn decode_entities(s: &str) -> Result<String, XmlParseError> {
    let mut out = String::with_capacity(s.len());
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '&' {
            let start = i;
            i += 1;
            let mut entity = String::new();
            while i < chars.len() && chars[i] != ';' {
                entity.push(chars[i]);
                i += 1;
            }
            if i >= chars.len() {
                return Err(XmlParseError::Invalid { pos: start, msg: "unterminated entity".to_string() });
            }
            i += 1; // skip ';'
            let decoded = match entity.as_str() {
                "amp" => '&',
                "lt" => '<',
                "gt" => '>',
                "quot" => '"',
                "apos" => '\'',
                _ if entity.starts_with('#') => {
                    let num_str = &entity[1..];
                    let code_point = if num_str.starts_with('x') || num_str.starts_with('X') {
                        u32::from_str_radix(&num_str[1..], 16).ok()
                    } else {
                        num_str.parse::<u32>().ok()
                    };
                    match code_point.and_then(char::from_u32) {
                        Some(c) => c,
                        None => return Err(XmlParseError::UnknownEntity(entity)),
                    }
                }
                _ => return Err(XmlParseError::UnknownEntity(entity)),
            };
            out.push(decoded);
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    Ok(out)
}

// ── SAX Parser ──────────────────────────────────────────────────

/// Parse XML into a stream of SAX events.
pub fn parse_events(input: &str) -> Result<Vec<XmlEvent>, XmlParseError> {
    let mut events = Vec::new();
    let mut parser = XmlParser::new(input);
    while !parser.is_eof() {
        parser.skip_whitespace();
        if parser.is_eof() { break; }
        if parser.peek() == Some('<') {
            parser.advance();
            if parser.peek() == Some('!') {
                parser.advance();
                if parser.starts_with("--") {
                    parser.advance();
                    parser.advance();
                    let comment = parser.read_until("-->")?;
                    events.push(XmlEvent::Comment(comment));
                } else if parser.starts_with("[CDATA[") {
                    for _ in 0..7 { parser.advance(); }
                    let cdata = parser.read_until("]]>")?;
                    events.push(XmlEvent::CData(cdata));
                } else {
                    // DOCTYPE or other — skip
                    let _ = parser.read_until(">")?;
                }
            } else if parser.peek() == Some('?') {
                parser.advance();
                let target = parser.read_name();
                parser.skip_whitespace();
                let data = parser.read_until("?>")?;
                if target == "xml" {
                    let version = extract_attr_from_pi(&data, "version").unwrap_or_else(|| "1.0".to_string());
                    let encoding = extract_attr_from_pi(&data, "encoding");
                    events.push(XmlEvent::Declaration { version, encoding });
                } else {
                    events.push(XmlEvent::ProcessingInstruction { target, data });
                }
            } else if parser.peek() == Some('/') {
                parser.advance();
                let name = parser.read_name();
                parser.skip_whitespace();
                parser.expect('>')?;
                events.push(XmlEvent::EndElement { name });
            } else {
                let name = parser.read_name();
                let (ns, local) = split_ns(&name);
                let mut attrs = Vec::new();
                loop {
                    parser.skip_whitespace();
                    if parser.peek() == Some('/') {
                        parser.advance();
                        parser.expect('>')?;
                        events.push(XmlEvent::StartElement {
                            name: name.clone(),
                            namespace: ns.clone(),
                            attributes: attrs,
                        });
                        events.push(XmlEvent::EndElement { name });
                        break;
                    }
                    if parser.peek() == Some('>') {
                        parser.advance();
                        events.push(XmlEvent::StartElement {
                            name,
                            namespace: ns,
                            attributes: attrs,
                        });
                        break;
                    }
                    if parser.is_eof() {
                        return Err(XmlParseError::UnexpectedEof(parser.pos));
                    }
                    let attr_name = parser.read_name();
                    parser.skip_whitespace();
                    parser.expect('=')?;
                    parser.skip_whitespace();
                    let attr_val = parser.read_quoted_string()?;
                    attrs.push((attr_name, attr_val));
                }
            }
        } else {
            let text = parser.read_text()?;
            if !text.trim().is_empty() {
                events.push(XmlEvent::Text(text));
            }
        }
    }
    Ok(events)
}

fn extract_attr_from_pi(data: &str, attr: &str) -> Option<String> {
    let needle = format!("{}=", attr);
    let pos = data.find(&needle)?;
    let rest = &data[pos + needle.len()..];
    let rest = rest.trim_start();
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' { return None; }
    let end = rest[1..].find(quote)?;
    Some(rest[1..1 + end].to_string())
}

fn split_ns(name: &str) -> (Option<String>, String) {
    if let Some(colon_pos) = name.find(':') {
        (Some(name[..colon_pos].to_string()), name[colon_pos + 1..].to_string())
    } else {
        (None, name.to_string())
    }
}

struct XmlParser {
    chars: Vec<char>,
    pos: usize,
}

impl XmlParser {
    fn new(input: &str) -> Self {
        Self { chars: input.chars().collect(), pos: 0 }
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied();
        if c.is_some() { self.pos += 1; }
        c
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_whitespace() { self.advance(); } else { break; }
        }
    }

    fn starts_with(&self, s: &str) -> bool {
        let remaining: String = self.chars[self.pos..].iter().take(s.len()).collect();
        remaining == s
    }

    fn expect(&mut self, expected: char) -> Result<(), XmlParseError> {
        match self.advance() {
            Some(c) if c == expected => Ok(()),
            _ => Err(XmlParseError::Expected {
                expected: expected.to_string(),
                pos: self.pos,
            }),
        }
    }

    fn read_name(&mut self) -> String {
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ':' {
                name.push(c);
                self.advance();
            } else {
                break;
            }
        }
        name
    }

    fn read_quoted_string(&mut self) -> Result<String, XmlParseError> {
        let quote = self.advance().ok_or(XmlParseError::UnexpectedEof(self.pos))?;
        if quote != '"' && quote != '\'' {
            return Err(XmlParseError::Expected { expected: "quote".to_string(), pos: self.pos });
        }
        let mut val = String::new();
        while let Some(c) = self.advance() {
            if c == quote { break; }
            val.push(c);
        }
        decode_entities(&val)
    }

    fn read_until(&mut self, end: &str) -> Result<String, XmlParseError> {
        let end_chars: Vec<char> = end.chars().collect();
        let mut buf = String::new();
        loop {
            if self.is_eof() {
                return Err(XmlParseError::UnexpectedEof(self.pos));
            }
            if self.starts_with(end) {
                for _ in 0..end_chars.len() { self.advance(); }
                return Ok(buf);
            }
            buf.push(self.advance().unwrap());
        }
    }

    fn read_text(&mut self) -> Result<String, XmlParseError> {
        let mut text = String::new();
        while let Some(c) = self.peek() {
            if c == '<' { break; }
            text.push(c);
            self.advance();
        }
        decode_entities(&text)
    }
}

// ── Tree Builder ────────────────────────────────────────────────

/// Parse XML into a tree of `XmlNode`.
pub fn parse_tree(input: &str) -> Result<XmlNode, XmlParseError> {
    let events = parse_events(input)?;
    build_tree(&events)
}

fn build_tree(events: &[XmlEvent]) -> Result<XmlNode, XmlParseError> {
    let mut stack: Vec<XmlNode> = vec![
        XmlNode::Element {
            name: "__root__".to_string(),
            namespace: None,
            attributes: Vec::new(),
            children: Vec::new(),
        }
    ];

    for event in events {
        match event {
            XmlEvent::StartElement { name, namespace, attributes } => {
                stack.push(XmlNode::Element {
                    name: name.clone(),
                    namespace: namespace.clone(),
                    attributes: attributes.clone(),
                    children: Vec::new(),
                });
            }
            XmlEvent::EndElement { name } => {
                let node = stack.pop().ok_or_else(|| XmlParseError::MismatchedTag {
                    expected: name.clone(),
                    found: String::new(),
                })?;
                if node.name() != Some(name.as_str()) {
                    return Err(XmlParseError::MismatchedTag {
                        expected: name.clone(),
                        found: node.name().unwrap_or("").to_string(),
                    });
                }
                if let Some(XmlNode::Element { children, .. }) = stack.last_mut() {
                    children.push(node);
                }
            }
            XmlEvent::Text(t) => {
                if let Some(XmlNode::Element { children, .. }) = stack.last_mut() {
                    children.push(XmlNode::Text(t.clone()));
                }
            }
            XmlEvent::CData(t) => {
                if let Some(XmlNode::Element { children, .. }) = stack.last_mut() {
                    children.push(XmlNode::CData(t.clone()));
                }
            }
            XmlEvent::Comment(t) => {
                if let Some(XmlNode::Element { children, .. }) = stack.last_mut() {
                    children.push(XmlNode::Comment(t.clone()));
                }
            }
            XmlEvent::Declaration { .. } | XmlEvent::ProcessingInstruction { .. } => {}
        }
    }

    let root = stack.pop().ok_or(XmlParseError::UnexpectedEof(0))?;
    match root {
        XmlNode::Element { mut children, .. } => {
            if children.len() == 1 {
                Ok(children.remove(0))
            } else {
                // Multiple root-level nodes — wrap in a document element
                Ok(XmlNode::Element {
                    name: "__document__".to_string(),
                    namespace: None,
                    attributes: Vec::new(),
                    children,
                })
            }
        }
        other => Ok(other),
    }
}

// ── Serialization ───────────────────────────────────────────────

/// Serialize an `XmlNode` tree to an XML string.
pub fn serialize(node: &XmlNode) -> String {
    let mut out = String::new();
    serialize_node(node, &mut out, 0);
    out
}

fn serialize_node(node: &XmlNode, out: &mut String, depth: usize) {
    let indent = "  ".repeat(depth);
    match node {
        XmlNode::Element { name, attributes, children, .. } => {
            out.push_str(&indent);
            out.push('<');
            out.push_str(name);
            for (k, v) in attributes {
                out.push(' ');
                out.push_str(k);
                out.push_str("=\"");
                out.push_str(&escape_attr(v));
                out.push('"');
            }
            if children.is_empty() {
                out.push_str("/>\n");
            } else if children.len() == 1 && matches!(&children[0], XmlNode::Text(_)) {
                // Inline single text child — no extra whitespace.
                out.push('>');
                if let XmlNode::Text(t) = &children[0] {
                    out.push_str(&escape_text(t));
                }
                out.push_str("</");
                out.push_str(name);
                out.push_str(">\n");
            } else {
                out.push_str(">\n");
                for child in children {
                    serialize_node(child, out, depth + 1);
                }
                out.push_str(&indent);
                out.push_str("</");
                out.push_str(name);
                out.push_str(">\n");
            }
        }
        XmlNode::Text(t) => {
            out.push_str(&indent);
            out.push_str(&escape_text(t));
            out.push('\n');
        }
        XmlNode::CData(t) => {
            out.push_str(&indent);
            out.push_str("<![CDATA[");
            out.push_str(t);
            out.push_str("]]>\n");
        }
        XmlNode::Comment(t) => {
            out.push_str(&indent);
            out.push_str("<!-- ");
            out.push_str(t);
            out.push_str(" -->\n");
        }
    }
}

fn escape_text(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
}

fn escape_attr(s: &str) -> String {
    s.replace('&', "&amp;")
     .replace('<', "&lt;")
     .replace('>', "&gt;")
     .replace('"', "&quot;")
}

impl fmt::Display for XmlNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", serialize(self))
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_element() {
        let xml = "<root><child>hello</child></root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.name(), Some("root"));
        assert_eq!(tree.children().len(), 1);
        assert_eq!(tree.first_child("child").unwrap().text_content(), "hello");
    }

    #[test]
    fn test_attributes() {
        let xml = "<div class=\"main\" id=\"top\">text</div>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.attr("class"), Some("main"));
        assert_eq!(tree.attr("id"), Some("top"));
    }

    #[test]
    fn test_nested() {
        let xml = "<a><b><c>deep</c></b></a>";
        let tree = parse_tree(xml).unwrap();
        let c = tree.first_child("b").unwrap().first_child("c").unwrap();
        assert_eq!(c.text_content(), "deep");
    }

    #[test]
    fn test_self_closing() {
        let xml = "<root><br/><hr /></root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.children().len(), 2);
        assert_eq!(tree.children()[0].name(), Some("br"));
        assert_eq!(tree.children()[1].name(), Some("hr"));
    }

    #[test]
    fn test_cdata() {
        let xml = "<root><![CDATA[<not xml>]]></root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.text_content(), "<not xml>");
    }

    #[test]
    fn test_comment() {
        let xml = "<root><!-- a comment --><child/></root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.children().len(), 2);
        match &tree.children()[0] {
            XmlNode::Comment(c) => assert_eq!(c, " a comment "),
            _ => panic!("expected comment"),
        }
    }

    #[test]
    fn test_entity_references() {
        let xml = "<root>a &amp; b &lt; c &gt; d</root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.text_content(), "a & b < c > d");
    }

    #[test]
    fn test_numeric_entity() {
        let xml = "<root>&#65;&#x42;</root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.text_content(), "AB");
    }

    #[test]
    fn test_namespace_prefix() {
        let xml = "<ns:root xmlns:ns=\"http://example.com\"><ns:child>val</ns:child></ns:root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.name(), Some("ns:root"));
    }

    #[test]
    fn test_mismatched_tag_error() {
        let xml = "<a></b>";
        let result = parse_tree(xml);
        assert!(result.is_err());
    }

    #[test]
    fn test_sax_events() {
        let xml = "<root><child attr=\"v\">text</child></root>";
        let events = parse_events(xml).unwrap();
        assert!(events.len() >= 4);
        match &events[0] {
            XmlEvent::StartElement { name, .. } => assert_eq!(name, "root"),
            _ => panic!("expected start element"),
        }
    }

    #[test]
    fn test_xpath_child() {
        let xml = "<root><a><b>1</b></a><a><b>2</b></a></root>";
        let tree = parse_tree(xml).unwrap();
        let results = tree.query("/a/b");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].text_content(), "1");
        assert_eq!(results[1].text_content(), "2");
    }

    #[test]
    fn test_xpath_descendant() {
        let xml = "<root><a><b><c>deep</c></b></a></root>";
        let tree = parse_tree(xml).unwrap();
        let results = tree.query("//c");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text_content(), "deep");
    }

    #[test]
    fn test_xpath_with_predicate() {
        let xml = "<root><item id=\"a\">1</item><item id=\"b\">2</item></root>";
        let tree = parse_tree(xml).unwrap();
        let results = tree.query("/item[@id='b']");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].text_content(), "2");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let xml = "<root><child attr=\"val\">text</child></root>";
        let tree = parse_tree(xml).unwrap();
        let output = serialize(&tree);
        let tree2 = parse_tree(&output).unwrap();
        assert_eq!(tree2.name(), Some("root"));
        assert_eq!(tree2.first_child("child").unwrap().text_content(), "text");
    }

    #[test]
    fn test_declaration() {
        let xml = "<?xml version=\"1.0\" encoding=\"UTF-8\"?><root/>";
        let events = parse_events(xml).unwrap();
        match &events[0] {
            XmlEvent::Declaration { version, encoding } => {
                assert_eq!(version, "1.0");
                assert_eq!(encoding.as_deref(), Some("UTF-8"));
            }
            _ => panic!("expected declaration"),
        }
    }

    #[test]
    fn test_processing_instruction() {
        let xml = "<?target data here?><root/>";
        let events = parse_events(xml).unwrap();
        match &events[0] {
            XmlEvent::ProcessingInstruction { target, .. } => {
                assert_eq!(target, "target");
            }
            _ => panic!("expected PI"),
        }
    }

    #[test]
    fn test_attr_with_entities() {
        let xml = "<a href=\"a&amp;b\"/>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.attr("href"), Some("a&b"));
    }

    #[test]
    fn test_children_by_name() {
        let xml = "<root><a/><b/><a/><c/></root>";
        let tree = parse_tree(xml).unwrap();
        assert_eq!(tree.children_by_name("a").len(), 2);
    }

    #[test]
    fn test_empty_element() {
        let xml = "<root></root>";
        let tree = parse_tree(xml).unwrap();
        assert!(tree.children().is_empty());
    }
}
