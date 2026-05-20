//! Simple XML parser and builder.
//!
//! Provides a recursive-descent parser for well-formed XML and a builder API
//! for constructing XML documents programmatically. All operations are
//! pure Rust with no external XML dependencies.

use std::collections::HashMap;
use std::fmt;
use thiserror::Error;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Error)]
pub enum XmlError {
    #[error("unexpected end of input at position {0}")]
    UnexpectedEof(usize),
    #[error("expected '{expected}' at position {pos}")]
    Expected { expected: String, pos: usize },
    #[error("mismatched closing tag: expected '{expected}', found '{found}'")]
    MismatchedTag { expected: String, found: String },
    #[error("invalid XML at position {0}: {1}")]
    Invalid(usize, String),
}

// ── XmlNode ─────────────────────────────────────────────────────

/// A node in an XML document tree.
#[derive(Debug, Clone, PartialEq)]
pub enum XmlNode {
    /// An element with tag name, attributes, and children.
    Element {
        tag: String,
        attrs: HashMap<String, String>,
        children: Vec<XmlNode>,
    },
    /// A text node.
    Text(String),
    /// A comment node (`<!-- ... -->`).
    Comment(String),
    /// A CDATA section (`<![CDATA[...]]>`).
    CData(String),
    /// A processing instruction (`<?target data?>`).
    ProcessingInstruction { target: String, data: String },
}

impl XmlNode {
    /// Returns the tag name if this is an Element.
    pub fn tag(&self) -> Option<&str> {
        match self {
            XmlNode::Element { tag, .. } => Some(tag),
            _ => None,
        }
    }

    /// Returns the value of the named attribute if this is an Element.
    pub fn attr(&self, name: &str) -> Option<&str> {
        match self {
            XmlNode::Element { attrs, .. } => attrs.get(name).map(|s| s.as_str()),
            _ => None,
        }
    }

    /// Returns the children if this is an Element, or an empty slice.
    pub fn children(&self) -> &[XmlNode] {
        match self {
            XmlNode::Element { children, .. } => children,
            _ => &[],
        }
    }

    /// Returns the recursive text content of this node.
    pub fn text_content(&self) -> String {
        match self {
            XmlNode::Text(t) => t.clone(),
            XmlNode::CData(t) => t.clone(),
            XmlNode::Element { children, .. } => {
                children.iter().map(|c| c.text_content()).collect()
            }
            XmlNode::Comment(_) | XmlNode::ProcessingInstruction { .. } => String::new(),
        }
    }

    /// Find the first descendant element with the given tag name.
    pub fn find_first(&self, tag_name: &str) -> Option<&XmlNode> {
        match self {
            XmlNode::Element { tag, children, .. } => {
                if tag == tag_name {
                    return Some(self);
                }
                for child in children {
                    if let Some(found) = child.find_first(tag_name) {
                        return Some(found);
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Find all descendant elements with the given tag name.
    pub fn find_all<'a>(&'a self, tag_name: &str) -> Vec<&'a XmlNode> {
        let mut results = Vec::new();
        self.find_all_inner(tag_name, &mut results);
        results
    }

    fn find_all_inner<'a>(&'a self, tag_name: &str, results: &mut Vec<&'a XmlNode>) {
        if let XmlNode::Element { tag, children, .. } = self {
            if tag == tag_name {
                results.push(self);
            }
            for child in children {
                child.find_all_inner(tag_name, results);
            }
        }
    }

    /// Serialize this node to an XML string.
    pub fn to_xml_string(&self) -> String {
        let mut buf = String::new();
        self.write_xml(&mut buf, false, 0);
        buf
    }

    /// Serialize this node to a pretty-printed XML string.
    pub fn to_string_pretty(&self, indent: usize) -> String {
        let mut buf = String::new();
        self.write_xml(&mut buf, true, indent);
        buf
    }

    fn write_xml(&self, buf: &mut String, pretty: bool, depth: usize) {
        let indent_str = if pretty {
            " ".repeat(depth)
        } else {
            String::new()
        };
        let newline = if pretty { "\n" } else { "" };

        match self {
            XmlNode::Element {
                tag,
                attrs,
                children,
            } => {
                buf.push_str(&indent_str);
                buf.push('<');
                buf.push_str(tag);
                let mut attr_keys: Vec<&String> = attrs.keys().collect();
                attr_keys.sort();
                for k in attr_keys {
                    let v = &attrs[k];
                    buf.push(' ');
                    buf.push_str(k);
                    buf.push_str("=\"");
                    buf.push_str(&escape_attr(v));
                    buf.push('"');
                }
                if children.is_empty() {
                    buf.push_str("/>");
                    buf.push_str(newline);
                } else {
                    buf.push('>');
                    let all_text = children
                        .iter()
                        .all(|c| matches!(c, XmlNode::Text(_) | XmlNode::CData(_)));
                    if all_text && !pretty {
                        for child in children {
                            child.write_xml(buf, false, 0);
                        }
                    } else if all_text && children.len() == 1 {
                        for child in children {
                            child.write_xml(buf, false, 0);
                        }
                    } else {
                        buf.push_str(newline);
                        for child in children {
                            child.write_xml(buf, pretty, depth + 2);
                        }
                        buf.push_str(&indent_str);
                    }
                    buf.push_str("</");
                    buf.push_str(tag);
                    buf.push('>');
                    buf.push_str(newline);
                }
            }
            XmlNode::Text(t) => {
                if pretty {
                    buf.push_str(&indent_str);
                }
                buf.push_str(&escape_text(t));
                if pretty {
                    buf.push_str(newline);
                }
            }
            XmlNode::Comment(c) => {
                buf.push_str(&indent_str);
                buf.push_str("<!--");
                buf.push_str(c);
                buf.push_str("-->");
                buf.push_str(newline);
            }
            XmlNode::CData(c) => {
                buf.push_str(&indent_str);
                buf.push_str("<![CDATA[");
                buf.push_str(c);
                buf.push_str("]]>");
                buf.push_str(newline);
            }
            XmlNode::ProcessingInstruction { target, data } => {
                buf.push_str(&indent_str);
                buf.push_str("<?");
                buf.push_str(target);
                if !data.is_empty() {
                    buf.push(' ');
                    buf.push_str(data);
                }
                buf.push_str("?>");
                buf.push_str(newline);
            }
        }
    }
}

impl fmt::Display for XmlNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_xml_string())
    }
}

// ── XmlBuilder ──────────────────────────────────────────────────

/// Builder for constructing XML nodes.
pub struct XmlBuilder {
    tag: String,
    attrs: HashMap<String, String>,
    children: Vec<XmlNode>,
}

impl XmlBuilder {
    /// Create a new builder for an element with the given tag.
    pub fn element(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            attrs: HashMap::new(),
            children: Vec::new(),
        }
    }

    /// Add an attribute.
    pub fn attr(mut self, key: &str, value: &str) -> Self {
        self.attrs.insert(key.to_string(), value.to_string());
        self
    }

    /// Add a child node.
    pub fn child(mut self, node: XmlNode) -> Self {
        self.children.push(node);
        self
    }

    /// Add a text child.
    pub fn text(mut self, s: &str) -> Self {
        self.children.push(XmlNode::Text(s.to_string()));
        self
    }

    /// Build the XmlNode.
    pub fn build(self) -> XmlNode {
        XmlNode::Element {
            tag: self.tag,
            attrs: self.attrs,
            children: self.children,
        }
    }
}

// ── Parser ──────────────────────────────────────────────────────

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

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek() {
            if c.is_ascii_whitespace() {
                self.advance(c.len_utf8());
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, s: &str) -> Result<(), XmlError> {
        if self.remaining().starts_with(s) {
            self.advance(s.len());
            Ok(())
        } else {
            Err(XmlError::Expected {
                expected: s.to_string(),
                pos: self.pos,
            })
        }
    }

    fn parse_name(&mut self) -> Result<String, XmlError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c.is_alphanumeric() || c == '_' || c == '-' || c == '.' || c == ':' {
                self.advance(c.len_utf8());
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err(XmlError::Invalid(self.pos, "expected name".into()));
        }
        Ok(self.input[start..self.pos].to_string())
    }

    fn parse_quoted_value(&mut self) -> Result<String, XmlError> {
        let quote = self.peek().ok_or(XmlError::UnexpectedEof(self.pos))?;
        if quote != '"' && quote != '\'' {
            return Err(XmlError::Expected {
                expected: "quote".into(),
                pos: self.pos,
            });
        }
        self.advance(1);
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == quote {
                let val = self.input[start..self.pos].to_string();
                self.advance(1);
                return Ok(decode_entities(&val));
            }
            self.advance(c.len_utf8());
        }
        Err(XmlError::UnexpectedEof(self.pos))
    }

    fn parse_node(&mut self) -> Result<XmlNode, XmlError> {
        self.skip_whitespace();
        if self.remaining().starts_with("<!--") {
            return self.parse_comment();
        }
        if self.remaining().starts_with("<![CDATA[") {
            return self.parse_cdata();
        }
        if self.remaining().starts_with("<?") {
            return self.parse_pi();
        }
        if self.remaining().starts_with('<') && !self.remaining().starts_with("</") {
            return self.parse_element();
        }
        self.parse_text()
    }

    fn parse_comment(&mut self) -> Result<XmlNode, XmlError> {
        self.expect("<!--")?;
        let start = self.pos;
        loop {
            if self.remaining().starts_with("-->") {
                let content = self.input[start..self.pos].to_string();
                self.advance(3);
                return Ok(XmlNode::Comment(content));
            }
            if self.peek().is_none() {
                return Err(XmlError::UnexpectedEof(self.pos));
            }
            self.advance(self.peek().unwrap().len_utf8());
        }
    }

    fn parse_cdata(&mut self) -> Result<XmlNode, XmlError> {
        self.expect("<![CDATA[")?;
        let start = self.pos;
        loop {
            if self.remaining().starts_with("]]>") {
                let content = self.input[start..self.pos].to_string();
                self.advance(3);
                return Ok(XmlNode::CData(content));
            }
            if self.peek().is_none() {
                return Err(XmlError::UnexpectedEof(self.pos));
            }
            self.advance(self.peek().unwrap().len_utf8());
        }
    }

    fn parse_pi(&mut self) -> Result<XmlNode, XmlError> {
        self.expect("<?")?;
        self.skip_whitespace();
        let target = self.parse_name()?;
        self.skip_whitespace();
        let start = self.pos;
        loop {
            if self.remaining().starts_with("?>") {
                let data = self.input[start..self.pos].trim().to_string();
                self.advance(2);
                return Ok(XmlNode::ProcessingInstruction { target, data });
            }
            if self.peek().is_none() {
                return Err(XmlError::UnexpectedEof(self.pos));
            }
            self.advance(self.peek().unwrap().len_utf8());
        }
    }

    fn parse_element(&mut self) -> Result<XmlNode, XmlError> {
        self.expect("<")?;
        let tag = self.parse_name()?;
        let mut attrs = HashMap::new();

        loop {
            self.skip_whitespace();
            if self.remaining().starts_with("/>") {
                self.advance(2);
                return Ok(XmlNode::Element {
                    tag,
                    attrs,
                    children: Vec::new(),
                });
            }
            if self.remaining().starts_with('>') {
                self.advance(1);
                break;
            }
            if self.peek().is_none() {
                return Err(XmlError::UnexpectedEof(self.pos));
            }
            let key = self.parse_name()?;
            self.skip_whitespace();
            self.expect("=")?;
            self.skip_whitespace();
            let value = self.parse_quoted_value()?;
            attrs.insert(key, value);
        }

        let mut children = Vec::new();
        loop {
            if self.remaining().starts_with("</") {
                self.expect("</")?;
                self.skip_whitespace();
                let close_tag = self.parse_name()?;
                self.skip_whitespace();
                self.expect(">")?;
                if close_tag != tag {
                    return Err(XmlError::MismatchedTag {
                        expected: tag,
                        found: close_tag,
                    });
                }
                return Ok(XmlNode::Element {
                    tag,
                    attrs,
                    children,
                });
            }
            if self.peek().is_none() {
                return Err(XmlError::UnexpectedEof(self.pos));
            }
            children.push(self.parse_node()?);
        }
    }

    fn parse_text(&mut self) -> Result<XmlNode, XmlError> {
        let start = self.pos;
        while let Some(c) = self.peek() {
            if c == '<' {
                break;
            }
            self.advance(c.len_utf8());
        }
        if self.pos == start {
            return Err(XmlError::Invalid(self.pos, "expected text".into()));
        }
        let raw = &self.input[start..self.pos];
        Ok(XmlNode::Text(decode_entities(raw)))
    }
}

// ── Entity encoding/decoding ────────────────────────────────────

fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
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
        .replace('\'', "&apos;")
}

// ── Public API ──────────────────────────────────────────────────

/// Parse well-formed XML into an `XmlNode` tree.
pub fn parse_xml(input: &str) -> Result<XmlNode, XmlError> {
    let mut parser = Parser::new(input.trim());
    parser.skip_whitespace();
    if parser.remaining().starts_with("<?xml") {
        parser.parse_pi()?;
        parser.skip_whitespace();
    }
    parser.parse_node()
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_element() {
        let node = parse_xml("<hello>world</hello>").unwrap();
        assert_eq!(node.tag(), Some("hello"));
        assert_eq!(node.text_content(), "world");
    }

    #[test]
    fn parse_attributes() {
        let node = parse_xml(r#"<div class="main" id="top">text</div>"#).unwrap();
        assert_eq!(node.attr("class"), Some("main"));
        assert_eq!(node.attr("id"), Some("top"));
        assert_eq!(node.text_content(), "text");
    }

    #[test]
    fn parse_nested_elements() {
        let xml = "<root><a><b>deep</b></a></root>";
        let node = parse_xml(xml).unwrap();
        assert_eq!(node.tag(), Some("root"));
        let b = node.find_first("b").unwrap();
        assert_eq!(b.text_content(), "deep");
    }

    #[test]
    fn parse_text_content_recursive() {
        let xml = "<p>Hello <b>world</b>!</p>";
        let node = parse_xml(xml).unwrap();
        assert_eq!(node.text_content(), "Hello world!");
    }

    #[test]
    fn parse_self_closing() {
        let node = parse_xml("<br/>").unwrap();
        assert_eq!(node.tag(), Some("br"));
        assert!(node.children().is_empty());
    }

    #[test]
    fn parse_comment() {
        let xml = "<root><!-- a comment --><child/></root>";
        let node = parse_xml(xml).unwrap();
        let children = node.children();
        assert_eq!(children.len(), 2);
        assert!(matches!(&children[0], XmlNode::Comment(c) if c == " a comment "));
    }

    #[test]
    fn parse_cdata() {
        let xml = "<data><![CDATA[<not>xml</not>]]></data>";
        let node = parse_xml(xml).unwrap();
        assert_eq!(node.text_content(), "<not>xml</not>");
    }

    #[test]
    fn entity_decoding() {
        let xml = "<p>&amp; &lt; &gt; &quot; &apos;</p>";
        let node = parse_xml(xml).unwrap();
        assert_eq!(node.text_content(), "& < > \" '");
    }

    #[test]
    fn roundtrip_parse_serialize() {
        let xml = r#"<root attr="val"><child>text</child></root>"#;
        let node = parse_xml(xml).unwrap();
        let serialized = node.to_xml_string();
        let node2 = parse_xml(&serialized).unwrap();
        assert_eq!(node, node2);
    }

    #[test]
    fn find_first_and_find_all() {
        let xml = "<root><item>1</item><group><item>2</item></group><item>3</item></root>";
        let node = parse_xml(xml).unwrap();
        let first = node.find_first("item").unwrap();
        assert_eq!(first.text_content(), "1");
        let all = node.find_all("item");
        assert_eq!(all.len(), 3);
    }

    #[test]
    fn builder_api() {
        let node = XmlBuilder::element("div")
            .attr("class", "container")
            .child(XmlBuilder::element("span").text("hello").build())
            .build();
        assert_eq!(node.tag(), Some("div"));
        assert_eq!(node.attr("class"), Some("container"));
        let span = node.find_first("span").unwrap();
        assert_eq!(span.text_content(), "hello");
    }

    #[test]
    fn invalid_xml_error() {
        assert!(parse_xml("<open>").is_err());
        assert!(parse_xml("<a></b>").is_err());
    }

    #[test]
    fn pretty_print() {
        let node = XmlBuilder::element("root")
            .child(XmlBuilder::element("child").text("val").build())
            .build();
        let pretty = node.to_string_pretty(0);
        assert!(pretty.contains("  <child>"));
    }

    #[test]
    fn processing_instruction() {
        let xml = "<root><?pi target data?><child/></root>";
        let node = parse_xml(xml).unwrap();
        let children = node.children();
        assert!(
            matches!(&children[0], XmlNode::ProcessingInstruction { target, .. } if target == "pi")
        );
    }
}
