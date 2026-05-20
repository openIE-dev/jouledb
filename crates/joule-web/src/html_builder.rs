//! HTML document builder — type-safe HTML construction with fluent API,
//! attribute management, entity encoding, and pretty-print output.
//!
//! Pure-Rust replacement for hyperscript, react-dom/server, and cheerio.
//! Supports document and fragment modes, self-closing tags, class/style
//! helpers, and proper HTML entity encoding.

use std::collections::HashMap;
use std::fmt;
use std::fmt::Write as FmtWrite;

// ── Constants ───────────────────────────────────────────────────

/// HTML void elements (self-closing tags).
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link",
    "meta", "param", "source", "track", "wbr",
];

/// Raw text elements (content is not escaped).
const RAW_TEXT_ELEMENTS: &[&str] = &["script", "style"];

// ── HTML Node ───────────────────────────────────────────────────

/// A node in the HTML document tree.
#[derive(Debug, Clone)]
pub enum HtmlNode {
    /// An HTML element with tag, attributes, and children.
    Element {
        tag: String,
        attrs: Vec<(String, Option<String>)>,
        children: Vec<HtmlNode>,
    },
    /// A text node (will be entity-encoded on output).
    Text(String),
    /// Raw HTML (not escaped).
    Raw(String),
    /// An HTML comment.
    Comment(String),
    /// A document type declaration.
    Doctype(String),
    /// A fragment (list of nodes with no wrapper element).
    Fragment(Vec<HtmlNode>),
}

impl HtmlNode {
    /// Render to an HTML string (compact, no indentation).
    pub fn to_html(&self) -> String {
        let mut out = String::new();
        self.write_html(&mut out, false, 0);
        out
    }

    /// Render to a pretty-printed HTML string.
    pub fn to_pretty_html(&self) -> String {
        let mut out = String::new();
        self.write_html(&mut out, true, 0);
        out
    }

    fn write_html(&self, out: &mut String, pretty: bool, depth: usize) {
        let indent = if pretty {
            "  ".repeat(depth)
        } else {
            String::new()
        };
        let newline = if pretty { "\n" } else { "" };

        match self {
            HtmlNode::Element {
                tag,
                attrs,
                children,
            } => {
                out.push_str(&indent);
                out.push('<');
                out.push_str(tag);

                for (key, value) in attrs {
                    if let Some(val) = value {
                        let _ = write!(out, " {key}=\"{}\"", encode_attr(val));
                    } else {
                        // Boolean attribute.
                        let _ = write!(out, " {key}");
                    }
                }

                let is_void = VOID_ELEMENTS.contains(&tag.as_str());

                if is_void {
                    out.push_str(" />");
                    out.push_str(newline);
                    return;
                }

                out.push('>');

                let is_raw = RAW_TEXT_ELEMENTS.contains(&tag.as_str());
                let is_inline_text =
                    children.len() == 1 && matches!(&children[0], HtmlNode::Text(_));

                if is_inline_text && !pretty {
                    // Inline text without newlines.
                    for child in children {
                        if is_raw {
                            if let HtmlNode::Text(t) = child {
                                out.push_str(t);
                            }
                        } else {
                            child.write_html(out, false, 0);
                        }
                    }
                } else if children.is_empty() {
                    // Empty element.
                } else if is_inline_text {
                    // Pretty mode but just one text child — keep inline.
                    if let HtmlNode::Text(t) = &children[0] {
                        if is_raw {
                            out.push_str(t);
                        } else {
                            out.push_str(&encode_text(t));
                        }
                    }
                } else {
                    out.push_str(newline);
                    for child in children {
                        if is_raw {
                            if let HtmlNode::Text(t) = child {
                                out.push_str(&indent);
                                out.push_str("  ");
                                out.push_str(t);
                                out.push_str(newline);
                            } else {
                                child.write_html(out, pretty, depth + 1);
                            }
                        } else {
                            child.write_html(out, pretty, depth + 1);
                        }
                    }
                    out.push_str(&indent);
                }

                let _ = write!(out, "</{tag}>");
                out.push_str(newline);
            }
            HtmlNode::Text(t) => {
                out.push_str(&indent);
                out.push_str(&encode_text(t));
                out.push_str(newline);
            }
            HtmlNode::Raw(r) => {
                out.push_str(&indent);
                out.push_str(r);
                out.push_str(newline);
            }
            HtmlNode::Comment(c) => {
                let _ = write!(out, "{indent}<!-- {c} -->");
                out.push_str(newline);
            }
            HtmlNode::Doctype(dt) => {
                let _ = write!(out, "<!DOCTYPE {dt}>");
                out.push_str(newline);
            }
            HtmlNode::Fragment(children) => {
                for child in children {
                    child.write_html(out, pretty, depth);
                }
            }
        }
    }

    /// Get the inner text content recursively.
    pub fn inner_text(&self) -> String {
        match self {
            HtmlNode::Text(t) => t.clone(),
            HtmlNode::Element { children, .. } | HtmlNode::Fragment(children) => {
                children.iter().map(|c| c.inner_text()).collect()
            }
            HtmlNode::Raw(r) => strip_tags(r),
            HtmlNode::Comment(_) | HtmlNode::Doctype(_) => String::new(),
        }
    }

    /// Count the total number of elements (not text nodes).
    pub fn element_count(&self) -> usize {
        match self {
            HtmlNode::Element { children, .. } => {
                1 + children.iter().map(|c| c.element_count()).sum::<usize>()
            }
            HtmlNode::Fragment(children) => {
                children.iter().map(|c| c.element_count()).sum()
            }
            _ => 0,
        }
    }
}

impl fmt::Display for HtmlNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_html())
    }
}

// ── Element builder ─────────────────────────────────────────────

/// Fluent builder for constructing HTML elements.
pub struct Element {
    tag: String,
    attrs: Vec<(String, Option<String>)>,
    children: Vec<HtmlNode>,
}

impl Element {
    /// Create a new element with the given tag.
    pub fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            attrs: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Set an attribute with a value.
    pub fn attr(mut self, key: &str, value: &str) -> Self {
        self.attrs.push((key.to_string(), Some(value.to_string())));
        self
    }

    /// Set a boolean attribute (no value).
    pub fn bool_attr(mut self, key: &str) -> Self {
        self.attrs.push((key.to_string(), None));
        self
    }

    /// Set the `id` attribute.
    pub fn id(self, id: &str) -> Self {
        self.attr("id", id)
    }

    /// Add a CSS class. Multiple calls append classes.
    pub fn class(mut self, cls: &str) -> Self {
        // Find existing class attribute or create one.
        let existing = self
            .attrs
            .iter_mut()
            .find(|(k, _)| k == "class");

        if let Some((_, Some(val))) = existing {
            if !val.is_empty() {
                val.push(' ');
            }
            val.push_str(cls);
        } else {
            self.attrs
                .push(("class".to_string(), Some(cls.to_string())));
        }
        self
    }

    /// Add multiple CSS classes at once.
    pub fn classes(mut self, classes: &[&str]) -> Self {
        for cls in classes {
            self = self.class(cls);
        }
        self
    }

    /// Set a CSS style property.
    pub fn style(mut self, property: &str, value: &str) -> Self {
        let existing = self
            .attrs
            .iter_mut()
            .find(|(k, _)| k == "style");

        let decl = format!("{property}: {value}");
        if let Some((_, Some(val))) = existing {
            if !val.is_empty() && !val.ends_with(';') {
                val.push(';');
            }
            val.push_str(" ");
            val.push_str(&decl);
        } else {
            self.attrs
                .push(("style".to_string(), Some(decl)));
        }
        self
    }

    /// Set the `href` attribute.
    pub fn href(self, url: &str) -> Self {
        self.attr("href", url)
    }

    /// Set the `src` attribute.
    pub fn src(self, url: &str) -> Self {
        self.attr("src", url)
    }

    /// Set the `type` attribute.
    pub fn type_attr(self, t: &str) -> Self {
        self.attr("type", t)
    }

    /// Set the `name` attribute.
    pub fn name(self, n: &str) -> Self {
        self.attr("name", n)
    }

    /// Set the `value` attribute.
    pub fn value(self, v: &str) -> Self {
        self.attr("value", v)
    }

    /// Set the `title` attribute.
    pub fn title(self, t: &str) -> Self {
        self.attr("title", t)
    }

    /// Set data-* attributes.
    pub fn data(self, key: &str, value: &str) -> Self {
        self.attr(&format!("data-{key}"), value)
    }

    /// Set aria-* attributes.
    pub fn aria(self, key: &str, value: &str) -> Self {
        self.attr(&format!("aria-{key}"), value)
    }

    /// Set the `role` attribute.
    pub fn role(self, r: &str) -> Self {
        self.attr("role", r)
    }

    /// Add a text child node.
    pub fn text(mut self, content: &str) -> Self {
        self.children.push(HtmlNode::Text(content.to_string()));
        self
    }

    /// Add a child element.
    pub fn child(mut self, element: Element) -> Self {
        self.children.push(element.build());
        self
    }

    /// Add a child HtmlNode directly.
    pub fn child_node(mut self, node: HtmlNode) -> Self {
        self.children.push(node);
        self
    }

    /// Add multiple children.
    pub fn children(mut self, elements: Vec<Element>) -> Self {
        for elem in elements {
            self.children.push(elem.build());
        }
        self
    }

    /// Add raw HTML content.
    pub fn raw_html(mut self, html: &str) -> Self {
        self.children.push(HtmlNode::Raw(html.to_string()));
        self
    }

    /// Build into an HtmlNode.
    pub fn build(self) -> HtmlNode {
        HtmlNode::Element {
            tag: self.tag,
            attrs: self.attrs,
            children: self.children,
        }
    }
}

// ── Document builder ────────────────────────────────────────────

/// Builder for complete HTML documents.
pub struct DocumentBuilder {
    lang: String,
    charset: String,
    title: String,
    meta_tags: Vec<(String, String)>,
    stylesheets: Vec<String>,
    scripts: Vec<String>,
    inline_styles: Vec<String>,
    body_children: Vec<HtmlNode>,
    body_classes: Vec<String>,
}

impl DocumentBuilder {
    pub fn new(title: &str) -> Self {
        Self {
            lang: "en".to_string(),
            charset: "utf-8".to_string(),
            title: title.to_string(),
            meta_tags: Vec::new(),
            stylesheets: Vec::new(),
            scripts: Vec::new(),
            inline_styles: Vec::new(),
            body_children: Vec::new(),
            body_classes: Vec::new(),
        }
    }

    /// Set the document language.
    pub fn lang(mut self, lang: &str) -> Self {
        self.lang = lang.to_string();
        self
    }

    /// Set the charset.
    pub fn charset(mut self, charset: &str) -> Self {
        self.charset = charset.to_string();
        self
    }

    /// Add a meta tag.
    pub fn meta(mut self, name: &str, content: &str) -> Self {
        self.meta_tags
            .push((name.to_string(), content.to_string()));
        self
    }

    /// Add a stylesheet link.
    pub fn stylesheet(mut self, href: &str) -> Self {
        self.stylesheets.push(href.to_string());
        self
    }

    /// Add a script source.
    pub fn script(mut self, src: &str) -> Self {
        self.scripts.push(src.to_string());
        self
    }

    /// Add inline CSS.
    pub fn inline_style(mut self, css: &str) -> Self {
        self.inline_styles.push(css.to_string());
        self
    }

    /// Add a class to the body element.
    pub fn body_class(mut self, cls: &str) -> Self {
        self.body_classes.push(cls.to_string());
        self
    }

    /// Add a child to the body.
    pub fn body(mut self, element: Element) -> Self {
        self.body_children.push(element.build());
        self
    }

    /// Add an HtmlNode to the body.
    pub fn body_node(mut self, node: HtmlNode) -> Self {
        self.body_children.push(node);
        self
    }

    /// Build the complete HTML document.
    pub fn build(&self) -> HtmlNode {
        // Head children.
        let mut head_children: Vec<HtmlNode> = Vec::new();

        // Charset meta.
        head_children.push(
            Element::new("meta")
                .attr("charset", &self.charset)
                .build(),
        );

        // Viewport meta.
        head_children.push(
            Element::new("meta")
                .attr("name", "viewport")
                .attr("content", "width=device-width, initial-scale=1.0")
                .build(),
        );

        // Title.
        head_children.push(Element::new("title").text(&self.title).build());

        // Custom meta tags.
        for (name, content) in &self.meta_tags {
            head_children.push(
                Element::new("meta")
                    .attr("name", name)
                    .attr("content", content)
                    .build(),
            );
        }

        // Stylesheets.
        for href in &self.stylesheets {
            head_children.push(
                Element::new("link")
                    .attr("rel", "stylesheet")
                    .attr("href", href)
                    .build(),
            );
        }

        // Inline styles.
        for css in &self.inline_styles {
            head_children.push(Element::new("style").text(css).build());
        }

        // Build body.
        let mut body = Element::new("body");
        for cls in &self.body_classes {
            body = body.class(cls);
        }
        for child in &self.body_children {
            body = body.child_node(child.clone());
        }

        // Add scripts at end of body.
        for src in &self.scripts {
            body = body.child(Element::new("script").attr("src", src));
        }

        // Assemble html element.
        let mut head = Element::new("head");
        for child in head_children {
            head = head.child_node(child);
        }

        let html = Element::new("html")
            .attr("lang", &self.lang)
            .child(head)
            .child(body);

        HtmlNode::Fragment(vec![
            HtmlNode::Doctype("html".to_string()),
            html.build(),
        ])
    }
}

// ── Convenience constructors ────────────────────────────────────

/// Create a `<div>` element.
pub fn div() -> Element {
    Element::new("div")
}

/// Create a `<span>` element.
pub fn span() -> Element {
    Element::new("span")
}

/// Create a `<p>` element.
pub fn p() -> Element {
    Element::new("p")
}

/// Create a heading element (`<h1>` through `<h6>`).
pub fn heading(level: u8) -> Element {
    let tag = format!("h{}", level.clamp(1, 6));
    Element::new(&tag)
}

/// Create an `<a>` element.
pub fn a(href: &str) -> Element {
    Element::new("a").href(href)
}

/// Create an `<img>` element.
pub fn img(src: &str, alt: &str) -> Element {
    Element::new("img").src(src).attr("alt", alt)
}

/// Create an `<input>` element.
pub fn input(input_type: &str) -> Element {
    Element::new("input").type_attr(input_type)
}

/// Create a `<button>` element.
pub fn button(text: &str) -> Element {
    Element::new("button").text(text)
}

/// Create a `<form>` element.
pub fn form() -> Element {
    Element::new("form")
}

/// Create a `<ul>` element from items.
pub fn ul(items: &[&str]) -> Element {
    let mut list = Element::new("ul");
    for item in items {
        list = list.child(Element::new("li").text(item));
    }
    list
}

/// Create an `<ol>` element from items.
pub fn ol(items: &[&str]) -> Element {
    let mut list = Element::new("ol");
    for item in items {
        list = list.child(Element::new("li").text(item));
    }
    list
}

/// Create a `<table>` element from headers and rows.
pub fn table(headers: &[&str], rows: &[Vec<&str>]) -> Element {
    let mut thead = Element::new("thead");
    let mut header_row = Element::new("tr");
    for h in headers {
        header_row = header_row.child(Element::new("th").text(h));
    }
    thead = thead.child(header_row);

    let mut tbody = Element::new("tbody");
    for row in rows {
        let mut tr = Element::new("tr");
        for cell in row {
            tr = tr.child(Element::new("td").text(cell));
        }
        tbody = tbody.child(tr);
    }

    Element::new("table").child(thead).child(tbody)
}

/// Create a `<br />` element.
pub fn br() -> Element {
    Element::new("br")
}

/// Create an `<hr />` element.
pub fn hr() -> Element {
    Element::new("hr")
}

/// Create a text node.
pub fn text_node(content: &str) -> HtmlNode {
    HtmlNode::Text(content.to_string())
}

/// Create a comment node.
pub fn comment(content: &str) -> HtmlNode {
    HtmlNode::Comment(content.to_string())
}

/// Create a fragment (multiple nodes without a wrapper).
pub fn fragment(nodes: Vec<HtmlNode>) -> HtmlNode {
    HtmlNode::Fragment(nodes)
}

// ── HTML entity encoding ────────────────────────────────────────

/// Encode text for HTML content (escapes <, >, &, etc.).
pub fn encode_text(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Encode text for HTML attribute values.
pub fn encode_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

/// Decode HTML entities back to their characters.
pub fn decode_entities(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;

    while !rest.is_empty() {
        if rest.starts_with('&') {
            if let Some(end) = rest.find(';') {
                let entity = &rest[..end + 1];
                let decoded = match entity {
                    "&amp;" => "&",
                    "&lt;" => "<",
                    "&gt;" => ">",
                    "&quot;" => "\"",
                    "&#x27;" | "&#39;" => "'",
                    "&apos;" => "'",
                    "&nbsp;" => "\u{00a0}",
                    _ => {
                        // Try numeric entity.
                        if entity.starts_with("&#x") || entity.starts_with("&#X") {
                            let hex = &entity[3..entity.len() - 1];
                            if let Ok(cp) = u32::from_str_radix(hex, 16) {
                                if let Some(ch) = char::from_u32(cp) {
                                    out.push(ch);
                                    rest = &rest[end + 1..];
                                    continue;
                                }
                            }
                        } else if entity.starts_with("&#") {
                            let num = &entity[2..entity.len() - 1];
                            if let Ok(cp) = num.parse::<u32>() {
                                if let Some(ch) = char::from_u32(cp) {
                                    out.push(ch);
                                    rest = &rest[end + 1..];
                                    continue;
                                }
                            }
                        }
                        out.push_str(entity);
                        rest = &rest[end + 1..];
                        continue;
                    }
                };
                out.push_str(decoded);
                rest = &rest[end + 1..];
            } else {
                out.push('&');
                rest = &rest[1..];
            }
        } else {
            out.push(rest.chars().next().unwrap());
            rest = &rest[rest.chars().next().unwrap().len_utf8()..];
        }
    }

    out
}

fn strip_tags(s: &str) -> String {
    let mut out = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            out.push(ch);
        }
    }
    out
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_element() {
        let html = Element::new("div").text("Hello").build().to_html();
        assert_eq!(html, "<div>Hello</div>");
    }

    #[test]
    fn test_attributes() {
        let html = Element::new("a")
            .attr("href", "https://example.com")
            .text("Link")
            .build()
            .to_html();
        assert_eq!(html, "<a href=\"https://example.com\">Link</a>");
    }

    #[test]
    fn test_boolean_attribute() {
        let html = Element::new("input")
            .type_attr("checkbox")
            .bool_attr("checked")
            .build()
            .to_html();
        assert!(html.contains("checked"));
        assert!(html.contains("type=\"checkbox\""));
    }

    #[test]
    fn test_self_closing() {
        let html = Element::new("br").build().to_html();
        assert_eq!(html, "<br />");
    }

    #[test]
    fn test_img_self_closing() {
        let html = img("photo.jpg", "Photo").build().to_html();
        assert!(html.contains("<img"));
        assert!(html.contains("src=\"photo.jpg\""));
        assert!(html.contains("alt=\"Photo\""));
        assert!(html.contains("/>"));
    }

    #[test]
    fn test_nested_elements() {
        let html = div()
            .child(p().text("Paragraph 1"))
            .child(p().text("Paragraph 2"))
            .build()
            .to_html();
        assert!(html.contains("<div>"));
        assert!(html.contains("<p>Paragraph 1</p>"));
        assert!(html.contains("<p>Paragraph 2</p>"));
    }

    #[test]
    fn test_class_helper() {
        let html = div().class("container").class("main").build().to_html();
        assert!(html.contains("class=\"container main\""));
    }

    #[test]
    fn test_style_helper() {
        let html = div()
            .style("color", "red")
            .style("margin", "10px")
            .build()
            .to_html();
        assert!(html.contains("color: red"));
        assert!(html.contains("margin: 10px"));
    }

    #[test]
    fn test_entity_encoding() {
        let html = p().text("A < B & C > D").build().to_html();
        assert_eq!(html, "<p>A &lt; B &amp; C &gt; D</p>");
    }

    #[test]
    fn test_attr_encoding() {
        let html = Element::new("div")
            .attr("title", "Say \"hello\"")
            .build()
            .to_html();
        assert!(html.contains("Say &quot;hello&quot;"));
    }

    #[test]
    fn test_pretty_print() {
        let html = div()
            .child(p().text("Hello"))
            .child(p().text("World"))
            .build()
            .to_pretty_html();
        assert!(html.contains("  <p>"));
        assert!(html.contains('\n'));
    }

    #[test]
    fn test_document_builder() {
        let doc = DocumentBuilder::new("My Page")
            .lang("en")
            .meta("description", "Test page")
            .stylesheet("style.css")
            .body(div().text("Content"))
            .build();
        let html = doc.to_pretty_html();
        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("<title>My Page</title>"));
        assert!(html.contains("description"));
        assert!(html.contains("style.css"));
    }

    #[test]
    fn test_fragment() {
        let frag = fragment(vec![
            p().text("One").build(),
            p().text("Two").build(),
        ]);
        let html = frag.to_html();
        assert!(html.contains("<p>One</p>"));
        assert!(html.contains("<p>Two</p>"));
        assert!(!html.contains("<div"));
    }

    #[test]
    fn test_comment() {
        let node = comment("This is a comment");
        let html = node.to_html();
        assert_eq!(html, "<!-- This is a comment -->");
    }

    #[test]
    fn test_raw_html() {
        let node = div().raw_html("<strong>Bold</strong>").build().to_html();
        assert!(node.contains("<strong>Bold</strong>"));
    }

    #[test]
    fn test_ul_helper() {
        let html = ul(&["Item 1", "Item 2", "Item 3"]).build().to_html();
        assert!(html.contains("<ul>"));
        assert!(html.contains("<li>Item 1</li>"));
        assert!(html.contains("<li>Item 3</li>"));
    }

    #[test]
    fn test_ol_helper() {
        let html = ol(&["First", "Second"]).build().to_html();
        assert!(html.contains("<ol>"));
        assert!(html.contains("<li>First</li>"));
    }

    #[test]
    fn test_table_helper() {
        let html = table(
            &["Name", "Age"],
            &[vec!["Alice", "30"], vec!["Bob", "25"]],
        )
        .build()
        .to_html();
        assert!(html.contains("<table>"));
        assert!(html.contains("<th>Name</th>"));
        assert!(html.contains("<td>Alice</td>"));
    }

    #[test]
    fn test_heading_helper() {
        let html = heading(1).text("Title").build().to_html();
        assert_eq!(html, "<h1>Title</h1>");
    }

    #[test]
    fn test_inner_text() {
        let node = div()
            .child(p().text("Hello "))
            .child(span().text("World"))
            .build();
        assert_eq!(node.inner_text(), "Hello World");
    }

    #[test]
    fn test_element_count() {
        let node = div()
            .child(p().text("A"))
            .child(p().text("B"))
            .build();
        assert_eq!(node.element_count(), 3); // div + 2 p
    }

    #[test]
    fn test_decode_entities() {
        assert_eq!(decode_entities("&amp;"), "&");
        assert_eq!(decode_entities("&lt;"), "<");
        assert_eq!(decode_entities("&gt;"), ">");
        assert_eq!(decode_entities("&quot;"), "\"");
        assert_eq!(decode_entities("&#x41;"), "A");
        assert_eq!(decode_entities("&#65;"), "A");
    }

    #[test]
    fn test_data_attribute() {
        let html = div().data("id", "123").build().to_html();
        assert!(html.contains("data-id=\"123\""));
    }

    #[test]
    fn test_aria_attribute() {
        let html = Element::new("button")
            .aria("label", "Close")
            .text("X")
            .build()
            .to_html();
        assert!(html.contains("aria-label=\"Close\""));
    }

    #[test]
    fn test_classes_helper() {
        let html = div().classes(&["a", "b", "c"]).build().to_html();
        assert!(html.contains("class=\"a b c\""));
    }

    #[test]
    fn test_input_types() {
        let html = input("email").name("user_email").build().to_html();
        assert!(html.contains("type=\"email\""));
        assert!(html.contains("name=\"user_email\""));
        assert!(html.contains("/>"));
    }

    #[test]
    fn test_script_in_document() {
        let doc = DocumentBuilder::new("Test")
            .script("app.js")
            .body(div().text("Content"))
            .build();
        let html = doc.to_html();
        assert!(html.contains("app.js"));
    }

    #[test]
    fn test_inline_style_in_document() {
        let doc = DocumentBuilder::new("Test")
            .inline_style("body { margin: 0; }")
            .body(div().text("Content"))
            .build();
        let html = doc.to_html();
        assert!(html.contains("body { margin: 0; }"));
    }

    #[test]
    fn test_empty_element() {
        let html = div().build().to_html();
        assert_eq!(html, "<div></div>");
    }

    #[test]
    fn test_br_and_hr() {
        assert!(br().build().to_html().contains("<br />"));
        assert!(hr().build().to_html().contains("<hr />"));
    }
}
