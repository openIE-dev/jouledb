//! Type-safe HTML builder — replaces JSX with a fluent Rust API.

use crate::vdom::{EventBinding, VNode};
use std::collections::HashMap;

// ── ElementBuilder ──────────────────────────────────────────────

/// Fluent builder for constructing `VNode::Element` values.
pub struct ElementBuilder {
    tag: String,
    attrs: HashMap<String, String>,
    children: Vec<VNode>,
    key: Option<String>,
    event_handlers: Vec<EventBinding>,
}

impl ElementBuilder {
    fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            attrs: HashMap::new(),
            children: Vec::new(),
            key: None,
            event_handlers: Vec::new(),
        }
    }

    // ── Generic attribute setters ───────────────────────────────

    pub fn attr(mut self, key: &str, value: &str) -> Self {
        self.attrs.insert(key.to_string(), value.to_string());
        self
    }

    pub fn class(mut self, name: &str) -> Self {
        let entry = self.attrs.entry("class".to_string()).or_default();
        if !entry.is_empty() {
            entry.push(' ');
        }
        entry.push_str(name);
        self
    }

    pub fn id(self, id: &str) -> Self {
        self.attr("id", id)
    }

    pub fn style(mut self, property: &str, value: &str) -> Self {
        let entry = self.attrs.entry("style".to_string()).or_default();
        if !entry.is_empty() && !entry.ends_with(';') {
            entry.push(';');
        }
        entry.push_str(property);
        entry.push(':');
        entry.push_str(value);
        self
    }

    // ── Children ────────────────────────────────────────────────

    pub fn child(mut self, node: impl Into<VNode>) -> Self {
        self.children.push(node.into());
        self
    }

    pub fn children(mut self, nodes: Vec<VNode>) -> Self {
        self.children.extend(nodes);
        self
    }

    pub fn text(mut self, s: &str) -> Self {
        self.children.push(VNode::Text(s.to_string()));
        self
    }

    // ── Key & events ────────────────────────────────────────────

    pub fn key(mut self, k: &str) -> Self {
        self.key = Some(k.to_string());
        self
    }

    pub fn on(mut self, event: &str, handler_id: u64) -> Self {
        self.event_handlers.push(EventBinding {
            event_name: event.to_string(),
            handler_id,
        });
        self
    }

    // ── Convenience attribute shortcuts ─────────────────────────

    pub fn href(self, url: &str) -> Self {
        self.attr("href", url)
    }

    pub fn src(self, url: &str) -> Self {
        self.attr("src", url)
    }

    pub fn type_(self, t: &str) -> Self {
        self.attr("type", t)
    }

    pub fn placeholder(self, t: &str) -> Self {
        self.attr("placeholder", t)
    }

    pub fn disabled(self, b: bool) -> Self {
        if b { self.attr("disabled", "true") } else { self }
    }

    pub fn checked(self, b: bool) -> Self {
        if b { self.attr("checked", "true") } else { self }
    }

    pub fn value(self, v: &str) -> Self {
        self.attr("value", v)
    }

    // ── Build ───────────────────────────────────────────────────

    pub fn build(self) -> VNode {
        VNode::Element {
            tag: self.tag,
            attrs: self.attrs,
            key: self.key,
            children: self.children,
            event_handlers: self.event_handlers,
        }
    }
}

/// Auto-convert `ElementBuilder` into `VNode`.
impl From<ElementBuilder> for VNode {
    fn from(b: ElementBuilder) -> Self {
        b.build()
    }
}

// ── Shorthand constructors ──────────────────────────────────────

/// Create a `VNode::Text` node.
pub fn text(s: &str) -> VNode {
    VNode::Text(s.to_string())
}

/// Create a `VNode::Fragment`.
pub fn fragment(children: Vec<VNode>) -> VNode {
    VNode::Fragment(children)
}

// ── Tag builder functions ───────────────────────────────────────

macro_rules! tag_fn {
    ($name:ident, $tag:expr) => {
        pub fn $name() -> ElementBuilder {
            ElementBuilder::new($tag)
        }
    };
}

tag_fn!(div, "div");
tag_fn!(span, "span");
tag_fn!(p, "p");
tag_fn!(h1, "h1");
tag_fn!(h2, "h2");
tag_fn!(h3, "h3");
tag_fn!(h4, "h4");
tag_fn!(h5, "h5");
tag_fn!(h6, "h6");
tag_fn!(a, "a");
tag_fn!(button, "button");
tag_fn!(input, "input");
tag_fn!(form, "form");
tag_fn!(ul, "ul");
tag_fn!(ol, "ol");
tag_fn!(li, "li");
tag_fn!(table, "table");
tag_fn!(tr, "tr");
tag_fn!(td, "td");
tag_fn!(th, "th");
tag_fn!(thead, "thead");
tag_fn!(tbody, "tbody");
tag_fn!(img, "img");
tag_fn!(video, "video");
tag_fn!(audio, "audio");
tag_fn!(canvas, "canvas");
tag_fn!(section, "section");
tag_fn!(article, "article");
tag_fn!(nav, "nav");
tag_fn!(header, "header");
tag_fn!(footer, "footer");
tag_fn!(main_, "main");
tag_fn!(aside, "aside");
tag_fn!(label, "label");
tag_fn!(textarea, "textarea");
tag_fn!(select, "select");
tag_fn!(option, "option");

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vdom::VNode;

    #[test]
    fn div_builds_element() {
        let node: VNode = div().build();
        match node {
            VNode::Element { tag, .. } => assert_eq!(tag, "div"),
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn nested_elements() {
        let node: VNode = ul()
            .child(li().text("item 1"))
            .child(li().text("item 2"))
            .build();
        match node {
            VNode::Element { children, .. } => assert_eq!(children.len(), 2),
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn class_chaining() {
        let node = div().class("a").class("b").build();
        match node {
            VNode::Element { attrs, .. } => {
                assert_eq!(attrs.get("class").map(String::as_str), Some("a b"));
            }
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn style_property() {
        let node = div().style("color", "red").style("margin", "0").build();
        match node {
            VNode::Element { attrs, .. } => {
                let s = attrs.get("style").expect("style attr");
                assert!(s.contains("color:red"));
                assert!(s.contains("margin:0"));
            }
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn text_shorthand() {
        let node = text("hello");
        assert!(matches!(node, VNode::Text(ref s) if s == "hello"));
    }

    #[test]
    fn fragment_builder() {
        let node = fragment(vec![text("a"), text("b")]);
        match node {
            VNode::Fragment(children) => assert_eq!(children.len(), 2),
            _ => panic!("expected fragment"),
        }
    }

    #[test]
    fn into_vnode_auto_converts() {
        // ElementBuilder implements Into<VNode>
        let node: VNode = div().class("x").into();
        match node {
            VNode::Element { tag, attrs, .. } => {
                assert_eq!(tag, "div");
                assert_eq!(attrs.get("class").map(String::as_str), Some("x"));
            }
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn convenience_attrs() {
        let link: VNode = a().href("https://example.com").build();
        match link {
            VNode::Element { attrs, .. } => {
                assert_eq!(attrs.get("href").map(String::as_str), Some("https://example.com"));
            }
            _ => panic!("expected element"),
        }

        let image: VNode = img().src("/logo.png").build();
        match image {
            VNode::Element { attrs, .. } => {
                assert_eq!(attrs.get("src").map(String::as_str), Some("/logo.png"));
            }
            _ => panic!("expected element"),
        }

        let btn: VNode = button().disabled(true).build();
        match btn {
            VNode::Element { attrs, .. } => {
                assert_eq!(attrs.get("disabled").map(String::as_str), Some("true"));
            }
            _ => panic!("expected element"),
        }

        let btn2: VNode = button().disabled(false).build();
        match btn2 {
            VNode::Element { attrs, .. } => {
                assert!(attrs.get("disabled").is_none());
            }
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn complex_nested_structure() {
        let expected = VNode::Element {
            tag: "div".into(),
            attrs: {
                let mut m = std::collections::HashMap::new();
                m.insert("class".into(), "container".into());
                m
            },
            key: None,
            children: vec![
                VNode::Element {
                    tag: "h1".into(),
                    attrs: std::collections::HashMap::new(),
                    key: None,
                    children: vec![VNode::Text("Title".into())],
                    event_handlers: vec![],
                },
                VNode::Element {
                    tag: "p".into(),
                    attrs: std::collections::HashMap::new(),
                    key: None,
                    children: vec![VNode::Text("Body".into())],
                    event_handlers: vec![],
                },
            ],
            event_handlers: vec![],
        };

        let built: VNode = div()
            .class("container")
            .child(h1().text("Title"))
            .child(p().text("Body"))
            .build();

        assert_eq!(built, expected);
    }

    #[test]
    fn input_with_type_and_placeholder() {
        let node = input().type_("email").placeholder("you@example.com").build();
        match node {
            VNode::Element { attrs, .. } => {
                assert_eq!(attrs.get("type").map(String::as_str), Some("email"));
                assert_eq!(attrs.get("placeholder").map(String::as_str), Some("you@example.com"));
            }
            _ => panic!("expected element"),
        }
    }

    #[test]
    fn event_handler_on_button() {
        let node = button().text("Click").on("click", 42).build();
        match node {
            VNode::Element { event_handlers, .. } => {
                assert_eq!(event_handlers.len(), 1);
                assert_eq!(event_handlers[0].event_name, "click");
                assert_eq!(event_handlers[0].handler_id, 42);
            }
            _ => panic!("expected element"),
        }
    }
}
