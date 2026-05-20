//! Server/client hydration — server render to HTML string, client hydration
//! (attach events to existing DOM), mismatch detection, partial hydration,
//! progressive hydration hints, and hydration statistics.
//!
//! Replaces React hydrate / Vue SSR hydration with a pure-Rust model.
//! Server-side renders a virtual tree to an HTML string; client-side
//! hydration walks the existing markup, attaches event handlers, and
//! detects mismatches between server and client output.

use std::collections::HashMap;

// ── Virtual node (simplified for hydration) ─────────────────────────────

/// A node in the virtual tree used for hydration.
#[derive(Debug, Clone, PartialEq)]
pub enum HydrationNode {
    Element {
        tag: String,
        attrs: HashMap<String, String>,
        children: Vec<HydrationNode>,
        hydration_key: Option<String>,
        interactive: bool,
    },
    Text(String),
    Comment(String),
}

impl HydrationNode {
    pub fn element(tag: &str) -> Self {
        HydrationNode::Element {
            tag: tag.to_string(),
            attrs: HashMap::new(),
            children: Vec::new(),
            hydration_key: None,
            interactive: false,
        }
    }

    pub fn text(s: &str) -> Self {
        HydrationNode::Text(s.to_string())
    }

    pub fn comment(s: &str) -> Self {
        HydrationNode::Comment(s.to_string())
    }

    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        if let HydrationNode::Element { ref mut attrs, .. } = self {
            attrs.insert(key.to_string(), value.to_string());
        }
        self
    }

    pub fn with_child(mut self, child: HydrationNode) -> Self {
        if let HydrationNode::Element { ref mut children, .. } = self {
            children.push(child);
        }
        self
    }

    pub fn with_key(mut self, key: &str) -> Self {
        if let HydrationNode::Element { ref mut hydration_key, .. } = self {
            *hydration_key = Some(key.to_string());
        }
        self
    }

    pub fn with_interactive(mut self) -> Self {
        if let HydrationNode::Element { ref mut interactive, .. } = self {
            *interactive = true;
        }
        self
    }

    /// Count total nodes in this subtree.
    pub fn node_count(&self) -> usize {
        match self {
            HydrationNode::Text(_) | HydrationNode::Comment(_) => 1,
            HydrationNode::Element { children, .. } => {
                1 + children.iter().map(|c| c.node_count()).sum::<usize>()
            }
        }
    }

    /// Count interactive elements (needing hydration).
    pub fn interactive_count(&self) -> usize {
        match self {
            HydrationNode::Text(_) | HydrationNode::Comment(_) => 0,
            HydrationNode::Element { interactive, children, .. } => {
                let self_count = if *interactive { 1 } else { 0 };
                self_count + children.iter().map(|c| c.interactive_count()).sum::<usize>()
            }
        }
    }
}

// ── Server render ───────────────────────────────────────────────────────

/// Void elements that should not have closing tags.
const VOID_ELEMENTS: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
];

/// Render a hydration node tree to an HTML string (server-side rendering).
pub fn render_to_html(node: &HydrationNode) -> String {
    let mut buf = String::new();
    render_node(node, &mut buf, 0);
    buf
}

/// Render with hydration markers (data-hk attributes for keys).
pub fn render_to_html_with_markers(node: &HydrationNode) -> String {
    let mut buf = String::new();
    render_node_with_markers(node, &mut buf, 0, &mut 0);
    buf
}

fn render_node(node: &HydrationNode, buf: &mut String, depth: usize) {
    match node {
        HydrationNode::Text(text) => {
            buf.push_str(&escape_html(text));
        }
        HydrationNode::Comment(text) => {
            buf.push_str("<!--");
            buf.push_str(text);
            buf.push_str("-->");
        }
        HydrationNode::Element { tag, attrs, children, .. } => {
            buf.push('<');
            buf.push_str(tag);

            // Sort attrs for deterministic output
            let mut sorted_attrs: Vec<_> = attrs.iter().collect();
            sorted_attrs.sort_by_key(|(k, _)| (*k).clone());

            for (key, value) in &sorted_attrs {
                buf.push(' ');
                buf.push_str(key);
                buf.push_str("=\"");
                buf.push_str(&escape_attr(value));
                buf.push('"');
            }

            let is_void = VOID_ELEMENTS.contains(&tag.as_str());
            if is_void {
                buf.push_str(" />");
                return;
            }

            buf.push('>');

            for child in children {
                render_node(child, buf, depth + 1);
            }

            buf.push_str("</");
            buf.push_str(tag);
            buf.push('>');
        }
    }
}

fn render_node_with_markers(
    node: &HydrationNode,
    buf: &mut String,
    depth: usize,
    counter: &mut u64,
) {
    match node {
        HydrationNode::Text(text) => {
            buf.push_str(&escape_html(text));
        }
        HydrationNode::Comment(text) => {
            buf.push_str("<!--");
            buf.push_str(text);
            buf.push_str("-->");
        }
        HydrationNode::Element { tag, attrs, children, hydration_key, .. } => {
            buf.push('<');
            buf.push_str(tag);

            // Add hydration marker
            let marker_id = *counter;
            *counter += 1;
            buf.push_str(" data-hid=\"");
            buf.push_str(&marker_id.to_string());
            buf.push('"');

            if let Some(key) = hydration_key {
                buf.push_str(" data-hk=\"");
                buf.push_str(&escape_attr(key));
                buf.push('"');
            }

            let mut sorted_attrs: Vec<_> = attrs.iter().collect();
            sorted_attrs.sort_by_key(|(k, _)| (*k).clone());

            for (key, value) in &sorted_attrs {
                buf.push(' ');
                buf.push_str(key);
                buf.push_str("=\"");
                buf.push_str(&escape_attr(value));
                buf.push('"');
            }

            let is_void = VOID_ELEMENTS.contains(&tag.as_str());
            if is_void {
                buf.push_str(" />");
                return;
            }

            buf.push('>');

            for child in children {
                render_node_with_markers(child, buf, depth + 1, counter);
            }

            buf.push_str("</");
            buf.push_str(tag);
            buf.push('>');
        }
    }
}

fn escape_html(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            _ => out.push(ch),
        }
    }
    out
}

fn escape_attr(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Hydration mismatch ──────────────────────────────────────────────────

/// A mismatch detected during hydration.
#[derive(Debug, Clone, PartialEq)]
pub enum HydrationMismatch {
    /// Tag name differs.
    TagMismatch {
        path: Vec<usize>,
        server_tag: String,
        client_tag: String,
    },
    /// Text content differs.
    TextMismatch {
        path: Vec<usize>,
        server_text: String,
        client_text: String,
    },
    /// Attribute value differs.
    AttrMismatch {
        path: Vec<usize>,
        attr: String,
        server_value: Option<String>,
        client_value: Option<String>,
    },
    /// Child count differs.
    ChildCountMismatch {
        path: Vec<usize>,
        server_count: usize,
        client_count: usize,
    },
    /// Node type differs (element vs text).
    TypeMismatch {
        path: Vec<usize>,
        server_type: String,
        client_type: String,
    },
}

/// Compare a server-rendered tree with a client-rendered tree and find mismatches.
pub fn detect_mismatches(
    server: &HydrationNode,
    client: &HydrationNode,
) -> Vec<HydrationMismatch> {
    let mut mismatches = Vec::new();
    detect_recursive(server, client, &[], &mut mismatches);
    mismatches
}

fn node_type_name(node: &HydrationNode) -> &str {
    match node {
        HydrationNode::Element { .. } => "element",
        HydrationNode::Text(_) => "text",
        HydrationNode::Comment(_) => "comment",
    }
}

fn detect_recursive(
    server: &HydrationNode,
    client: &HydrationNode,
    path: &[usize],
    mismatches: &mut Vec<HydrationMismatch>,
) {
    match (server, client) {
        (HydrationNode::Text(st), HydrationNode::Text(ct)) => {
            if st != ct {
                mismatches.push(HydrationMismatch::TextMismatch {
                    path: path.to_vec(),
                    server_text: st.clone(),
                    client_text: ct.clone(),
                });
            }
        }
        (
            HydrationNode::Element { tag: stag, attrs: sattrs, children: schildren, .. },
            HydrationNode::Element { tag: ctag, attrs: cattrs, children: cchildren, .. },
        ) => {
            if stag != ctag {
                mismatches.push(HydrationMismatch::TagMismatch {
                    path: path.to_vec(),
                    server_tag: stag.clone(),
                    client_tag: ctag.clone(),
                });
                return;
            }

            // Check attributes
            let mut all_keys: Vec<String> = sattrs.keys().chain(cattrs.keys()).cloned().collect();
            all_keys.sort();
            all_keys.dedup();
            for key in &all_keys {
                let sv = sattrs.get(key);
                let cv = cattrs.get(key);
                if sv != cv {
                    mismatches.push(HydrationMismatch::AttrMismatch {
                        path: path.to_vec(),
                        attr: key.clone(),
                        server_value: sv.cloned(),
                        client_value: cv.cloned(),
                    });
                }
            }

            // Check child count
            if schildren.len() != cchildren.len() {
                mismatches.push(HydrationMismatch::ChildCountMismatch {
                    path: path.to_vec(),
                    server_count: schildren.len(),
                    client_count: cchildren.len(),
                });
            }

            // Recurse into children (up to the shorter list)
            let min_len = schildren.len().min(cchildren.len());
            for i in 0..min_len {
                let mut child_path = path.to_vec();
                child_path.push(i);
                detect_recursive(&schildren[i], &cchildren[i], &child_path, mismatches);
            }
        }
        _ => {
            mismatches.push(HydrationMismatch::TypeMismatch {
                path: path.to_vec(),
                server_type: node_type_name(server).to_string(),
                client_type: node_type_name(client).to_string(),
            });
        }
    }
}

// ── Hydration hints ─────────────────────────────────────────────────────

/// Progressive hydration hint for a subtree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HydrationHint {
    /// Hydrate immediately on load.
    Eager,
    /// Hydrate when the element becomes visible.
    Visible,
    /// Hydrate on user interaction (click/focus).
    Interaction,
    /// Hydrate when the browser is idle.
    Idle,
    /// Never hydrate (static content).
    Never,
}

/// Assign hydration hints to a node tree based on interactivity.
pub fn assign_hydration_hints(node: &HydrationNode) -> Vec<(Vec<usize>, HydrationHint)> {
    let mut hints = Vec::new();
    assign_hints_recursive(node, &[], &mut hints);
    hints
}

fn assign_hints_recursive(
    node: &HydrationNode,
    path: &[usize],
    hints: &mut Vec<(Vec<usize>, HydrationHint)>,
) {
    match node {
        HydrationNode::Element { interactive, children, .. } => {
            let hint = if *interactive {
                HydrationHint::Eager
            } else if children.iter().any(|c| {
                matches!(c, HydrationNode::Element { interactive: true, .. })
            }) {
                HydrationHint::Visible
            } else {
                HydrationHint::Never
            };
            hints.push((path.to_vec(), hint));

            for (i, child) in children.iter().enumerate() {
                let mut child_path = path.to_vec();
                child_path.push(i);
                assign_hints_recursive(child, &child_path, hints);
            }
        }
        HydrationNode::Text(_) => {
            hints.push((path.to_vec(), HydrationHint::Never));
        }
        HydrationNode::Comment(_) => {
            hints.push((path.to_vec(), HydrationHint::Never));
        }
    }
}

// ── Hydration statistics ────────────────────────────────────────────────

/// Statistics from a hydration pass.
#[derive(Debug, Clone, Default)]
pub struct HydrationStats {
    pub total_nodes: usize,
    pub elements_hydrated: usize,
    pub text_nodes: usize,
    pub interactive_nodes: usize,
    pub mismatch_count: usize,
    pub skipped_static: usize,
}

impl HydrationStats {
    /// Compute stats from a node tree and optional mismatch list.
    pub fn compute(
        node: &HydrationNode,
        mismatches: &[HydrationMismatch],
    ) -> Self {
        let mut stats = Self::default();
        stats.mismatch_count = mismatches.len();
        count_stats(node, &mut stats);
        stats
    }

    /// Hydration success rate (0.0 to 1.0).
    pub fn success_rate(&self) -> f64 {
        if self.total_nodes == 0 {
            return 1.0;
        }
        let mismatched = self.mismatch_count as f64;
        let total = self.total_nodes as f64;
        (1.0 - mismatched / total).max(0.0)
    }
}

fn count_stats(node: &HydrationNode, stats: &mut HydrationStats) {
    stats.total_nodes += 1;
    match node {
        HydrationNode::Text(_) => {
            stats.text_nodes += 1;
        }
        HydrationNode::Comment(_) => {}
        HydrationNode::Element { interactive, children, .. } => {
            stats.elements_hydrated += 1;
            if *interactive {
                stats.interactive_nodes += 1;
            } else {
                stats.skipped_static += 1;
            }
            for child in children {
                count_stats(child, stats);
            }
        }
    }
}

// ── Partial hydration ───────────────────────────────────────────────────

/// Extract only the interactive subtrees that need hydration.
pub fn extract_interactive_subtrees(node: &HydrationNode) -> Vec<(Vec<usize>, HydrationNode)> {
    let mut result = Vec::new();
    extract_interactive_recursive(node, &[], &mut result);
    result
}

fn extract_interactive_recursive(
    node: &HydrationNode,
    path: &[usize],
    result: &mut Vec<(Vec<usize>, HydrationNode)>,
) {
    if let HydrationNode::Element { interactive, children, .. } = node {
        if *interactive {
            result.push((path.to_vec(), node.clone()));
        }
        for (i, child) in children.iter().enumerate() {
            let mut child_path = path.to_vec();
            child_path.push(i);
            extract_interactive_recursive(child, &child_path, result);
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> HydrationNode {
        HydrationNode::element("div")
            .with_attr("class", "container")
            .with_child(HydrationNode::element("h1").with_child(HydrationNode::text("Hello")))
            .with_child(
                HydrationNode::element("button")
                    .with_attr("id", "btn")
                    .with_interactive()
                    .with_child(HydrationNode::text("Click")),
            )
    }

    #[test]
    fn render_simple_text() {
        let node = HydrationNode::text("hello");
        assert_eq!(render_to_html(&node), "hello");
    }

    #[test]
    fn render_element_with_attrs() {
        let node = HydrationNode::element("div")
            .with_attr("class", "red");
        let html = render_to_html(&node);
        assert!(html.contains("<div"));
        assert!(html.contains("class=\"red\""));
        assert!(html.contains("</div>"));
    }

    #[test]
    fn render_void_element() {
        let node = HydrationNode::element("br");
        let html = render_to_html(&node);
        assert_eq!(html, "<br />");
    }

    #[test]
    fn render_nested_tree() {
        let tree = sample_tree();
        let html = render_to_html(&tree);
        assert!(html.contains("<div"));
        assert!(html.contains("<h1>Hello</h1>"));
        assert!(html.contains("<button"));
        assert!(html.contains("Click"));
    }

    #[test]
    fn render_escapes_html() {
        let node = HydrationNode::text("<script>alert('xss')</script>");
        let html = render_to_html(&node);
        assert!(!html.contains("<script>"));
        assert!(html.contains("&lt;script&gt;"));
    }

    #[test]
    fn render_escapes_attributes() {
        let node = HydrationNode::element("div")
            .with_attr("title", "say \"hello\"");
        let html = render_to_html(&node);
        assert!(html.contains("&quot;hello&quot;"));
    }

    #[test]
    fn render_comment() {
        let node = HydrationNode::comment("hydration boundary");
        let html = render_to_html(&node);
        assert_eq!(html, "<!--hydration boundary-->");
    }

    #[test]
    fn render_with_markers_adds_hid() {
        let node = HydrationNode::element("div")
            .with_child(HydrationNode::element("span"));
        let html = render_to_html_with_markers(&node);
        assert!(html.contains("data-hid=\"0\""));
        assert!(html.contains("data-hid=\"1\""));
    }

    #[test]
    fn render_with_markers_adds_hk() {
        let node = HydrationNode::element("div").with_key("my-key");
        let html = render_to_html_with_markers(&node);
        assert!(html.contains("data-hk=\"my-key\""));
    }

    #[test]
    fn detect_no_mismatches_identical() {
        let tree = sample_tree();
        let mismatches = detect_mismatches(&tree, &tree);
        assert!(mismatches.is_empty());
    }

    #[test]
    fn detect_text_mismatch() {
        let server = HydrationNode::text("hello");
        let client = HydrationNode::text("world");
        let mismatches = detect_mismatches(&server, &client);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(&mismatches[0], HydrationMismatch::TextMismatch { .. }));
    }

    #[test]
    fn detect_tag_mismatch() {
        let server = HydrationNode::element("div");
        let client = HydrationNode::element("span");
        let mismatches = detect_mismatches(&server, &client);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(&mismatches[0], HydrationMismatch::TagMismatch { .. }));
    }

    #[test]
    fn detect_attr_mismatch() {
        let server = HydrationNode::element("div").with_attr("class", "a");
        let client = HydrationNode::element("div").with_attr("class", "b");
        let mismatches = detect_mismatches(&server, &client);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(&mismatches[0], HydrationMismatch::AttrMismatch { .. }));
    }

    #[test]
    fn detect_child_count_mismatch() {
        let server = HydrationNode::element("div")
            .with_child(HydrationNode::text("a"));
        let client = HydrationNode::element("div")
            .with_child(HydrationNode::text("a"))
            .with_child(HydrationNode::text("b"));
        let mismatches = detect_mismatches(&server, &client);
        assert!(mismatches.iter().any(|m| matches!(m, HydrationMismatch::ChildCountMismatch { .. })));
    }

    #[test]
    fn detect_type_mismatch() {
        let server = HydrationNode::element("div");
        let client = HydrationNode::text("text");
        let mismatches = detect_mismatches(&server, &client);
        assert_eq!(mismatches.len(), 1);
        assert!(matches!(&mismatches[0], HydrationMismatch::TypeMismatch { .. }));
    }

    #[test]
    fn hydration_stats_compute() {
        let tree = sample_tree();
        let mismatches = vec![];
        let stats = HydrationStats::compute(&tree, &mismatches);
        assert_eq!(stats.total_nodes, 5); // div + h1 + "Hello" + button + "Click"
        assert_eq!(stats.interactive_nodes, 1); // button
        assert_eq!(stats.success_rate(), 1.0);
    }

    #[test]
    fn hydration_stats_with_mismatches() {
        let tree = sample_tree();
        let mismatches = vec![
            HydrationMismatch::TextMismatch {
                path: vec![0],
                server_text: "a".to_string(),
                client_text: "b".to_string(),
            },
        ];
        let stats = HydrationStats::compute(&tree, &mismatches);
        assert_eq!(stats.mismatch_count, 1);
        assert!(stats.success_rate() < 1.0);
    }

    #[test]
    fn hydration_hints_assigned() {
        let tree = sample_tree();
        let hints = assign_hydration_hints(&tree);
        assert!(!hints.is_empty());
        // Button (index 1 of root) should be Eager
        let button_hint = hints.iter().find(|(p, _)| *p == vec![1]);
        assert!(button_hint.is_some());
        assert_eq!(button_hint.unwrap().1, HydrationHint::Eager);
    }

    #[test]
    fn extract_interactive_subtrees_finds_button() {
        let tree = sample_tree();
        let subtrees = extract_interactive_subtrees(&tree);
        assert_eq!(subtrees.len(), 1);
        assert_eq!(subtrees[0].0, vec![1]); // button is at index 1
    }

    #[test]
    fn node_count() {
        let tree = sample_tree();
        assert_eq!(tree.node_count(), 5);
    }

    #[test]
    fn interactive_count() {
        let tree = sample_tree();
        assert_eq!(tree.interactive_count(), 1);
    }
}
