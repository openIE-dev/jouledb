//! DOM/component inspector model with node trees, selectors, layout overlays, and editing.
//!
//! Provides a pure-Rust representation of an inspectable element tree — like the
//! browser DevTools Elements panel — with search, breadcrumb navigation, computed
//! styles, and layout box information.

use std::collections::HashMap;

// ── Types ──

/// A rectangle (x, y, width, height) for layout overlays.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn area(&self) -> f64 {
        self.width * self.height
    }
}

/// Layout overlay information for the CSS box model.
#[derive(Debug, Clone)]
pub struct LayoutOverlay {
    pub content_box: Rect,
    pub padding_box: Rect,
    pub border_box: Rect,
    pub margin_box: Rect,
}

impl LayoutOverlay {
    pub fn from_content(
        content: Rect,
        padding: f64,
        border: f64,
        margin: f64,
    ) -> Self {
        let padding_box = Rect::new(
            content.x - padding,
            content.y - padding,
            content.width + 2.0 * padding,
            content.height + 2.0 * padding,
        );
        let border_box = Rect::new(
            padding_box.x - border,
            padding_box.y - border,
            padding_box.width + 2.0 * border,
            padding_box.height + 2.0 * border,
        );
        let margin_box = Rect::new(
            border_box.x - margin,
            border_box.y - margin,
            border_box.width + 2.0 * margin,
            border_box.height + 2.0 * margin,
        );
        Self {
            content_box: content,
            padding_box,
            border_box,
            margin_box,
        }
    }
}

/// A path of child indices to reach a node in the tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreePath(pub Vec<usize>);

impl TreePath {
    pub fn root() -> Self {
        Self(Vec::new())
    }

    pub fn child(&self, index: usize) -> Self {
        let mut p = self.0.clone();
        p.push(index);
        Self(p)
    }

    pub fn depth(&self) -> usize {
        self.0.len()
    }
}

/// An inspector node representing a DOM element or component.
#[derive(Debug, Clone)]
pub struct InspectorNode {
    pub tag: String,
    pub id: Option<String>,
    pub classes: Vec<String>,
    pub attributes: HashMap<String, String>,
    pub styles_computed: HashMap<String, String>,
    pub text_content: Option<String>,
    pub children: Vec<InspectorNode>,
    pub layout: Option<LayoutOverlay>,
}

impl InspectorNode {
    pub fn new(tag: &str) -> Self {
        Self {
            tag: tag.to_string(),
            id: None,
            classes: Vec::new(),
            attributes: HashMap::new(),
            styles_computed: HashMap::new(),
            text_content: None,
            children: Vec::new(),
            layout: None,
        }
    }

    pub fn with_id(mut self, id: &str) -> Self {
        self.id = Some(id.to_string());
        self
    }

    pub fn with_class(mut self, class: &str) -> Self {
        self.classes.push(class.to_string());
        self
    }

    pub fn with_attr(mut self, key: &str, value: &str) -> Self {
        self.attributes.insert(key.to_string(), value.to_string());
        self
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.text_content = Some(text.to_string());
        self
    }

    pub fn with_child(mut self, child: InspectorNode) -> Self {
        self.children.push(child);
        self
    }

    pub fn with_style(mut self, prop: &str, value: &str) -> Self {
        self.styles_computed
            .insert(prop.to_string(), value.to_string());
        self
    }

    pub fn with_layout(mut self, layout: LayoutOverlay) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Add a child node.
    pub fn add_child(&mut self, child: InspectorNode) {
        self.children.push(child);
    }

    /// Select a descendant node by tree path.
    pub fn select(&self, path: &TreePath) -> Option<&InspectorNode> {
        let mut current = self;
        for &idx in &path.0 {
            current = current.children.get(idx)?;
        }
        Some(current)
    }

    /// Select a mutable descendant node by tree path.
    pub fn select_mut(&mut self, path: &TreePath) -> Option<&mut InspectorNode> {
        let mut current = self;
        for &idx in &path.0 {
            current = current.children.get_mut(idx)?;
        }
        Some(current)
    }

    /// Compute breadcrumb (ancestor chain) from root to the node at `path`.
    pub fn breadcrumb(&self, path: &TreePath) -> Vec<String> {
        let mut crumbs = vec![self.display_tag()];
        let mut current = self;
        for &idx in &path.0 {
            if let Some(child) = current.children.get(idx) {
                crumbs.push(child.display_tag());
                current = child;
            } else {
                break;
            }
        }
        crumbs
    }

    /// Display tag with id/class for breadcrumb.
    pub fn display_tag(&self) -> String {
        let mut s = self.tag.clone();
        if let Some(id) = &self.id {
            s.push('#');
            s.push_str(id);
        }
        for cls in &self.classes {
            s.push('.');
            s.push_str(cls);
        }
        s
    }

    /// Search nodes by tag name, returning paths to matching nodes.
    pub fn search_by_tag(&self, tag: &str) -> Vec<TreePath> {
        let mut results = Vec::new();
        self.search_by_tag_inner(tag, &mut Vec::new(), &mut results);
        results
    }

    fn search_by_tag_inner(
        &self,
        tag: &str,
        current_path: &mut Vec<usize>,
        results: &mut Vec<TreePath>,
    ) {
        if self.tag == tag {
            results.push(TreePath(current_path.clone()));
        }
        for (i, child) in self.children.iter().enumerate() {
            current_path.push(i);
            child.search_by_tag_inner(tag, current_path, results);
            current_path.pop();
        }
    }

    /// Search nodes by text content substring.
    pub fn search_by_text(&self, text: &str) -> Vec<TreePath> {
        let mut results = Vec::new();
        self.search_by_text_inner(text, &mut Vec::new(), &mut results);
        results
    }

    fn search_by_text_inner(
        &self,
        text: &str,
        current_path: &mut Vec<usize>,
        results: &mut Vec<TreePath>,
    ) {
        if let Some(tc) = &self.text_content {
            if tc.contains(text) {
                results.push(TreePath(current_path.clone()));
            }
        }
        for (i, child) in self.children.iter().enumerate() {
            current_path.push(i);
            child.search_by_text_inner(text, current_path, results);
            current_path.pop();
        }
    }

    /// Edit an attribute on a node at the given path.
    pub fn edit_attribute(
        &mut self,
        path: &TreePath,
        key: &str,
        value: &str,
    ) -> bool {
        if let Some(node) = self.select_mut(path) {
            node.attributes.insert(key.to_string(), value.to_string());
            true
        } else {
            false
        }
    }

    /// Edit a computed style on a node at the given path.
    pub fn edit_style(
        &mut self,
        path: &TreePath,
        prop: &str,
        value: &str,
    ) -> bool {
        if let Some(node) = self.select_mut(path) {
            node.styles_computed
                .insert(prop.to_string(), value.to_string());
            true
        } else {
            false
        }
    }

    /// Count total nodes in the tree.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tree() -> InspectorNode {
        InspectorNode::new("div")
            .with_id("root")
            .with_class("container")
            .with_child(
                InspectorNode::new("header")
                    .with_child(InspectorNode::new("h1").with_text("Title")),
            )
            .with_child(
                InspectorNode::new("main")
                    .with_child(InspectorNode::new("p").with_text("Hello world"))
                    .with_child(InspectorNode::new("p").with_text("Second paragraph")),
            )
            .with_child(InspectorNode::new("footer").with_text("Copyright"))
    }

    #[test]
    fn test_node_creation() {
        let node = InspectorNode::new("div").with_id("app").with_class("main");
        assert_eq!(node.tag, "div");
        assert_eq!(node.id.as_deref(), Some("app"));
        assert_eq!(node.classes, vec!["main"]);
    }

    #[test]
    fn test_tree_path_root() {
        let tree = sample_tree();
        let root = tree.select(&TreePath::root()).unwrap();
        assert_eq!(root.tag, "div");
    }

    #[test]
    fn test_select_by_path() {
        let tree = sample_tree();
        // header -> h1
        let path = TreePath(vec![0, 0]);
        let node = tree.select(&path).unwrap();
        assert_eq!(node.tag, "h1");
        assert_eq!(node.text_content.as_deref(), Some("Title"));
    }

    #[test]
    fn test_select_invalid_path() {
        let tree = sample_tree();
        assert!(tree.select(&TreePath(vec![99])).is_none());
    }

    #[test]
    fn test_breadcrumb() {
        let tree = sample_tree();
        let path = TreePath(vec![1, 0]); // main -> p
        let crumbs = tree.breadcrumb(&path);
        assert_eq!(crumbs, vec!["div#root.container", "main", "p"]);
    }

    #[test]
    fn test_search_by_tag() {
        let tree = sample_tree();
        let results = tree.search_by_tag("p");
        assert_eq!(results.len(), 2);
        let node = tree.select(&results[0]).unwrap();
        assert_eq!(node.text_content.as_deref(), Some("Hello world"));
    }

    #[test]
    fn test_search_by_text() {
        let tree = sample_tree();
        let results = tree.search_by_text("paragraph");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_edit_attribute() {
        let mut tree = sample_tree();
        let path = TreePath(vec![0]); // header
        assert!(tree.edit_attribute(&path, "role", "banner"));
        let node = tree.select(&path).unwrap();
        assert_eq!(node.attributes.get("role").unwrap(), "banner");
    }

    #[test]
    fn test_edit_style() {
        let mut tree = sample_tree();
        let path = TreePath::root();
        assert!(tree.edit_style(&path, "color", "red"));
        let node = tree.select(&path).unwrap();
        assert_eq!(node.styles_computed.get("color").unwrap(), "red");
    }

    #[test]
    fn test_layout_overlay() {
        let content = Rect::new(100.0, 100.0, 200.0, 50.0);
        let layout = LayoutOverlay::from_content(content, 10.0, 2.0, 8.0);
        assert_eq!(layout.padding_box.x, 90.0);
        assert_eq!(layout.padding_box.width, 220.0);
        assert_eq!(layout.border_box.x, 88.0);
        assert_eq!(layout.border_box.width, 224.0);
        assert_eq!(layout.margin_box.x, 80.0);
        assert_eq!(layout.margin_box.width, 240.0);
    }

    #[test]
    fn test_node_count() {
        let tree = sample_tree();
        // div(root) + header + h1 + main + p + p + footer = 7
        assert_eq!(tree.node_count(), 7);
    }

    #[test]
    fn test_display_tag() {
        let node = InspectorNode::new("div")
            .with_id("app")
            .with_class("flex")
            .with_class("dark");
        assert_eq!(node.display_tag(), "div#app.flex.dark");
    }

    #[test]
    fn test_rect_area() {
        let r = Rect::new(0.0, 0.0, 100.0, 50.0);
        assert!((r.area() - 5000.0).abs() < f64::EPSILON);
    }
}
