//! Breadcrumb Navigation: trail builder with collapse and SEO schema.
//!
//! Generates breadcrumb trails from URL paths, supports ellipsis-based
//! collapse for deep hierarchies, and emits BreadcrumbList structured data.

use serde_json::{json, Value};

// ── Core Types ──────────────────────────────────────────────────

/// A single breadcrumb entry.
#[derive(Debug, Clone, PartialEq)]
pub struct Breadcrumb {
    pub label: String,
    pub href: Option<String>,
    pub icon: Option<String>,
    pub active: bool,
}

/// What the UI should render for a visible slot.
#[derive(Debug, Clone, PartialEq)]
pub enum BreadcrumbView {
    Item(Breadcrumb),
    Ellipsis,
}

/// A complete breadcrumb trail.
#[derive(Debug, Clone)]
pub struct BreadcrumbTrail {
    items: Vec<Breadcrumb>,
    pub separator: String,
    pub max_items: Option<usize>,
    pub collapse_after: Option<usize>,
}

// ── Implementation ──────────────────────────────────────────────

impl BreadcrumbTrail {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            separator: "/".into(),
            max_items: None,
            collapse_after: None,
        }
    }

    pub fn push(&mut self, label: impl Into<String>, href: impl Into<String>) {
        if let Some(last) = self.items.last_mut() { last.active = false; }
        self.items.push(Breadcrumb {
            label: label.into(),
            href: Some(href.into()),
            icon: None,
            active: false,
        });
    }

    pub fn push_active(&mut self, label: impl Into<String>) {
        if let Some(last) = self.items.last_mut() { last.active = false; }
        self.items.push(Breadcrumb {
            label: label.into(),
            href: None,
            icon: None,
            active: true,
        });
    }

    pub fn pop(&mut self) -> Option<Breadcrumb> {
        let item = self.items.pop();
        if let Some(last) = self.items.last_mut() { last.active = true; }
        item
    }

    pub fn clear(&mut self) { self.items.clear(); }

    pub fn len(&self) -> usize { self.items.len() }
    pub fn is_empty(&self) -> bool { self.items.is_empty() }

    /// Build a trail from a URL path like "/a/b/c".
    pub fn from_path(path: &str) -> Self {
        let mut trail = Self::new();
        let segments: Vec<&str> = path.split('/')
            .filter(|s| !s.is_empty())
            .collect();
        let mut href_acc = String::new();
        for (i, seg) in segments.iter().enumerate() {
            href_acc.push('/');
            href_acc.push_str(seg);
            if i == segments.len() - 1 {
                trail.push_active(capitalize(seg));
            } else {
                trail.push(capitalize(seg), href_acc.clone());
            }
        }
        trail
    }

    /// Return visible items, collapsing middle items to an ellipsis
    /// if len() > collapse_after.
    pub fn visible_items(&self) -> Vec<BreadcrumbView> {
        let Some(collapse) = self.collapse_after else {
            return self.items.iter().cloned().map(BreadcrumbView::Item).collect();
        };
        let n = self.items.len();
        if n <= collapse {
            return self.items.iter().cloned().map(BreadcrumbView::Item).collect();
        }
        // Show first item, ellipsis, last (collapse - 1) items.
        let tail_count = collapse.saturating_sub(1).max(1);
        let mut out = Vec::new();
        out.push(BreadcrumbView::Item(self.items[0].clone()));
        out.push(BreadcrumbView::Ellipsis);
        for item in &self.items[n - tail_count..] {
            out.push(BreadcrumbView::Item(item.clone()));
        }
        out
    }

    /// Items as a slice.
    pub fn items(&self) -> &[Breadcrumb] { &self.items }

    /// Emit BreadcrumbList JSON-LD structured data for SEO.
    pub fn to_schema_json(&self) -> Value {
        let list: Vec<Value> = self.items.iter().enumerate().map(|(i, b)| {
            json!({
                "@type": "ListItem",
                "position": i + 1,
                "name": b.label,
                "item": b.href.as_deref().unwrap_or("")
            })
        }).collect();
        json!({
            "@context": "https://schema.org",
            "@type": "BreadcrumbList",
            "itemListElement": list
        })
    }
}

impl Default for BreadcrumbTrail {
    fn default() -> Self { Self::new() }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_path_basic() {
        let t = BreadcrumbTrail::from_path("/products/widgets/bolt");
        assert_eq!(t.len(), 3);
        assert_eq!(t.items[0].label, "Products");
        assert_eq!(t.items[0].href.as_deref(), Some("/products"));
        assert_eq!(t.items[1].href.as_deref(), Some("/products/widgets"));
        assert!(t.items[2].active);
        assert!(t.items[2].href.is_none());
    }

    #[test]
    fn push_pop() {
        let mut t = BreadcrumbTrail::new();
        t.push("Home", "/");
        t.push_active("About");
        assert_eq!(t.len(), 2);
        assert!(t.items[1].active);
        let popped = t.pop().unwrap();
        assert_eq!(popped.label, "About");
        assert!(t.items[0].active);
    }

    #[test]
    fn visible_with_collapse() {
        let mut t = BreadcrumbTrail::from_path("/a/b/c/d/e");
        t.collapse_after = Some(3);
        let vis = t.visible_items();
        assert_eq!(vis.len(), 4);
        assert!(matches!(&vis[0], BreadcrumbView::Item(b) if b.label == "A"));
        assert!(matches!(&vis[1], BreadcrumbView::Ellipsis));
        assert!(matches!(&vis[2], BreadcrumbView::Item(b) if b.label == "D"));
        assert!(matches!(&vis[3], BreadcrumbView::Item(b) if b.label == "E"));
    }

    #[test]
    fn no_collapse_when_short() {
        let mut t = BreadcrumbTrail::from_path("/a/b");
        t.collapse_after = Some(5);
        let vis = t.visible_items();
        assert_eq!(vis.len(), 2);
        assert!(!vis.iter().any(|v| matches!(v, BreadcrumbView::Ellipsis)));
    }

    #[test]
    fn schema_json() {
        let t = BreadcrumbTrail::from_path("/docs/api");
        let schema = t.to_schema_json();
        assert_eq!(schema["@type"], "BreadcrumbList");
        let elements = schema["itemListElement"].as_array().unwrap();
        assert_eq!(elements.len(), 2);
        assert_eq!(elements[0]["position"], 1);
        assert_eq!(elements[0]["name"], "Docs");
    }

    #[test]
    fn active_item() {
        let t = BreadcrumbTrail::from_path("/x/y");
        assert!(!t.items[0].active);
        assert!(t.items[1].active);
    }

    #[test]
    fn empty_trail() {
        let t = BreadcrumbTrail::new();
        assert!(t.is_empty());
        assert_eq!(t.len(), 0);
        assert_eq!(t.visible_items().len(), 0);
    }

    #[test]
    fn separator_custom() {
        let mut t = BreadcrumbTrail::new();
        t.separator = ">".into();
        t.push("A", "/a");
        assert_eq!(t.separator, ">");
    }

    #[test]
    fn clear_trail() {
        let mut t = BreadcrumbTrail::from_path("/a/b/c");
        assert_eq!(t.len(), 3);
        t.clear();
        assert!(t.is_empty());
    }
}
