//! Render tree construction — from virtual DOM to render objects with box model,
//! display types, paint order, and layer compositing.
//!
//! Replaces browser layout engine abstractions. Constructs a render tree from
//! a virtual DOM, assigns box model geometry (margin/border/padding/content),
//! resolves display types (block/inline/flex/none), and computes paint order
//! with layer compositing for z-index stacking contexts.

use std::collections::HashMap;

// ── Display types ───────────────────────────────────────────────────────

/// CSS display type for a render object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayType {
    Block,
    Inline,
    Flex,
    InlineBlock,
    None,
}

impl DisplayType {
    /// Parse from a CSS display value string.
    pub fn from_str_value(s: &str) -> Self {
        match s {
            "block" => DisplayType::Block,
            "inline" => DisplayType::Inline,
            "flex" => DisplayType::Flex,
            "inline-block" => DisplayType::InlineBlock,
            "none" => DisplayType::None,
            _ => DisplayType::Block,
        }
    }

    /// Whether this display type participates in layout.
    pub fn is_visible(&self) -> bool {
        *self != DisplayType::None
    }

    /// Whether this display generates a block-level box.
    pub fn is_block_level(&self) -> bool {
        matches!(self, DisplayType::Block | DisplayType::Flex)
    }
}

// ── Box model ───────────────────────────────────────────────────────────

/// Edge values (top, right, bottom, left) used for margin, border, padding.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EdgeSizes {
    pub top: f64,
    pub right: f64,
    pub bottom: f64,
    pub left: f64,
}

impl EdgeSizes {
    pub fn zero() -> Self {
        Self { top: 0.0, right: 0.0, bottom: 0.0, left: 0.0 }
    }

    pub fn uniform(v: f64) -> Self {
        Self { top: v, right: v, bottom: v, left: v }
    }

    pub fn horizontal(&self) -> f64 {
        self.left + self.right
    }

    pub fn vertical(&self) -> f64 {
        self.top + self.bottom
    }

    pub fn total(&self) -> f64 {
        self.horizontal() + self.vertical()
    }
}

impl Default for EdgeSizes {
    fn default() -> Self {
        Self::zero()
    }
}

/// Box model geometry for a render object.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxModel {
    pub content_width: f64,
    pub content_height: f64,
    pub padding: EdgeSizes,
    pub border: EdgeSizes,
    pub margin: EdgeSizes,
}

impl BoxModel {
    pub fn new(width: f64, height: f64) -> Self {
        Self {
            content_width: width,
            content_height: height,
            padding: EdgeSizes::zero(),
            border: EdgeSizes::zero(),
            margin: EdgeSizes::zero(),
        }
    }

    /// Width including padding and border.
    pub fn border_box_width(&self) -> f64 {
        self.content_width + self.padding.horizontal() + self.border.horizontal()
    }

    /// Height including padding and border.
    pub fn border_box_height(&self) -> f64 {
        self.content_height + self.padding.vertical() + self.border.vertical()
    }

    /// Width including padding, border, and margin.
    pub fn margin_box_width(&self) -> f64 {
        self.border_box_width() + self.margin.horizontal()
    }

    /// Height including padding, border, and margin.
    pub fn margin_box_height(&self) -> f64 {
        self.border_box_height() + self.margin.vertical()
    }

    /// The padding box (content + padding).
    pub fn padding_box_width(&self) -> f64 {
        self.content_width + self.padding.horizontal()
    }

    pub fn padding_box_height(&self) -> f64 {
        self.content_height + self.padding.vertical()
    }
}

impl Default for BoxModel {
    fn default() -> Self {
        Self::new(0.0, 0.0)
    }
}

// ── Render object ───────────────────────────────────────────────────────

/// Position of a render object in the coordinate system.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

impl Position {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    pub fn origin() -> Self {
        Self { x: 0.0, y: 0.0 }
    }

    pub fn offset(&self, dx: f64, dy: f64) -> Self {
        Self { x: self.x + dx, y: self.y + dy }
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::origin()
    }
}

/// A render object — one node in the render tree.
#[derive(Debug, Clone)]
pub struct RenderObject {
    pub id: u64,
    pub tag: String,
    pub display: DisplayType,
    pub box_model: BoxModel,
    pub position: Position,
    pub z_index: i32,
    pub opacity: f64,
    pub creates_stacking_context: bool,
    pub children: Vec<RenderObject>,
    pub attributes: HashMap<String, String>,
}

impl RenderObject {
    pub fn new(id: u64, tag: &str) -> Self {
        Self {
            id,
            tag: tag.to_string(),
            display: DisplayType::Block,
            box_model: BoxModel::default(),
            position: Position::origin(),
            z_index: 0,
            opacity: 1.0,
            creates_stacking_context: false,
            children: Vec::new(),
            attributes: HashMap::new(),
        }
    }

    pub fn with_display(mut self, display: DisplayType) -> Self {
        self.display = display;
        self
    }

    pub fn with_box(mut self, bm: BoxModel) -> Self {
        self.box_model = bm;
        self
    }

    pub fn with_position(mut self, pos: Position) -> Self {
        self.position = pos;
        self
    }

    pub fn with_z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        if z != 0 {
            self.creates_stacking_context = true;
        }
        self
    }

    pub fn with_opacity(mut self, opacity: f64) -> Self {
        self.opacity = opacity.clamp(0.0, 1.0);
        if self.opacity < 1.0 {
            self.creates_stacking_context = true;
        }
        self
    }

    pub fn with_child(mut self, child: RenderObject) -> Self {
        self.children.push(child);
        self
    }

    /// Check if this object is visible (display != None and opacity > 0).
    pub fn is_visible(&self) -> bool {
        self.display.is_visible() && self.opacity > 0.0
    }

    /// Visible children (filtered by display != none).
    pub fn visible_children(&self) -> Vec<&RenderObject> {
        self.children.iter().filter(|c| c.is_visible()).collect()
    }

    /// Total node count in this subtree.
    pub fn node_count(&self) -> usize {
        1 + self.children.iter().map(|c| c.node_count()).sum::<usize>()
    }
}

// ── Render tree builder ─────────────────────────────────────────────────

/// Input node for building a render tree.
#[derive(Debug, Clone)]
pub struct InputNode {
    pub tag: String,
    pub display: DisplayType,
    pub width: f64,
    pub height: f64,
    pub padding: EdgeSizes,
    pub border: EdgeSizes,
    pub margin: EdgeSizes,
    pub z_index: i32,
    pub opacity: f64,
    pub children: Vec<InputNode>,
}

impl InputNode {
    pub fn new(tag: &str, width: f64, height: f64) -> Self {
        Self {
            tag: tag.to_string(),
            display: DisplayType::Block,
            width,
            height,
            padding: EdgeSizes::zero(),
            border: EdgeSizes::zero(),
            margin: EdgeSizes::zero(),
            z_index: 0,
            opacity: 1.0,
            children: Vec::new(),
        }
    }

    pub fn with_display(mut self, d: DisplayType) -> Self {
        self.display = d;
        self
    }

    pub fn with_padding(mut self, p: EdgeSizes) -> Self {
        self.padding = p;
        self
    }

    pub fn with_border(mut self, b: EdgeSizes) -> Self {
        self.border = b;
        self
    }

    pub fn with_margin(mut self, m: EdgeSizes) -> Self {
        self.margin = m;
        self
    }

    pub fn with_z_index(mut self, z: i32) -> Self {
        self.z_index = z;
        self
    }

    pub fn with_opacity(mut self, o: f64) -> Self {
        self.opacity = o;
        self
    }

    pub fn with_child(mut self, child: InputNode) -> Self {
        self.children.push(child);
        self
    }
}

/// Build a render tree from an input node tree.
pub fn build_render_tree(input: &InputNode) -> RenderObject {
    let mut next_id = 0u64;
    build_recursive(input, &mut next_id, Position::origin())
}

fn build_recursive(input: &InputNode, next_id: &mut u64, offset: Position) -> RenderObject {
    let id = *next_id;
    *next_id += 1;

    let bm = BoxModel {
        content_width: input.width,
        content_height: input.height,
        padding: input.padding,
        border: input.border,
        margin: input.margin,
    };

    let pos = offset.offset(input.margin.left, input.margin.top);

    let mut obj = RenderObject::new(id, &input.tag)
        .with_display(input.display)
        .with_box(bm)
        .with_position(pos)
        .with_z_index(input.z_index)
        .with_opacity(input.opacity);

    // Layout children (simple block stacking)
    let content_start = Position::new(
        pos.x + input.border.left + input.padding.left,
        pos.y + input.border.top + input.padding.top,
    );

    let mut child_y = content_start.y;
    for child_input in &input.children {
        if child_input.display == DisplayType::None {
            let child = build_recursive(child_input, next_id, Position::new(content_start.x, child_y));
            obj.children.push(child);
            continue;
        }

        let child_offset = Position::new(content_start.x, child_y);
        let child = build_recursive(child_input, next_id, child_offset);

        if child_input.display.is_block_level() {
            child_y += child.box_model.margin_box_height();
        }

        obj.children.push(child);
    }

    obj
}

// ── Paint order ─────────────────────────────────────────────────────────

/// A paint command in z-order.
#[derive(Debug, Clone)]
pub struct PaintEntry {
    pub render_id: u64,
    pub tag: String,
    pub z_index: i32,
    pub layer: u32,
    pub position: Position,
}

/// Traverse the render tree and produce a paint-ordered list.
pub fn compute_paint_order(root: &RenderObject) -> Vec<PaintEntry> {
    let mut entries = Vec::new();
    collect_paint_entries(root, 0, &mut entries);
    // Sort by layer, then z-index, then id for stability
    entries.sort_by(|a, b| {
        a.layer.cmp(&b.layer)
            .then(a.z_index.cmp(&b.z_index))
            .then(a.render_id.cmp(&b.render_id))
    });
    entries
}

fn collect_paint_entries(obj: &RenderObject, layer: u32, entries: &mut Vec<PaintEntry>) {
    if !obj.is_visible() {
        return;
    }

    let current_layer = if obj.creates_stacking_context { layer + 1 } else { layer };

    entries.push(PaintEntry {
        render_id: obj.id,
        tag: obj.tag.clone(),
        z_index: obj.z_index,
        layer: current_layer,
        position: obj.position,
    });

    for child in &obj.children {
        collect_paint_entries(child, current_layer, entries);
    }
}

/// Collect all stacking contexts in the render tree.
pub fn collect_stacking_contexts(root: &RenderObject) -> Vec<u64> {
    let mut ids = Vec::new();
    collect_stacking_recursive(root, &mut ids);
    ids
}

fn collect_stacking_recursive(obj: &RenderObject, ids: &mut Vec<u64>) {
    if obj.creates_stacking_context {
        ids.push(obj.id);
    }
    for child in &obj.children {
        collect_stacking_recursive(child, ids);
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edge_sizes_uniform() {
        let e = EdgeSizes::uniform(10.0);
        assert_eq!(e.horizontal(), 20.0);
        assert_eq!(e.vertical(), 20.0);
        assert_eq!(e.total(), 40.0);
    }

    #[test]
    fn edge_sizes_zero() {
        let e = EdgeSizes::zero();
        assert_eq!(e.total(), 0.0);
    }

    #[test]
    fn box_model_content_only() {
        let bm = BoxModel::new(100.0, 50.0);
        assert_eq!(bm.border_box_width(), 100.0);
        assert_eq!(bm.border_box_height(), 50.0);
        assert_eq!(bm.margin_box_width(), 100.0);
        assert_eq!(bm.margin_box_height(), 50.0);
    }

    #[test]
    fn box_model_with_padding() {
        let mut bm = BoxModel::new(100.0, 50.0);
        bm.padding = EdgeSizes::uniform(10.0);
        assert_eq!(bm.padding_box_width(), 120.0);
        assert_eq!(bm.padding_box_height(), 70.0);
        assert_eq!(bm.border_box_width(), 120.0);
    }

    #[test]
    fn box_model_full() {
        let mut bm = BoxModel::new(100.0, 50.0);
        bm.padding = EdgeSizes::uniform(5.0);
        bm.border = EdgeSizes::uniform(2.0);
        bm.margin = EdgeSizes::uniform(10.0);
        assert_eq!(bm.border_box_width(), 114.0);
        assert_eq!(bm.border_box_height(), 64.0);
        assert_eq!(bm.margin_box_width(), 134.0);
        assert_eq!(bm.margin_box_height(), 84.0);
    }

    #[test]
    fn display_type_from_str() {
        assert_eq!(DisplayType::from_str_value("block"), DisplayType::Block);
        assert_eq!(DisplayType::from_str_value("inline"), DisplayType::Inline);
        assert_eq!(DisplayType::from_str_value("flex"), DisplayType::Flex);
        assert_eq!(DisplayType::from_str_value("none"), DisplayType::None);
        assert_eq!(DisplayType::from_str_value("unknown"), DisplayType::Block);
    }

    #[test]
    fn display_type_visibility() {
        assert!(DisplayType::Block.is_visible());
        assert!(DisplayType::Inline.is_visible());
        assert!(!DisplayType::None.is_visible());
    }

    #[test]
    fn display_type_block_level() {
        assert!(DisplayType::Block.is_block_level());
        assert!(DisplayType::Flex.is_block_level());
        assert!(!DisplayType::Inline.is_block_level());
    }

    #[test]
    fn render_object_visibility() {
        let obj = RenderObject::new(0, "div");
        assert!(obj.is_visible());

        let hidden = RenderObject::new(1, "div").with_display(DisplayType::None);
        assert!(!hidden.is_visible());

        let transparent = RenderObject::new(2, "div").with_opacity(0.0);
        assert!(!transparent.is_visible());
    }

    #[test]
    fn render_object_stacking_context() {
        let obj = RenderObject::new(0, "div").with_z_index(5);
        assert!(obj.creates_stacking_context);

        let obj2 = RenderObject::new(1, "div").with_opacity(0.5);
        assert!(obj2.creates_stacking_context);

        let obj3 = RenderObject::new(2, "div");
        assert!(!obj3.creates_stacking_context);
    }

    #[test]
    fn build_simple_render_tree() {
        let input = InputNode::new("div", 200.0, 100.0)
            .with_child(InputNode::new("p", 180.0, 20.0))
            .with_child(InputNode::new("p", 180.0, 20.0));
        let tree = build_render_tree(&input);
        assert_eq!(tree.node_count(), 3);
        assert_eq!(tree.children.len(), 2);
    }

    #[test]
    fn build_tree_with_box_model() {
        let input = InputNode::new("div", 200.0, 100.0)
            .with_padding(EdgeSizes::uniform(10.0))
            .with_border(EdgeSizes::uniform(2.0))
            .with_margin(EdgeSizes::uniform(5.0));
        let tree = build_render_tree(&input);
        assert_eq!(tree.box_model.content_width, 200.0);
        assert_eq!(tree.box_model.border_box_width(), 224.0);
        assert_eq!(tree.box_model.margin_box_width(), 234.0);
    }

    #[test]
    fn paint_order_simple() {
        let input = InputNode::new("div", 200.0, 100.0)
            .with_child(InputNode::new("span", 50.0, 20.0));
        let tree = build_render_tree(&input);
        let paint = compute_paint_order(&tree);
        assert_eq!(paint.len(), 2);
    }

    #[test]
    fn paint_order_respects_z_index() {
        let input = InputNode::new("div", 200.0, 100.0)
            .with_child(InputNode::new("a", 50.0, 20.0).with_z_index(10))
            .with_child(InputNode::new("b", 50.0, 20.0).with_z_index(1));
        let tree = build_render_tree(&input);
        let paint = compute_paint_order(&tree);
        // The parent is layer 0 z=0. Children create stacking contexts (layer 1).
        // b (z=1) should come before a (z=10) in paint order.
        let child_entries: Vec<_> = paint.iter().filter(|e| e.tag == "a" || e.tag == "b").collect();
        assert_eq!(child_entries.len(), 2);
        assert!(child_entries[0].z_index <= child_entries[1].z_index);
    }

    #[test]
    fn hidden_nodes_excluded_from_paint() {
        let input = InputNode::new("div", 200.0, 100.0)
            .with_child(InputNode::new("hidden", 50.0, 20.0).with_display(DisplayType::None));
        let tree = build_render_tree(&input);
        let paint = compute_paint_order(&tree);
        assert_eq!(paint.len(), 1); // only the parent
    }

    #[test]
    fn stacking_contexts_collected() {
        let input = InputNode::new("div", 200.0, 100.0)
            .with_child(InputNode::new("a", 50.0, 20.0).with_z_index(1))
            .with_child(InputNode::new("b", 50.0, 20.0).with_opacity(0.5));
        let tree = build_render_tree(&input);
        let contexts = collect_stacking_contexts(&tree);
        assert_eq!(contexts.len(), 2);
    }

    #[test]
    fn position_offset() {
        let p = Position::new(10.0, 20.0);
        let p2 = p.offset(5.0, -3.0);
        assert_eq!(p2.x, 15.0);
        assert_eq!(p2.y, 17.0);
    }

    #[test]
    fn visible_children_filter() {
        let parent = RenderObject::new(0, "div")
            .with_child(RenderObject::new(1, "a"))
            .with_child(RenderObject::new(2, "b").with_display(DisplayType::None))
            .with_child(RenderObject::new(3, "c"));
        let vis = parent.visible_children();
        assert_eq!(vis.len(), 2);
    }

    #[test]
    fn block_children_stack_vertically() {
        let input = InputNode::new("div", 200.0, 100.0)
            .with_child(InputNode::new("p", 180.0, 30.0))
            .with_child(InputNode::new("p", 180.0, 30.0));
        let tree = build_render_tree(&input);
        let c0 = &tree.children[0];
        let c1 = &tree.children[1];
        // Second child should start below the first
        assert!(c1.position.y > c0.position.y);
    }
}
