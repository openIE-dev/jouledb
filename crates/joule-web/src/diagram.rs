//! Diagram canvas model for shape-based diagramming.
//!
//! Replaces draw.io / Excalidraw core model. Provides shapes, connectors,
//! hit testing, and canvas operations. Pure Rust — no browser dependency.

use std::collections::HashMap;

// ── Shape types ──────────────────────────────────────────────────

/// Geometric shape primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Shape {
    Rect,
    RoundedRect,
    Ellipse,
    Diamond,
    Parallelogram,
    Hexagon,
    Cylinder,
    Cloud,
}

/// 2D position.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Position {
    pub x: f64,
    pub y: f64,
}

impl Position {
    pub fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }
}

/// 2D size.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Size {
    pub width: f64,
    pub height: f64,
}

impl Size {
    pub fn new(width: f64, height: f64) -> Self {
        Self { width, height }
    }
}

/// Visual style for shapes.
#[derive(Debug, Clone)]
pub struct ShapeStyle {
    pub fill: String,
    pub stroke: String,
    pub stroke_width: f64,
    pub opacity: f64,
    pub corner_radius: f64,
}

impl Default for ShapeStyle {
    fn default() -> Self {
        Self {
            fill: "#ffffff".to_string(),
            stroke: "#333333".to_string(),
            stroke_width: 2.0,
            opacity: 1.0,
            corner_radius: 8.0,
        }
    }
}

/// An instance of a shape on the canvas.
#[derive(Debug, Clone)]
pub struct ShapeInstance {
    pub id: String,
    pub shape: Shape,
    pub position: Position,
    pub size: Size,
    pub style: ShapeStyle,
    pub label: Option<String>,
}

impl ShapeInstance {
    pub fn new(
        id: impl Into<String>,
        shape: Shape,
        position: Position,
        size: Size,
    ) -> Self {
        Self {
            id: id.into(),
            shape,
            position,
            size,
            style: ShapeStyle::default(),
            label: None,
        }
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    pub fn with_style(mut self, style: ShapeStyle) -> Self {
        self.style = style;
        self
    }

    pub fn center(&self) -> Position {
        Position {
            x: self.position.x + self.size.width / 2.0,
            y: self.position.y + self.size.height / 2.0,
        }
    }

    /// Hit test: is a point inside this shape?
    pub fn contains_point(&self, px: f64, py: f64) -> bool {
        let cx = self.position.x + self.size.width / 2.0;
        let cy = self.position.y + self.size.height / 2.0;
        let hw = self.size.width / 2.0;
        let hh = self.size.height / 2.0;

        match self.shape {
            Shape::Rect | Shape::RoundedRect | Shape::Cylinder => {
                px >= self.position.x
                    && px <= self.position.x + self.size.width
                    && py >= self.position.y
                    && py <= self.position.y + self.size.height
            }
            Shape::Ellipse | Shape::Cloud => {
                let dx = (px - cx) / hw;
                let dy = (py - cy) / hh;
                dx * dx + dy * dy <= 1.0
            }
            Shape::Diamond => {
                let dx = (px - cx).abs() / hw;
                let dy = (py - cy).abs() / hh;
                dx + dy <= 1.0
            }
            Shape::Parallelogram => {
                // Skew factor: 25% of width.
                let skew = self.size.width * 0.25;
                let local_x = px - self.position.x;
                let local_y = py - self.position.y;
                if local_y < 0.0 || local_y > self.size.height {
                    return false;
                }
                let t = local_y / self.size.height;
                let left = skew * (1.0 - t);
                let right = self.size.width - skew * t;
                local_x >= left && local_x <= right
            }
            Shape::Hexagon => {
                // Regular hexagon inscribed in bounding box.
                let dx = (px - cx).abs() / hw;
                let dy = (py - cy).abs() / hh;
                dx <= 1.0 && dy <= 1.0 && (dx + dy * 0.5) <= 1.0
            }
        }
    }
}

// ── Connector ────────────────────────────────────────────────────

/// Path style for connectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathStyle {
    Straight,
    Curved,
    Orthogonal,
}

/// Arrowhead type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arrowhead {
    None,
    Triangle,
    Diamond,
    Circle,
}

/// A connector between two shapes.
#[derive(Debug, Clone)]
pub struct Connector {
    pub id: String,
    pub source_id: String,
    pub target_id: String,
    pub path_style: PathStyle,
    pub arrowhead: Arrowhead,
    pub label: Option<String>,
}

impl Connector {
    pub fn new(
        id: impl Into<String>,
        source_id: impl Into<String>,
        target_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            source_id: source_id.into(),
            target_id: target_id.into(),
            path_style: PathStyle::Straight,
            arrowhead: Arrowhead::Triangle,
            label: None,
        }
    }

    pub fn with_path_style(mut self, style: PathStyle) -> Self {
        self.path_style = style;
        self
    }

    pub fn with_arrowhead(mut self, arrowhead: Arrowhead) -> Self {
        self.arrowhead = arrowhead;
        self
    }

    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ── Canvas ───────────────────────────────────────────────────────

/// The diagram canvas containing shapes and connectors.
#[derive(Debug, Clone)]
pub struct Canvas {
    shapes: HashMap<String, ShapeInstance>,
    connectors: Vec<Connector>,
    insertion_order: Vec<String>,
}

impl Canvas {
    pub fn new() -> Self {
        Self {
            shapes: HashMap::new(),
            connectors: Vec::new(),
            insertion_order: Vec::new(),
        }
    }

    pub fn add_shape(&mut self, shape: ShapeInstance) {
        let id = shape.id.clone();
        self.shapes.insert(id.clone(), shape);
        self.insertion_order.push(id);
    }

    pub fn get_shape(&self, id: &str) -> Option<&ShapeInstance> {
        self.shapes.get(id)
    }

    pub fn get_shape_mut(&mut self, id: &str) -> Option<&mut ShapeInstance> {
        self.shapes.get_mut(id)
    }

    pub fn remove_shape(&mut self, id: &str) -> Option<ShapeInstance> {
        self.insertion_order.retain(|s| s != id);
        // Remove connectors referencing this shape.
        self.connectors
            .retain(|c| c.source_id != id && c.target_id != id);
        self.shapes.remove(id)
    }

    pub fn move_shape(&mut self, id: &str, new_pos: Position) -> bool {
        if let Some(shape) = self.shapes.get_mut(id) {
            shape.position = new_pos;
            true
        } else {
            false
        }
    }

    pub fn resize_shape(&mut self, id: &str, new_size: Size) -> bool {
        if let Some(shape) = self.shapes.get_mut(id) {
            shape.size = new_size;
            true
        } else {
            false
        }
    }

    pub fn add_connector(&mut self, connector: Connector) {
        self.connectors.push(connector);
    }

    pub fn connectors(&self) -> &[Connector] {
        &self.connectors
    }

    pub fn shape_count(&self) -> usize {
        self.shapes.len()
    }

    pub fn shapes(&self) -> impl Iterator<Item = &ShapeInstance> {
        self.insertion_order
            .iter()
            .filter_map(|id| self.shapes.get(id))
    }

    /// Find the topmost shape at a given point.
    pub fn hit_test(&self, x: f64, y: f64) -> Option<&ShapeInstance> {
        // Iterate in reverse insertion order (topmost first).
        for id in self.insertion_order.iter().rev() {
            if let Some(shape) = self.shapes.get(id) {
                if shape.contains_point(x, y) {
                    return Some(shape);
                }
            }
        }
        None
    }

    /// Find all shapes intersecting a rectangular region.
    pub fn shapes_in_region(
        &self,
        rx: f64,
        ry: f64,
        rw: f64,
        rh: f64,
    ) -> Vec<&ShapeInstance> {
        self.shapes
            .values()
            .filter(|s| {
                let sx = s.position.x;
                let sy = s.position.y;
                let sw = s.size.width;
                let sh = s.size.height;
                sx < rx + rw && sx + sw > rx && sy < ry + rh && sy + sh > ry
            })
            .collect()
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn rect_shape(id: &str, x: f64, y: f64) -> ShapeInstance {
        ShapeInstance::new(id, Shape::Rect, Position::new(x, y), Size::new(100.0, 60.0))
    }

    #[test]
    fn test_add_and_get_shape() {
        let mut canvas = Canvas::new();
        canvas.add_shape(rect_shape("s1", 10.0, 20.0));
        assert_eq!(canvas.shape_count(), 1);
        assert!(canvas.get_shape("s1").is_some());
    }

    #[test]
    fn test_remove_shape() {
        let mut canvas = Canvas::new();
        canvas.add_shape(rect_shape("s1", 0.0, 0.0));
        canvas.add_shape(rect_shape("s2", 200.0, 0.0));
        canvas.add_connector(Connector::new("c1", "s1", "s2"));
        canvas.remove_shape("s1");
        assert_eq!(canvas.shape_count(), 1);
        assert!(canvas.connectors().is_empty());
    }

    #[test]
    fn test_move_shape() {
        let mut canvas = Canvas::new();
        canvas.add_shape(rect_shape("s1", 0.0, 0.0));
        assert!(canvas.move_shape("s1", Position::new(50.0, 75.0)));
        let s = canvas.get_shape("s1").unwrap();
        assert_eq!(s.position.x, 50.0);
        assert_eq!(s.position.y, 75.0);
    }

    #[test]
    fn test_resize_shape() {
        let mut canvas = Canvas::new();
        canvas.add_shape(rect_shape("s1", 0.0, 0.0));
        assert!(canvas.resize_shape("s1", Size::new(200.0, 150.0)));
        let s = canvas.get_shape("s1").unwrap();
        assert_eq!(s.size.width, 200.0);
    }

    #[test]
    fn test_hit_test_rect() {
        let mut canvas = Canvas::new();
        canvas.add_shape(rect_shape("s1", 10.0, 10.0));
        assert!(canvas.hit_test(50.0, 40.0).is_some());
        assert!(canvas.hit_test(0.0, 0.0).is_none());
    }

    #[test]
    fn test_hit_test_ellipse() {
        let shape = ShapeInstance::new(
            "e1",
            Shape::Ellipse,
            Position::new(0.0, 0.0),
            Size::new(100.0, 60.0),
        );
        // Center should be inside.
        assert!(shape.contains_point(50.0, 30.0));
        // Corner of bounding box should be outside ellipse.
        assert!(!shape.contains_point(0.0, 0.0));
    }

    #[test]
    fn test_hit_test_diamond() {
        let shape = ShapeInstance::new(
            "d1",
            Shape::Diamond,
            Position::new(0.0, 0.0),
            Size::new(100.0, 100.0),
        );
        // Center is inside.
        assert!(shape.contains_point(50.0, 50.0));
        // Corner of bounding box is outside diamond.
        assert!(!shape.contains_point(1.0, 1.0));
    }

    #[test]
    fn test_hit_test_topmost() {
        let mut canvas = Canvas::new();
        canvas.add_shape(rect_shape("bottom", 0.0, 0.0));
        canvas.add_shape(rect_shape("top", 0.0, 0.0));
        let hit = canvas.hit_test(50.0, 30.0).unwrap();
        assert_eq!(hit.id, "top");
    }

    #[test]
    fn test_connector_builder() {
        let c = Connector::new("c1", "a", "b")
            .with_path_style(PathStyle::Curved)
            .with_arrowhead(Arrowhead::Diamond)
            .with_label("flow");
        assert_eq!(c.path_style, PathStyle::Curved);
        assert_eq!(c.arrowhead, Arrowhead::Diamond);
        assert_eq!(c.label.as_deref(), Some("flow"));
    }

    #[test]
    fn test_shapes_in_region() {
        let mut canvas = Canvas::new();
        canvas.add_shape(rect_shape("s1", 0.0, 0.0));
        canvas.add_shape(rect_shape("s2", 500.0, 500.0));
        let hits = canvas.shapes_in_region(0.0, 0.0, 200.0, 200.0);
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, "s1");
    }

    #[test]
    fn test_shape_center() {
        let s = rect_shape("s1", 10.0, 20.0);
        let c = s.center();
        assert_eq!(c.x, 60.0);
        assert_eq!(c.y, 50.0);
    }

    #[test]
    fn test_all_shape_types_hit_test_center() {
        let shapes = [
            Shape::Rect, Shape::RoundedRect, Shape::Ellipse, Shape::Diamond,
            Shape::Parallelogram, Shape::Hexagon, Shape::Cylinder, Shape::Cloud,
        ];
        for shape_type in &shapes {
            let s = ShapeInstance::new(
                "test",
                *shape_type,
                Position::new(0.0, 0.0),
                Size::new(100.0, 100.0),
            );
            assert!(
                s.contains_point(50.0, 50.0),
                "center should be inside {shape_type:?}"
            );
        }
    }
}
