//! Lottie animation model (After Effects JSON format).
//!
//! Provides an in-memory representation of Lottie compositions,
//! layers, shapes, and transforms — all pure Rust. Supports
//! keyframe-based interpolation at any frame. Does NOT parse JSON
//! directly; use `serde_json` + `from_json` to deserialize.

use std::collections::HashMap;

// ── Keyframe ───────────────────────────────────────────────────

/// A keyframed value that changes over time.
#[derive(Debug, Clone)]
pub struct KeyframedValue {
    /// (frame, value) pairs sorted by frame number.
    keyframes: Vec<(f64, f64)>,
}

impl KeyframedValue {
    /// Create a static (non-animated) value.
    pub fn static_value(value: f64) -> Self {
        Self { keyframes: vec![(0.0, value)] }
    }

    /// Create from a list of (frame, value) pairs.
    pub fn from_keyframes(mut keyframes: Vec<(f64, f64)>) -> Self {
        keyframes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        Self { keyframes }
    }

    /// Interpolate the value at a given frame.
    pub fn value_at(&self, frame: f64) -> f64 {
        if self.keyframes.is_empty() {
            return 0.0;
        }
        if self.keyframes.len() == 1 {
            return self.keyframes[0].1;
        }
        if frame <= self.keyframes[0].0 {
            return self.keyframes[0].1;
        }
        if frame >= self.keyframes.last().unwrap().0 {
            return self.keyframes.last().unwrap().1;
        }

        // Find surrounding keyframes.
        for i in 0..self.keyframes.len() - 1 {
            let (f0, v0) = self.keyframes[i];
            let (f1, v1) = self.keyframes[i + 1];
            if frame >= f0 && frame <= f1 {
                let range = f1 - f0;
                if range < 1e-12 {
                    return v0;
                }
                let t = (frame - f0) / range;
                return v0 + (v1 - v0) * t;
            }
        }

        self.keyframes.last().unwrap().1
    }

    /// Whether this value has multiple keyframes (is animated).
    pub fn is_animated(&self) -> bool {
        self.keyframes.len() > 1
    }
}

// ── Transform ──────────────────────────────────────────────────

/// Layer transform with keyframed properties.
#[derive(Debug, Clone)]
pub struct Transform {
    pub position_x: KeyframedValue,
    pub position_y: KeyframedValue,
    pub scale_x: KeyframedValue,
    pub scale_y: KeyframedValue,
    pub rotation: KeyframedValue,
    pub opacity: KeyframedValue,
}

impl Transform {
    pub fn identity() -> Self {
        Self {
            position_x: KeyframedValue::static_value(0.0),
            position_y: KeyframedValue::static_value(0.0),
            scale_x: KeyframedValue::static_value(100.0),
            scale_y: KeyframedValue::static_value(100.0),
            rotation: KeyframedValue::static_value(0.0),
            opacity: KeyframedValue::static_value(100.0),
        }
    }

    /// Evaluate all transform properties at a given frame.
    pub fn at_frame(&self, frame: f64) -> TransformSnapshot {
        TransformSnapshot {
            position_x: self.position_x.value_at(frame),
            position_y: self.position_y.value_at(frame),
            scale_x: self.scale_x.value_at(frame),
            scale_y: self.scale_y.value_at(frame),
            rotation: self.rotation.value_at(frame),
            opacity: self.opacity.value_at(frame),
        }
    }
}

/// A snapshot of transform values at a specific frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformSnapshot {
    pub position_x: f64,
    pub position_y: f64,
    pub scale_x: f64,
    pub scale_y: f64,
    pub rotation: f64,
    pub opacity: f64,
}

// ── Shape ──────────────────────────────────────────────────────

/// Shape types within a shape layer.
#[derive(Debug, Clone)]
pub enum Shape {
    Rect {
        x: KeyframedValue,
        y: KeyframedValue,
        width: KeyframedValue,
        height: KeyframedValue,
        corner_radius: KeyframedValue,
    },
    Ellipse {
        cx: KeyframedValue,
        cy: KeyframedValue,
        rx: KeyframedValue,
        ry: KeyframedValue,
    },
    Path {
        /// Simplified: list of (x, y) control points (non-animated for now).
        vertices: Vec<(f64, f64)>,
        closed: bool,
    },
    Fill {
        r: KeyframedValue,
        g: KeyframedValue,
        b: KeyframedValue,
        opacity: KeyframedValue,
    },
    Stroke {
        r: KeyframedValue,
        g: KeyframedValue,
        b: KeyframedValue,
        opacity: KeyframedValue,
        width: KeyframedValue,
    },
}

/// A group of shapes.
#[derive(Debug, Clone)]
pub struct ShapeGroup {
    pub name: String,
    pub shapes: Vec<Shape>,
}

impl ShapeGroup {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into(), shapes: Vec::new() }
    }

    pub fn add_shape(&mut self, shape: Shape) {
        self.shapes.push(shape);
    }

    pub fn with_shape(mut self, shape: Shape) -> Self {
        self.shapes.push(shape);
        self
    }
}

// ── Layer ──────────────────────────────────────────────────────

/// Type of Lottie layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    Shape,
    Solid,
    Null,
}

/// A single layer in the composition.
#[derive(Debug, Clone)]
pub struct Layer {
    pub name: String,
    pub layer_type: LayerType,
    pub transform: Transform,
    /// First frame this layer is visible.
    pub in_point: f64,
    /// Last frame this layer is visible.
    pub out_point: f64,
    /// Shape groups (only for Shape layers).
    pub shape_groups: Vec<ShapeGroup>,
    /// Solid color (only for Solid layers): (r, g, b).
    pub solid_color: Option<(u8, u8, u8)>,
    /// Parent layer index (for parenting hierarchy).
    pub parent_index: Option<usize>,
}

impl Layer {
    pub fn shape(name: impl Into<String>, in_pt: f64, out_pt: f64) -> Self {
        Self {
            name: name.into(),
            layer_type: LayerType::Shape,
            transform: Transform::identity(),
            in_point: in_pt,
            out_point: out_pt,
            shape_groups: Vec::new(),
            solid_color: None,
            parent_index: None,
        }
    }

    pub fn solid(name: impl Into<String>, in_pt: f64, out_pt: f64, color: (u8, u8, u8)) -> Self {
        Self {
            name: name.into(),
            layer_type: LayerType::Solid,
            transform: Transform::identity(),
            in_point: in_pt,
            out_point: out_pt,
            shape_groups: Vec::new(),
            solid_color: Some(color),
            parent_index: None,
        }
    }

    pub fn null(name: impl Into<String>, in_pt: f64, out_pt: f64) -> Self {
        Self {
            name: name.into(),
            layer_type: LayerType::Null,
            transform: Transform::identity(),
            in_point: in_pt,
            out_point: out_pt,
            shape_groups: Vec::new(),
            solid_color: None,
            parent_index: None,
        }
    }

    pub fn with_transform(mut self, transform: Transform) -> Self {
        self.transform = transform;
        self
    }

    pub fn with_shape_group(mut self, group: ShapeGroup) -> Self {
        self.shape_groups.push(group);
        self
    }

    pub fn with_parent(mut self, index: usize) -> Self {
        self.parent_index = Some(index);
        self
    }

    /// Whether the layer is visible at the given frame.
    pub fn is_visible_at(&self, frame: f64) -> bool {
        frame >= self.in_point && frame < self.out_point
    }
}

// ── Composition ────────────────────────────────────────────────

/// A full Lottie composition.
#[derive(Debug, Clone)]
pub struct LottieComposition {
    pub width: f64,
    pub height: f64,
    pub fps: f64,
    pub total_frames: f64,
    pub layers: Vec<Layer>,
    /// Named markers: name -> frame.
    pub markers: HashMap<String, f64>,
}

impl LottieComposition {
    pub fn new(width: f64, height: f64, fps: f64, total_frames: f64) -> Self {
        Self {
            width,
            height,
            fps,
            total_frames,
            layers: Vec::new(),
            markers: HashMap::new(),
        }
    }

    pub fn add_layer(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    pub fn with_layer(mut self, layer: Layer) -> Self {
        self.layers.push(layer);
        self
    }

    pub fn add_marker(&mut self, name: impl Into<String>, frame: f64) {
        self.markers.insert(name.into(), frame);
    }

    /// Duration in seconds.
    pub fn duration_secs(&self) -> f64 {
        if self.fps > 0.0 { self.total_frames / self.fps } else { 0.0 }
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> f64 {
        self.duration_secs() * 1000.0
    }

    /// Frame number from time in seconds.
    pub fn frame_at_time(&self, time_secs: f64) -> f64 {
        (time_secs * self.fps).clamp(0.0, self.total_frames)
    }

    /// Get all visible layers at a given frame, with their transform snapshots.
    pub fn layers_at_frame(&self, frame: f64) -> Vec<(&Layer, TransformSnapshot)> {
        self.layers.iter()
            .filter(|l| l.is_visible_at(frame))
            .map(|l| (l, l.transform.at_frame(frame)))
            .collect()
    }

    /// Get a layer by name.
    pub fn layer_by_name(&self, name: &str) -> Option<&Layer> {
        self.layers.iter().find(|l| l.name == name)
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn test_composition() -> LottieComposition {
        let mut transform = Transform::identity();
        transform.position_x = KeyframedValue::from_keyframes(vec![
            (0.0, 0.0),
            (30.0, 100.0),
        ]);
        transform.opacity = KeyframedValue::from_keyframes(vec![
            (0.0, 0.0),
            (15.0, 100.0),
        ]);

        let shape = Shape::Rect {
            x: KeyframedValue::static_value(0.0),
            y: KeyframedValue::static_value(0.0),
            width: KeyframedValue::static_value(50.0),
            height: KeyframedValue::static_value(50.0),
            corner_radius: KeyframedValue::static_value(5.0),
        };

        let group = ShapeGroup::new("rect-group").with_shape(shape);

        let layer = Layer::shape("moving-rect", 0.0, 30.0)
            .with_transform(transform)
            .with_shape_group(group);

        LottieComposition::new(400.0, 300.0, 30.0, 30.0)
            .with_layer(layer)
    }

    #[test]
    fn composition_dimensions() {
        let comp = test_composition();
        assert_eq!(comp.width, 400.0);
        assert_eq!(comp.height, 300.0);
        assert_eq!(comp.fps, 30.0);
    }

    #[test]
    fn duration_calculation() {
        let comp = test_composition();
        assert!((comp.duration_secs() - 1.0).abs() < 1e-10);
        assert!((comp.duration_ms() - 1000.0).abs() < 1e-10);
    }

    #[test]
    fn frame_at_time() {
        let comp = test_composition();
        assert!((comp.frame_at_time(0.5) - 15.0).abs() < 1e-10);
    }

    #[test]
    fn transform_interpolation() {
        let comp = test_composition();
        let layers = comp.layers_at_frame(15.0);
        assert_eq!(layers.len(), 1);
        let (_, snap) = &layers[0];
        assert!((snap.position_x - 50.0).abs() < 0.1);
        assert!((snap.opacity - 100.0).abs() < 0.1);
    }

    #[test]
    fn layer_visibility() {
        let comp = test_composition();
        assert_eq!(comp.layers_at_frame(0.0).len(), 1);
        assert_eq!(comp.layers_at_frame(29.9).len(), 1);
        assert_eq!(comp.layers_at_frame(30.0).len(), 0);
    }

    #[test]
    fn keyframed_value_static() {
        let v = KeyframedValue::static_value(42.0);
        assert!(!v.is_animated());
        assert!((v.value_at(0.0) - 42.0).abs() < 1e-10);
        assert!((v.value_at(100.0) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn keyframed_value_animated() {
        let v = KeyframedValue::from_keyframes(vec![(0.0, 0.0), (10.0, 100.0)]);
        assert!(v.is_animated());
        assert!((v.value_at(5.0) - 50.0).abs() < 0.01);
    }

    #[test]
    fn keyframed_value_clamped() {
        let v = KeyframedValue::from_keyframes(vec![(5.0, 10.0), (15.0, 20.0)]);
        // Before first keyframe.
        assert!((v.value_at(0.0) - 10.0).abs() < 1e-10);
        // After last keyframe.
        assert!((v.value_at(100.0) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn solid_layer() {
        let layer = Layer::solid("bg", 0.0, 30.0, (255, 0, 0));
        assert_eq!(layer.layer_type, LayerType::Solid);
        assert_eq!(layer.solid_color, Some((255, 0, 0)));
    }

    #[test]
    fn null_layer_parenting() {
        let null = Layer::null("controller", 0.0, 30.0);
        let child = Layer::shape("child", 0.0, 30.0).with_parent(0);
        let comp = LottieComposition::new(100.0, 100.0, 30.0, 30.0)
            .with_layer(null)
            .with_layer(child);
        assert_eq!(comp.layers[1].parent_index, Some(0));
    }

    #[test]
    fn markers() {
        let mut comp = test_composition();
        comp.add_marker("intro", 0.0);
        comp.add_marker("outro", 25.0);
        assert_eq!(comp.markers.len(), 2);
        assert!((comp.markers["outro"] - 25.0).abs() < 1e-10);
    }

    #[test]
    fn layer_by_name() {
        let comp = test_composition();
        let layer = comp.layer_by_name("moving-rect");
        assert!(layer.is_some());
        assert_eq!(layer.unwrap().layer_type, LayerType::Shape);
        assert!(comp.layer_by_name("nonexistent").is_none());
    }
}
