// UI Renderer — Immediate-mode UI rendering backend
// Draw primitives, clip stack, z-ordering, draw command list, batching, vertex generation

/// RGBA color.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Color {
    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    pub const WHITE: Self = Self::new(1.0, 1.0, 1.0, 1.0);
    pub const BLACK: Self = Self::new(0.0, 0.0, 0.0, 1.0);
    pub const TRANSPARENT: Self = Self::new(0.0, 0.0, 0.0, 0.0);

    pub fn with_alpha(self, a: f32) -> Self {
        Self { a, ..self }
    }
}

/// A 2D vertex for rendering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: Color,
}

/// Scissor rectangle for clipping.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScissorRect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl ScissorRect {
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Intersect two scissor rects (nested clipping).
    pub fn intersect(&self, other: &ScissorRect) -> ScissorRect {
        let x1 = self.x.max(other.x);
        let y1 = self.y.max(other.y);
        let x2 = (self.x + self.width).min(other.x + other.width);
        let y2 = (self.y + self.height).min(other.y + other.height);
        ScissorRect {
            x: x1,
            y: y1,
            width: (x2 - x1).max(0.0),
            height: (y2 - y1).max(0.0),
        }
    }

    pub fn contains(&self, x: f32, y: f32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }
}

/// Theme colors for the UI.
#[derive(Debug, Clone, PartialEq)]
pub struct Theme {
    pub background: Color,
    pub foreground: Color,
    pub accent: Color,
    pub border: Color,
    pub hover: Color,
    pub disabled: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            background: Color::new(0.15, 0.15, 0.15, 1.0),
            foreground: Color::new(0.9, 0.9, 0.9, 1.0),
            accent: Color::new(0.2, 0.5, 0.9, 1.0),
            border: Color::new(0.3, 0.3, 0.3, 1.0),
            hover: Color::new(0.25, 0.25, 0.25, 1.0),
            disabled: Color::new(0.5, 0.5, 0.5, 0.5),
        }
    }
}

/// A draw command in the command list.
#[derive(Debug, Clone, PartialEq)]
pub enum DrawCommand {
    FilledRect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: Color,
    },
    StrokedRect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: Color,
        stroke_width: f32,
    },
    RoundedRect {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        color: Color,
        filled: bool,
        stroke_width: f32,
    },
    Circle {
        cx: f32,
        cy: f32,
        r: f32,
        color: Color,
        filled: bool,
        stroke_width: f32,
    },
    Line {
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: Color,
        width: f32,
    },
    Triangle {
        points: [[f32; 2]; 3],
        color: Color,
        filled: bool,
    },
    Text {
        x: f32,
        y: f32,
        text: String,
        color: Color,
        font_size: f32,
    },
    Image {
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        texture_id: u32,
    },
    PushClip(ScissorRect),
    PopClip,
}

/// Z-ordered draw layer.
#[derive(Debug, Clone)]
struct DrawLayer {
    z_index: i32,
    commands: Vec<DrawCommand>,
}

/// Batched draw call (consecutive same-texture commands merged).
#[derive(Debug, Clone, PartialEq)]
pub struct DrawBatch {
    pub texture_id: Option<u32>,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub scissor: Option<ScissorRect>,
}

/// The immediate-mode UI renderer.
#[derive(Debug)]
pub struct UiRenderer {
    layers: Vec<DrawLayer>,
    current_z: i32,
    clip_stack: Vec<ScissorRect>,
    pub theme: Theme,
}

impl UiRenderer {
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            current_z: 0,
            clip_stack: Vec::new(),
            theme: Theme::default(),
        }
    }

    pub fn with_theme(theme: Theme) -> Self {
        Self {
            layers: Vec::new(),
            current_z: 0,
            clip_stack: Vec::new(),
            theme,
        }
    }

    /// Begin a new frame (clear all commands).
    pub fn begin_frame(&mut self) {
        self.layers.clear();
        self.current_z = 0;
        self.clip_stack.clear();
    }

    /// Set the current z-layer.
    pub fn set_z(&mut self, z: i32) {
        self.current_z = z;
    }

    /// Push a clip rectangle (nested).
    pub fn push_clip(&mut self, rect: ScissorRect) {
        let effective = if let Some(top) = self.clip_stack.last() {
            top.intersect(&rect)
        } else {
            rect
        };
        self.clip_stack.push(effective);
        self.push_command(DrawCommand::PushClip(effective));
    }

    /// Pop the top clip rectangle.
    pub fn pop_clip(&mut self) {
        self.clip_stack.pop();
        self.push_command(DrawCommand::PopClip);
    }

    pub fn current_clip(&self) -> Option<&ScissorRect> {
        self.clip_stack.last()
    }

    fn push_command(&mut self, cmd: DrawCommand) {
        let z = self.current_z;
        if let Some(layer) = self.layers.iter_mut().find(|l| l.z_index == z) {
            layer.commands.push(cmd);
        } else {
            self.layers.push(DrawLayer {
                z_index: z,
                commands: vec![cmd],
            });
        }
    }

    // --- Drawing primitives ---

    pub fn draw_filled_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: Color) {
        self.push_command(DrawCommand::FilledRect { x, y, w, h, color });
    }

    pub fn draw_stroked_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        color: Color,
        stroke_width: f32,
    ) {
        self.push_command(DrawCommand::StrokedRect {
            x,
            y,
            w,
            h,
            color,
            stroke_width,
        });
    }

    pub fn draw_rounded_rect(
        &mut self,
        x: f32,
        y: f32,
        w: f32,
        h: f32,
        radius: f32,
        color: Color,
        filled: bool,
        stroke_width: f32,
    ) {
        self.push_command(DrawCommand::RoundedRect {
            x,
            y,
            w,
            h,
            radius,
            color,
            filled,
            stroke_width,
        });
    }

    pub fn draw_circle(
        &mut self,
        cx: f32,
        cy: f32,
        r: f32,
        color: Color,
        filled: bool,
        stroke_width: f32,
    ) {
        self.push_command(DrawCommand::Circle {
            cx,
            cy,
            r,
            color,
            filled,
            stroke_width,
        });
    }

    pub fn draw_line(
        &mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: Color,
        width: f32,
    ) {
        self.push_command(DrawCommand::Line {
            x1,
            y1,
            x2,
            y2,
            color,
            width,
        });
    }

    pub fn draw_triangle(
        &mut self,
        points: [[f32; 2]; 3],
        color: Color,
        filled: bool,
    ) {
        self.push_command(DrawCommand::Triangle {
            points,
            color,
            filled,
        });
    }

    pub fn draw_text(&mut self, x: f32, y: f32, text: &str, color: Color, font_size: f32) {
        self.push_command(DrawCommand::Text {
            x,
            y,
            text: text.to_string(),
            color,
            font_size,
        });
    }

    pub fn draw_image(&mut self, x: f32, y: f32, w: f32, h: f32, texture_id: u32) {
        self.push_command(DrawCommand::Image {
            x,
            y,
            w,
            h,
            texture_id,
        });
    }

    /// Finalize and return the sorted command list.
    pub fn finish(&mut self) -> Vec<DrawCommand> {
        self.layers.sort_by_key(|l| l.z_index);
        let mut commands = Vec::new();
        for layer in &self.layers {
            commands.extend(layer.commands.iter().cloned());
        }
        commands
    }

    /// Generate batched draw calls from commands. Merges consecutive same-texture calls.
    pub fn generate_batches(&mut self) -> Vec<DrawBatch> {
        let commands = self.finish();
        let mut batches: Vec<DrawBatch> = Vec::new();
        let mut current_scissor: Option<ScissorRect> = None;

        for cmd in &commands {
            match cmd {
                DrawCommand::PushClip(rect) => {
                    current_scissor = Some(*rect);
                }
                DrawCommand::PopClip => {
                    current_scissor = None;
                }
                DrawCommand::FilledRect { x, y, w, h, color } => {
                    let (verts, idxs) = generate_rect_vertices(*x, *y, *w, *h, *color);
                    push_to_batch(&mut batches, None, verts, idxs, current_scissor);
                }
                DrawCommand::Line {
                    x1,
                    y1,
                    x2,
                    y2,
                    color,
                    width,
                } => {
                    let (verts, idxs) =
                        generate_line_vertices(*x1, *y1, *x2, *y2, *width, *color);
                    push_to_batch(&mut batches, None, verts, idxs, current_scissor);
                }
                DrawCommand::Circle {
                    cx,
                    cy,
                    r,
                    color,
                    filled,
                    ..
                } => {
                    if *filled {
                        let (verts, idxs) = generate_circle_vertices(*cx, *cy, *r, 16, *color);
                        push_to_batch(&mut batches, None, verts, idxs, current_scissor);
                    }
                }
                DrawCommand::Image {
                    x,
                    y,
                    w,
                    h,
                    texture_id,
                } => {
                    let (verts, idxs) =
                        generate_image_vertices(*x, *y, *w, *h, Color::WHITE);
                    push_to_batch(
                        &mut batches,
                        Some(*texture_id),
                        verts,
                        idxs,
                        current_scissor,
                    );
                }
                DrawCommand::Triangle {
                    points,
                    color,
                    filled,
                } => {
                    if *filled {
                        let (verts, idxs) = generate_triangle_vertices(points, *color);
                        push_to_batch(&mut batches, None, verts, idxs, current_scissor);
                    }
                }
                _ => {
                    // StrokedRect, RoundedRect, Text: would need font/stroke tessellation
                    // Record as empty batch for command tracking
                }
            }
        }

        batches
    }
}

fn push_to_batch(
    batches: &mut Vec<DrawBatch>,
    texture_id: Option<u32>,
    verts: Vec<Vertex>,
    idxs: Vec<u32>,
    scissor: Option<ScissorRect>,
) {
    // Try to merge with last batch if same texture and scissor
    if let Some(last) = batches.last_mut() {
        if last.texture_id == texture_id && last.scissor == scissor {
            let offset = last.vertices.len() as u32;
            last.vertices.extend(verts);
            last.indices.extend(idxs.iter().map(|i| i + offset));
            return;
        }
    }

    batches.push(DrawBatch {
        texture_id,
        vertices: verts,
        indices: idxs,
        scissor,
    });
}

/// Generate 4 vertices + 6 indices for a filled rectangle.
pub fn generate_rect_vertices(x: f32, y: f32, w: f32, h: f32, color: Color) -> (Vec<Vertex>, Vec<u32>) {
    let verts = vec![
        Vertex {
            position: [x, y],
            uv: [0.0, 0.0],
            color,
        },
        Vertex {
            position: [x + w, y],
            uv: [1.0, 0.0],
            color,
        },
        Vertex {
            position: [x + w, y + h],
            uv: [1.0, 1.0],
            color,
        },
        Vertex {
            position: [x, y + h],
            uv: [0.0, 1.0],
            color,
        },
    ];
    let idxs = vec![0, 1, 2, 0, 2, 3];
    (verts, idxs)
}

fn generate_image_vertices(x: f32, y: f32, w: f32, h: f32, color: Color) -> (Vec<Vertex>, Vec<u32>) {
    generate_rect_vertices(x, y, w, h, color)
}

/// Generate vertices for a line (as a thin quad).
pub fn generate_line_vertices(
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    width: f32,
    color: Color,
) -> (Vec<Vertex>, Vec<u32>) {
    let dx = x2 - x1;
    let dy = y2 - y1;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-8 {
        return (Vec::new(), Vec::new());
    }
    let hw = width * 0.5;
    let nx = -dy / len * hw;
    let ny = dx / len * hw;

    let verts = vec![
        Vertex {
            position: [x1 + nx, y1 + ny],
            uv: [0.0, 0.0],
            color,
        },
        Vertex {
            position: [x1 - nx, y1 - ny],
            uv: [0.0, 1.0],
            color,
        },
        Vertex {
            position: [x2 - nx, y2 - ny],
            uv: [1.0, 1.0],
            color,
        },
        Vertex {
            position: [x2 + nx, y2 + ny],
            uv: [1.0, 0.0],
            color,
        },
    ];
    (verts, vec![0, 1, 2, 0, 2, 3])
}

/// Generate vertices for a filled circle (triangle fan).
pub fn generate_circle_vertices(
    cx: f32,
    cy: f32,
    r: f32,
    segments: u32,
    color: Color,
) -> (Vec<Vertex>, Vec<u32>) {
    let mut verts = vec![Vertex {
        position: [cx, cy],
        uv: [0.5, 0.5],
        color,
    }];
    let mut idxs = Vec::new();

    for i in 0..=segments {
        let angle = (i as f32 / segments as f32) * std::f32::consts::TAU;
        let px = cx + r * angle.cos();
        let py = cy + r * angle.sin();
        verts.push(Vertex {
            position: [px, py],
            uv: [0.5 + 0.5 * angle.cos(), 0.5 + 0.5 * angle.sin()],
            color,
        });
        if i > 0 {
            idxs.push(0);
            idxs.push(i);
            idxs.push(i + 1);
        }
    }

    (verts, idxs)
}

fn generate_triangle_vertices(points: &[[f32; 2]; 3], color: Color) -> (Vec<Vertex>, Vec<u32>) {
    let verts = vec![
        Vertex {
            position: points[0],
            uv: [0.0, 0.0],
            color,
        },
        Vertex {
            position: points[1],
            uv: [1.0, 0.0],
            color,
        },
        Vertex {
            position: points[2],
            uv: [0.5, 1.0],
            color,
        },
    ];
    (verts, vec![0, 1, 2])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_constants() {
        assert!((Color::WHITE.r - 1.0).abs() < 1e-6);
        assert!((Color::BLACK.r).abs() < 1e-6);
        assert!((Color::TRANSPARENT.a).abs() < 1e-6);
    }

    #[test]
    fn test_color_with_alpha() {
        let c = Color::WHITE.with_alpha(0.5);
        assert!((c.r - 1.0).abs() < 1e-6);
        assert!((c.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_scissor_rect_contains() {
        let s = ScissorRect::new(10.0, 10.0, 100.0, 100.0);
        assert!(s.contains(50.0, 50.0));
        assert!(!s.contains(5.0, 5.0));
        assert!(!s.contains(111.0, 50.0));
    }

    #[test]
    fn test_scissor_intersect() {
        let a = ScissorRect::new(0.0, 0.0, 100.0, 100.0);
        let b = ScissorRect::new(50.0, 50.0, 100.0, 100.0);
        let c = a.intersect(&b);
        assert!((c.x - 50.0).abs() < 1e-6);
        assert!((c.y - 50.0).abs() < 1e-6);
        assert!((c.width - 50.0).abs() < 1e-6);
        assert!((c.height - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_scissor_intersect_no_overlap() {
        let a = ScissorRect::new(0.0, 0.0, 10.0, 10.0);
        let b = ScissorRect::new(20.0, 20.0, 10.0, 10.0);
        let c = a.intersect(&b);
        assert!((c.width).abs() < 1e-6);
        assert!((c.height).abs() < 1e-6);
    }

    #[test]
    fn test_theme_default() {
        let t = Theme::default();
        assert!(t.background.r < t.foreground.r);
    }

    #[test]
    fn test_renderer_new() {
        let r = UiRenderer::new();
        assert_eq!(r.current_z, 0);
        assert!(r.clip_stack.is_empty());
    }

    #[test]
    fn test_draw_filled_rect() {
        let mut r = UiRenderer::new();
        r.draw_filled_rect(10.0, 20.0, 100.0, 50.0, Color::WHITE);
        let cmds = r.finish();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            DrawCommand::FilledRect { x, y, w, h, .. } => {
                assert!((x - 10.0).abs() < 1e-6);
                assert!((y - 20.0).abs() < 1e-6);
                assert!((w - 100.0).abs() < 1e-6);
                assert!((h - 50.0).abs() < 1e-6);
            }
            _ => panic!("Wrong command"),
        }
    }

    #[test]
    fn test_z_ordering() {
        let mut r = UiRenderer::new();
        r.set_z(1);
        r.draw_filled_rect(0.0, 0.0, 10.0, 10.0, Color::WHITE);
        r.set_z(0);
        r.draw_filled_rect(0.0, 0.0, 20.0, 20.0, Color::BLACK);
        let cmds = r.finish();
        // Z=0 should come first
        match &cmds[0] {
            DrawCommand::FilledRect { w, .. } => assert!((w - 20.0).abs() < 1e-6),
            _ => panic!("wrong"),
        }
    }

    #[test]
    fn test_clip_stack() {
        let mut r = UiRenderer::new();
        r.push_clip(ScissorRect::new(0.0, 0.0, 100.0, 100.0));
        assert!(r.current_clip().is_some());
        r.push_clip(ScissorRect::new(50.0, 50.0, 100.0, 100.0));
        // Nested: should be intersection
        let c = r.current_clip().unwrap();
        assert!((c.x - 50.0).abs() < 1e-6);
        assert!((c.width - 50.0).abs() < 1e-6);
        r.pop_clip();
        r.pop_clip();
        assert!(r.current_clip().is_none());
    }

    #[test]
    fn test_begin_frame_resets() {
        let mut r = UiRenderer::new();
        r.draw_filled_rect(0.0, 0.0, 10.0, 10.0, Color::WHITE);
        r.begin_frame();
        let cmds = r.finish();
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_generate_rect_vertices() {
        let (verts, idxs) = generate_rect_vertices(10.0, 20.0, 30.0, 40.0, Color::WHITE);
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs.len(), 6);
        assert!((verts[0].position[0] - 10.0).abs() < 1e-6);
        assert!((verts[2].position[0] - 40.0).abs() < 1e-6);
    }

    #[test]
    fn test_generate_line_vertices() {
        let (verts, idxs) = generate_line_vertices(0.0, 0.0, 10.0, 0.0, 2.0, Color::WHITE);
        assert_eq!(verts.len(), 4);
        assert_eq!(idxs.len(), 6);
    }

    #[test]
    fn test_generate_line_zero_length() {
        let (verts, idxs) = generate_line_vertices(5.0, 5.0, 5.0, 5.0, 2.0, Color::WHITE);
        assert!(verts.is_empty());
        assert!(idxs.is_empty());
    }

    #[test]
    fn test_generate_circle_vertices() {
        let (verts, idxs) = generate_circle_vertices(50.0, 50.0, 25.0, 8, Color::WHITE);
        assert_eq!(verts.len(), 10); // center + 9 edge verts
        assert!(!idxs.is_empty());
    }

    #[test]
    fn test_draw_multiple_primitives() {
        let mut r = UiRenderer::new();
        r.draw_filled_rect(0.0, 0.0, 10.0, 10.0, Color::WHITE);
        r.draw_circle(50.0, 50.0, 10.0, Color::BLACK, true, 0.0);
        r.draw_line(0.0, 0.0, 100.0, 100.0, Color::WHITE, 1.0);
        let cmds = r.finish();
        assert_eq!(cmds.len(), 3);
    }

    #[test]
    fn test_batching_merges() {
        let mut r = UiRenderer::new();
        // Two consecutive filled rects (same texture = None) should batch
        r.draw_filled_rect(0.0, 0.0, 10.0, 10.0, Color::WHITE);
        r.draw_filled_rect(20.0, 0.0, 10.0, 10.0, Color::WHITE);
        let batches = r.generate_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].vertices.len(), 8); // 2 rects * 4 verts
        assert_eq!(batches[0].indices.len(), 12); // 2 rects * 6 indices
    }

    #[test]
    fn test_batching_breaks_on_texture() {
        let mut r = UiRenderer::new();
        r.draw_filled_rect(0.0, 0.0, 10.0, 10.0, Color::WHITE);
        r.draw_image(20.0, 0.0, 10.0, 10.0, 1);
        let batches = r.generate_batches();
        assert_eq!(batches.len(), 2);
        assert!(batches[0].texture_id.is_none());
        assert_eq!(batches[1].texture_id, Some(1));
    }

    #[test]
    fn test_draw_text_command() {
        let mut r = UiRenderer::new();
        r.draw_text(10.0, 20.0, "Hello", Color::WHITE, 16.0);
        let cmds = r.finish();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            DrawCommand::Text { text, font_size, .. } => {
                assert_eq!(text, "Hello");
                assert!((*font_size - 16.0).abs() < 1e-6);
            }
            _ => panic!("wrong command"),
        }
    }

    #[test]
    fn test_draw_triangle_command() {
        let mut r = UiRenderer::new();
        let pts = [[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]];
        r.draw_triangle(pts, Color::WHITE, true);
        let batches = r.generate_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].vertices.len(), 3);
        assert_eq!(batches[0].indices.len(), 3);
    }

    #[test]
    fn test_with_theme() {
        let theme = Theme {
            accent: Color::new(1.0, 0.0, 0.0, 1.0),
            ..Default::default()
        };
        let r = UiRenderer::with_theme(theme);
        assert!((r.theme.accent.r - 1.0).abs() < 1e-6);
        assert!((r.theme.accent.g).abs() < 1e-6);
    }
}
