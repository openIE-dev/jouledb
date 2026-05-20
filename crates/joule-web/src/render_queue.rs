//! Render command sorting and batching.
//!
//! Commands carry a material ID, mesh ID, transform, and sort key.
//! Opaque geometry is sorted front-to-back (to maximise early-z),
//! transparent geometry is sorted back-to-front (for correct blending).
//! Consecutive same-material draws are batched. Layer / render-order
//! support. Statistics tracking (draw calls, triangles, batches).

// ── Render layer ────────────────────────────────────────────────

/// Render layers control coarse ordering before per-object sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RenderLayer(pub i32);

impl RenderLayer {
    pub const BACKGROUND: Self = Self(-100);
    pub const DEFAULT: Self = Self(0);
    pub const TRANSPARENT: Self = Self(100);
    pub const OVERLAY: Self = Self(200);
    pub const UI: Self = Self(300);
}

// ── Blend mode ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    Opaque,
    AlphaBlend,
    Additive,
    Multiply,
}

impl BlendMode {
    pub fn is_transparent(&self) -> bool {
        !matches!(self, BlendMode::Opaque)
    }
}

// ── Render command ──────────────────────────────────────────────

/// A single draw command submitted to the render queue.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderCommand {
    pub material_id: u64,
    pub mesh_id: u64,
    pub transform: [f64; 16],
    pub layer: RenderLayer,
    pub blend: BlendMode,
    /// Distance from camera (squared). Used for depth sorting.
    pub depth_sq: f64,
    /// Number of triangles in this draw call.
    pub triangle_count: u32,
    /// User-defined sub-order within the same depth/layer.
    pub sub_order: i32,
}

impl RenderCommand {
    pub fn new(material_id: u64, mesh_id: u64, transform: [f64; 16]) -> Self {
        Self {
            material_id,
            mesh_id,
            transform,
            layer: RenderLayer::DEFAULT,
            blend: BlendMode::Opaque,
            depth_sq: 0.0,
            triangle_count: 0,
            sub_order: 0,
        }
    }

    pub fn with_layer(mut self, layer: RenderLayer) -> Self {
        self.layer = layer;
        self
    }

    pub fn with_blend(mut self, blend: BlendMode) -> Self {
        self.blend = blend;
        self
    }

    pub fn with_depth_sq(mut self, d: f64) -> Self {
        self.depth_sq = d;
        self
    }

    pub fn with_triangles(mut self, n: u32) -> Self {
        self.triangle_count = n;
        self
    }

    pub fn with_sub_order(mut self, order: i32) -> Self {
        self.sub_order = order;
        self
    }

    /// Sort key for opaque: layer ASC, depth ASC (front-to-back), material ASC.
    fn opaque_key(&self) -> (i32, i64, u64, i32) {
        (self.layer.0, float_to_sort_key(self.depth_sq), self.material_id, self.sub_order)
    }

    /// Sort key for transparent: layer ASC, depth DESC (back-to-front), material ASC.
    fn transparent_key(&self) -> (i32, i64, u64, i32) {
        (self.layer.0, -float_to_sort_key(self.depth_sq), self.material_id, self.sub_order)
    }
}

/// Convert an f64 to a sortable i64 representation.
fn float_to_sort_key(v: f64) -> i64 {
    let bits = v.to_bits() as i64;
    if bits < 0 {
        // Negative floats: flip all bits.
        !bits
    } else {
        // Positive floats: flip sign bit.
        bits ^ (1i64 << 63)
    }
}

// ── Batch ───────────────────────────────────────────────────────

/// A batch of draw calls that share the same material.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderBatch {
    pub material_id: u64,
    pub commands: Vec<RenderCommand>,
    pub total_triangles: u32,
}

// ── Statistics ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RenderStats {
    pub draw_calls: u32,
    pub batches: u32,
    pub triangles: u32,
    pub opaque_calls: u32,
    pub transparent_calls: u32,
}

// ── Render queue ────────────────────────────────────────────────

/// Collects render commands, sorts them, and produces sorted batches.
#[derive(Debug)]
pub struct RenderQueue {
    commands: Vec<RenderCommand>,
}

impl RenderQueue {
    pub fn new() -> Self {
        Self { commands: Vec::new() }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self { commands: Vec::with_capacity(cap) }
    }

    /// Submit a render command.
    pub fn submit(&mut self, cmd: RenderCommand) {
        self.commands.push(cmd);
    }

    /// Submit multiple commands.
    pub fn submit_many(&mut self, cmds: impl IntoIterator<Item = RenderCommand>) {
        self.commands.extend(cmds);
    }

    /// Number of pending commands.
    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    /// Clear all pending commands (call after processing a frame).
    pub fn clear(&mut self) {
        self.commands.clear();
    }

    /// Sort commands and return them in draw order.
    /// Opaque commands are front-to-back, transparent are back-to-front.
    pub fn sorted(&mut self) -> Vec<RenderCommand> {
        let (mut opaque, mut transparent): (Vec<_>, Vec<_>) = self.commands
            .drain(..)
            .partition(|cmd| !cmd.blend.is_transparent());

        opaque.sort_by(|a, b| a.opaque_key().cmp(&b.opaque_key()));
        transparent.sort_by(|a, b| a.transparent_key().cmp(&b.transparent_key()));

        opaque.extend(transparent);
        opaque
    }

    /// Sort and batch consecutive commands that share the same material.
    pub fn sorted_batches(&mut self) -> Vec<RenderBatch> {
        let sorted = self.sorted();
        let mut batches: Vec<RenderBatch> = Vec::new();

        for cmd in sorted {
            let mat = cmd.material_id;
            let tri = cmd.triangle_count;
            if let Some(last) = batches.last_mut() {
                if last.material_id == mat {
                    last.total_triangles += tri;
                    last.commands.push(cmd);
                    continue;
                }
            }
            batches.push(RenderBatch {
                material_id: mat,
                commands: vec![cmd],
                total_triangles: tri,
            });
        }
        batches
    }

    /// Compute statistics from the current pending commands (does not sort/drain).
    pub fn stats(&self) -> RenderStats {
        let draw_calls = self.commands.len() as u32;
        let triangles: u32 = self.commands.iter().map(|c| c.triangle_count).sum();
        let opaque_calls = self.commands.iter().filter(|c| !c.blend.is_transparent()).count() as u32;
        let transparent_calls = draw_calls - opaque_calls;

        // Estimate batch count by material grouping (exact after sort).
        let mut sorted_mats: Vec<u64> = self.commands.iter().map(|c| c.material_id).collect();
        sorted_mats.sort();
        let batches = if sorted_mats.is_empty() {
            0u32
        } else {
            let mut count = 1u32;
            for w in sorted_mats.windows(2) {
                if w[0] != w[1] {
                    count += 1;
                }
            }
            count
        };

        RenderStats { draw_calls, batches, triangles, opaque_calls, transparent_calls }
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn identity() -> [f64; 16] {
        let mut m = [0.0; 16];
        m[0] = 1.0; m[5] = 1.0; m[10] = 1.0; m[15] = 1.0;
        m
    }

    fn cmd(mat: u64, mesh: u64, depth: f64, tris: u32) -> RenderCommand {
        RenderCommand::new(mat, mesh, identity())
            .with_depth_sq(depth)
            .with_triangles(tris)
    }

    #[test]
    fn test_submit_and_len() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 0.0, 10));
        q.submit(cmd(2, 2, 0.0, 20));
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_clear() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 0.0, 10));
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn test_opaque_front_to_back() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 100.0, 10)); // far
        q.submit(cmd(1, 1, 1.0, 10));   // near
        q.submit(cmd(1, 1, 50.0, 10));  // mid
        let sorted = q.sorted();
        assert!(sorted[0].depth_sq <= sorted[1].depth_sq);
        assert!(sorted[1].depth_sq <= sorted[2].depth_sq);
    }

    #[test]
    fn test_transparent_back_to_front() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 1.0, 10).with_blend(BlendMode::AlphaBlend));
        q.submit(cmd(1, 1, 100.0, 10).with_blend(BlendMode::AlphaBlend));
        q.submit(cmd(1, 1, 50.0, 10).with_blend(BlendMode::AlphaBlend));
        let sorted = q.sorted();
        assert!(sorted[0].depth_sq >= sorted[1].depth_sq);
        assert!(sorted[1].depth_sq >= sorted[2].depth_sq);
    }

    #[test]
    fn test_opaque_before_transparent() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 10.0, 10).with_blend(BlendMode::AlphaBlend)
            .with_layer(RenderLayer::TRANSPARENT));
        q.submit(cmd(2, 2, 5.0, 10).with_layer(RenderLayer::DEFAULT));
        let sorted = q.sorted();
        // Opaque (layer 0) should come before transparent (layer 100).
        assert_eq!(sorted[0].material_id, 2);
        assert_eq!(sorted[1].material_id, 1);
    }

    #[test]
    fn test_layer_ordering() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 0.0, 10).with_layer(RenderLayer::UI));
        q.submit(cmd(2, 2, 0.0, 10).with_layer(RenderLayer::BACKGROUND));
        q.submit(cmd(3, 3, 0.0, 10).with_layer(RenderLayer::DEFAULT));
        let sorted = q.sorted();
        assert_eq!(sorted[0].layer, RenderLayer::BACKGROUND);
        assert_eq!(sorted[1].layer, RenderLayer::DEFAULT);
        assert_eq!(sorted[2].layer, RenderLayer::UI);
    }

    #[test]
    fn test_batching_same_material() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 1.0, 10));
        q.submit(cmd(1, 2, 2.0, 20));
        q.submit(cmd(1, 3, 3.0, 30));
        let batches = q.sorted_batches();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].material_id, 1);
        assert_eq!(batches[0].total_triangles, 60);
        assert_eq!(batches[0].commands.len(), 3);
    }

    #[test]
    fn test_batching_different_materials() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 1.0, 10));
        q.submit(cmd(2, 2, 2.0, 20));
        q.submit(cmd(3, 3, 3.0, 30));
        let batches = q.sorted_batches();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_batching_interleaved_materials() {
        let mut q = RenderQueue::new();
        // Same material at different depths — after sorting, consecutive.
        q.submit(cmd(1, 1, 1.0, 10));
        q.submit(cmd(2, 2, 2.0, 20));
        q.submit(cmd(1, 3, 3.0, 30));
        let batches = q.sorted_batches();
        // After sorting by depth, order is mat1(depth1), mat2(depth2), mat1(depth3).
        // That breaks into 3 batches.
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_stats_basic() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 0.0, 100));
        q.submit(cmd(1, 2, 0.0, 200));
        q.submit(cmd(2, 3, 0.0, 50).with_blend(BlendMode::AlphaBlend));
        let stats = q.stats();
        assert_eq!(stats.draw_calls, 3);
        assert_eq!(stats.triangles, 350);
        assert_eq!(stats.opaque_calls, 2);
        assert_eq!(stats.transparent_calls, 1);
        assert_eq!(stats.batches, 2); // materials 1 and 2
    }

    #[test]
    fn test_stats_empty() {
        let q = RenderQueue::new();
        let stats = q.stats();
        assert_eq!(stats.draw_calls, 0);
        assert_eq!(stats.triangles, 0);
        assert_eq!(stats.batches, 0);
    }

    #[test]
    fn test_submit_many() {
        let mut q = RenderQueue::new();
        let cmds = vec![cmd(1, 1, 0.0, 10), cmd(2, 2, 0.0, 20)];
        q.submit_many(cmds);
        assert_eq!(q.len(), 2);
    }

    #[test]
    fn test_with_capacity() {
        let q = RenderQueue::with_capacity(1000);
        assert!(q.is_empty());
    }

    #[test]
    fn test_blend_mode_is_transparent() {
        assert!(!BlendMode::Opaque.is_transparent());
        assert!(BlendMode::AlphaBlend.is_transparent());
        assert!(BlendMode::Additive.is_transparent());
        assert!(BlendMode::Multiply.is_transparent());
    }

    #[test]
    fn test_render_layer_ordering() {
        assert!(RenderLayer::BACKGROUND < RenderLayer::DEFAULT);
        assert!(RenderLayer::DEFAULT < RenderLayer::TRANSPARENT);
        assert!(RenderLayer::TRANSPARENT < RenderLayer::OVERLAY);
        assert!(RenderLayer::OVERLAY < RenderLayer::UI);
    }

    #[test]
    fn test_sub_order_within_same_depth() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 5.0, 10).with_sub_order(2));
        q.submit(cmd(1, 2, 5.0, 10).with_sub_order(1));
        let sorted = q.sorted();
        assert_eq!(sorted[0].sub_order, 1);
        assert_eq!(sorted[1].sub_order, 2);
    }

    #[test]
    fn test_sorted_drains_queue() {
        let mut q = RenderQueue::new();
        q.submit(cmd(1, 1, 0.0, 10));
        let _sorted = q.sorted();
        assert!(q.is_empty());
    }

    #[test]
    fn test_large_batch() {
        let mut q = RenderQueue::new();
        for i in 0..1000 {
            q.submit(cmd(i % 5, i, (i as f64) * 0.1, 100));
        }
        let stats = q.stats();
        assert_eq!(stats.draw_calls, 1000);
        assert_eq!(stats.triangles, 100_000);
        assert_eq!(stats.batches, 5);
    }

    #[test]
    fn test_float_sort_key_ordering() {
        let a = float_to_sort_key(1.0);
        let b = float_to_sort_key(2.0);
        let c = float_to_sort_key(100.0);
        assert!(a < b);
        assert!(b < c);
    }

    #[test]
    fn test_command_builder() {
        let cmd = RenderCommand::new(42, 99, identity())
            .with_layer(RenderLayer::UI)
            .with_blend(BlendMode::Additive)
            .with_depth_sq(25.0)
            .with_triangles(500)
            .with_sub_order(3);
        assert_eq!(cmd.material_id, 42);
        assert_eq!(cmd.mesh_id, 99);
        assert_eq!(cmd.layer, RenderLayer::UI);
        assert_eq!(cmd.blend, BlendMode::Additive);
        assert!((cmd.depth_sq - 25.0).abs() < 1e-9);
        assert_eq!(cmd.triangle_count, 500);
        assert_eq!(cmd.sub_order, 3);
    }
}
