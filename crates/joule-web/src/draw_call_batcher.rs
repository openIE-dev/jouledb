//! Draw call optimization via instanced batching.
//!
//! Merges compatible draw calls (same material + mesh) into instanced
//! draws. Instance data buffer management. Hybrid batching: static
//! batching (pre-merged meshes) and dynamic batching (per-frame
//! instancing). Batch break detection.

use std::collections::HashMap;

// ── Instance data ───────────────────────────────────────────────

/// Per-instance data attached to a draw call.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceData {
    /// 4x4 column-major model transform.
    pub transform: [f64; 16],
    /// RGBA colour tint.
    pub color: [f32; 4],
    /// Arbitrary user data (UV offset, animation frame, etc.).
    pub custom: [f32; 4],
}

impl InstanceData {
    pub fn from_transform(transform: [f64; 16]) -> Self {
        Self {
            transform,
            color: [1.0, 1.0, 1.0, 1.0],
            custom: [0.0; 4],
        }
    }

    pub fn with_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }

    pub fn with_custom(mut self, v: [f32; 4]) -> Self {
        self.custom = v;
        self
    }
}

// ── Draw call ───────────────────────────────────────────────────

/// A single un-batched draw call.
#[derive(Debug, Clone, PartialEq)]
pub struct DrawCall {
    pub material_id: u64,
    pub mesh_id: u64,
    pub instance: InstanceData,
    pub triangle_count: u32,
    /// Render order (lower = earlier).
    pub order: i32,
}

// ── Batch key ───────────────────────────────────────────────────

/// Key used to group compatible draw calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BatchKey {
    pub material_id: u64,
    pub mesh_id: u64,
}

// ── Instanced batch ─────────────────────────────────────────────

/// A batch of instances sharing the same material and mesh.
#[derive(Debug, Clone, PartialEq)]
pub struct InstancedBatch {
    pub key: BatchKey,
    pub instances: Vec<InstanceData>,
    pub triangle_count: u32,
    pub order: i32,
}

impl InstancedBatch {
    pub fn instance_count(&self) -> usize {
        self.instances.len()
    }

    /// Total triangles = per-mesh tris * instance count.
    pub fn total_triangles(&self) -> u64 {
        self.triangle_count as u64 * self.instances.len() as u64
    }
}

// ── Static batch ────────────────────────────────────────────────

/// A pre-merged (static) batch — geometry is baked once.
#[derive(Debug, Clone, PartialEq)]
pub struct StaticBatch {
    pub id: u64,
    pub material_id: u64,
    pub merged_vertex_count: u32,
    pub merged_triangle_count: u32,
    pub transform: [f64; 16],
}

// ── Batch break ─────────────────────────────────────────────────

/// Reasons a batch must be broken (cannot merge with previous).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchBreak {
    None,
    MaterialChange,
    MeshChange,
    OrderChange,
    BufferFull,
}

fn detect_break(prev: &DrawCall, curr: &DrawCall, max_instances: usize, current_count: usize) -> BatchBreak {
    if current_count >= max_instances {
        return BatchBreak::BufferFull;
    }
    if prev.material_id != curr.material_id {
        return BatchBreak::MaterialChange;
    }
    if prev.mesh_id != curr.mesh_id {
        return BatchBreak::MeshChange;
    }
    if prev.order != curr.order {
        return BatchBreak::OrderChange;
    }
    BatchBreak::None
}

// ── Batcher statistics ──────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BatcherStats {
    pub input_draw_calls: u32,
    pub output_batches: u32,
    pub total_instances: u32,
    pub total_triangles: u64,
    pub batch_breaks: u32,
    pub static_batches: u32,
}

// ── Draw call batcher ───────────────────────────────────────────

/// Optimises draw calls by merging compatible calls into instanced batches.
#[derive(Debug)]
pub struct DrawCallBatcher {
    /// Maximum instances per batch before forcing a break.
    pub max_instances_per_batch: usize,
    draw_calls: Vec<DrawCall>,
    static_batches: Vec<StaticBatch>,
    next_static_id: u64,
}

impl DrawCallBatcher {
    pub fn new() -> Self {
        Self {
            max_instances_per_batch: 1024,
            draw_calls: Vec::new(),
            static_batches: Vec::new(),
            next_static_id: 1,
        }
    }

    pub fn with_max_instances(mut self, max: usize) -> Self {
        self.max_instances_per_batch = max;
        self
    }

    // ── Dynamic (per-frame) batching ────────────────────────────

    /// Submit a draw call for the current frame.
    pub fn submit(&mut self, call: DrawCall) {
        self.draw_calls.push(call);
    }

    pub fn submit_many(&mut self, calls: impl IntoIterator<Item = DrawCall>) {
        self.draw_calls.extend(calls);
    }

    pub fn pending_count(&self) -> usize {
        self.draw_calls.len()
    }

    /// Clear pending draw calls (call after flush/batch).
    pub fn clear(&mut self) {
        self.draw_calls.clear();
    }

    /// Batch pending draw calls into instanced batches.
    /// Calls are first sorted by (order, material, mesh), then
    /// consecutive compatible calls are merged.
    pub fn batch(&mut self) -> (Vec<InstancedBatch>, BatcherStats) {
        let mut calls = std::mem::take(&mut self.draw_calls);
        let input_count = calls.len() as u32;

        calls.sort_by(|a, b| {
            a.order.cmp(&b.order)
                .then(a.material_id.cmp(&b.material_id))
                .then(a.mesh_id.cmp(&b.mesh_id))
        });

        let mut batches: Vec<InstancedBatch> = Vec::new();
        let mut break_count: u32 = 0;

        for call in calls {
            let should_break = if let Some(last_batch) = batches.last() {
                let prev_call = DrawCall {
                    material_id: last_batch.key.material_id,
                    mesh_id: last_batch.key.mesh_id,
                    instance: InstanceData::from_transform([0.0; 16]),
                    triangle_count: last_batch.triangle_count,
                    order: last_batch.order,
                };
                detect_break(&prev_call, &call, self.max_instances_per_batch, last_batch.instance_count())
            } else {
                BatchBreak::None
            };

            match should_break {
                BatchBreak::None if !batches.is_empty() => {
                    let last = batches.last_mut().unwrap();
                    last.instances.push(call.instance);
                }
                _ => {
                    if !batches.is_empty() {
                        break_count += 1;
                    }
                    batches.push(InstancedBatch {
                        key: BatchKey { material_id: call.material_id, mesh_id: call.mesh_id },
                        instances: vec![call.instance],
                        triangle_count: call.triangle_count,
                        order: call.order,
                    });
                }
            }
        }

        let total_instances: u32 = batches.iter().map(|b| b.instance_count() as u32).sum();
        let total_triangles: u64 = batches.iter().map(|b| b.total_triangles()).sum();

        let stats = BatcherStats {
            input_draw_calls: input_count,
            output_batches: batches.len() as u32,
            total_instances,
            total_triangles,
            batch_breaks: break_count,
            static_batches: self.static_batches.len() as u32,
        };

        (batches, stats)
    }

    // ── Static batching ─────────────────────────────────────────

    /// Register a pre-merged static batch (baked geometry).
    pub fn add_static_batch(&mut self, material_id: u64, vertex_count: u32, tri_count: u32, transform: [f64; 16]) -> u64 {
        let id = self.next_static_id;
        self.next_static_id += 1;
        self.static_batches.push(StaticBatch {
            id,
            material_id,
            merged_vertex_count: vertex_count,
            merged_triangle_count: tri_count,
            transform,
        });
        id
    }

    pub fn remove_static_batch(&mut self, id: u64) -> bool {
        let len = self.static_batches.len();
        self.static_batches.retain(|b| b.id != id);
        self.static_batches.len() < len
    }

    pub fn static_batch_count(&self) -> usize {
        self.static_batches.len()
    }

    pub fn static_batches(&self) -> &[StaticBatch] {
        &self.static_batches
    }

    // ── Hybrid flush ────────────────────────────────────────────

    /// Collect static batches as single-instance InstancedBatches,
    /// appended after the dynamic batches.
    pub fn hybrid_batch(&mut self) -> (Vec<InstancedBatch>, BatcherStats) {
        let (mut dynamic, mut stats) = self.batch();

        for sb in &self.static_batches {
            dynamic.push(InstancedBatch {
                key: BatchKey { material_id: sb.material_id, mesh_id: 0 },
                instances: vec![InstanceData::from_transform(sb.transform)],
                triangle_count: sb.merged_triangle_count,
                order: i32::MAX, // render after dynamic
            });
            stats.total_triangles += sb.merged_triangle_count as u64;
            stats.total_instances += 1;
        }
        stats.output_batches += self.static_batches.len() as u32;
        stats.static_batches = self.static_batches.len() as u32;

        (dynamic, stats)
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

    fn dc(mat: u64, mesh: u64, tris: u32, order: i32) -> DrawCall {
        DrawCall {
            material_id: mat,
            mesh_id: mesh,
            instance: InstanceData::from_transform(identity()),
            triangle_count: tris,
            order,
        }
    }

    #[test]
    fn test_instance_data_from_transform() {
        let inst = InstanceData::from_transform(identity());
        assert_eq!(inst.color, [1.0, 1.0, 1.0, 1.0]);
        assert_eq!(inst.custom, [0.0; 4]);
    }

    #[test]
    fn test_instance_data_with_color() {
        let inst = InstanceData::from_transform(identity()).with_color(1.0, 0.0, 0.0, 0.5);
        assert_eq!(inst.color, [1.0, 0.0, 0.0, 0.5]);
    }

    #[test]
    fn test_instance_data_with_custom() {
        let inst = InstanceData::from_transform(identity()).with_custom([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(inst.custom, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_single_draw_call() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit(dc(1, 1, 100, 0));
        let (batches, stats) = batcher.batch();
        assert_eq!(batches.len(), 1);
        assert_eq!(stats.input_draw_calls, 1);
        assert_eq!(stats.output_batches, 1);
        assert_eq!(stats.total_instances, 1);
    }

    #[test]
    fn test_merge_same_material_mesh() {
        let mut batcher = DrawCallBatcher::new();
        for _ in 0..5 {
            batcher.submit(dc(1, 1, 100, 0));
        }
        let (batches, stats) = batcher.batch();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].instance_count(), 5);
        assert_eq!(stats.total_instances, 5);
        assert_eq!(stats.total_triangles, 500);
    }

    #[test]
    fn test_different_materials_separate() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit(dc(1, 1, 100, 0));
        batcher.submit(dc(2, 1, 100, 0));
        let (batches, _) = batcher.batch();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_different_meshes_separate() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit(dc(1, 1, 100, 0));
        batcher.submit(dc(1, 2, 100, 0));
        let (batches, _) = batcher.batch();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_order_break() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit(dc(1, 1, 100, 0));
        batcher.submit(dc(1, 1, 100, 1));
        let (batches, stats) = batcher.batch();
        assert_eq!(batches.len(), 2);
        assert!(stats.batch_breaks > 0);
    }

    #[test]
    fn test_buffer_full_break() {
        let mut batcher = DrawCallBatcher::new().with_max_instances(3);
        for _ in 0..5 {
            batcher.submit(dc(1, 1, 10, 0));
        }
        let (batches, _) = batcher.batch();
        assert_eq!(batches.len(), 2); // 3 + 2
        assert_eq!(batches[0].instance_count(), 3);
        assert_eq!(batches[1].instance_count(), 2);
    }

    #[test]
    fn test_batch_clears_pending() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit(dc(1, 1, 100, 0));
        batcher.batch();
        assert_eq!(batcher.pending_count(), 0);
    }

    #[test]
    fn test_submit_many() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit_many(vec![dc(1, 1, 10, 0), dc(2, 2, 20, 0)]);
        assert_eq!(batcher.pending_count(), 2);
    }

    #[test]
    fn test_static_batch_add_remove() {
        let mut batcher = DrawCallBatcher::new();
        let id = batcher.add_static_batch(1, 1000, 500, identity());
        assert_eq!(batcher.static_batch_count(), 1);
        assert!(batcher.remove_static_batch(id));
        assert_eq!(batcher.static_batch_count(), 0);
    }

    #[test]
    fn test_static_batch_remove_nonexistent() {
        let mut batcher = DrawCallBatcher::new();
        assert!(!batcher.remove_static_batch(999));
    }

    #[test]
    fn test_hybrid_batch() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit(dc(1, 1, 100, 0));
        batcher.add_static_batch(2, 500, 200, identity());
        let (batches, stats) = batcher.hybrid_batch();
        assert_eq!(batches.len(), 2); // 1 dynamic + 1 static
        assert_eq!(stats.static_batches, 1);
        assert_eq!(stats.total_triangles, 300); // 100 + 200
    }

    #[test]
    fn test_instanced_batch_total_triangles() {
        let batch = InstancedBatch {
            key: BatchKey { material_id: 1, mesh_id: 1 },
            instances: vec![
                InstanceData::from_transform(identity()),
                InstanceData::from_transform(identity()),
                InstanceData::from_transform(identity()),
            ],
            triangle_count: 100,
            order: 0,
        };
        assert_eq!(batch.total_triangles(), 300);
    }

    #[test]
    fn test_detect_break_none() {
        let a = dc(1, 1, 100, 0);
        let b = dc(1, 1, 100, 0);
        assert_eq!(detect_break(&a, &b, 1024, 1), BatchBreak::None);
    }

    #[test]
    fn test_detect_break_material() {
        let a = dc(1, 1, 100, 0);
        let b = dc(2, 1, 100, 0);
        assert_eq!(detect_break(&a, &b, 1024, 1), BatchBreak::MaterialChange);
    }

    #[test]
    fn test_detect_break_mesh() {
        let a = dc(1, 1, 100, 0);
        let b = dc(1, 2, 100, 0);
        assert_eq!(detect_break(&a, &b, 1024, 1), BatchBreak::MeshChange);
    }

    #[test]
    fn test_detect_break_order() {
        let a = dc(1, 1, 100, 0);
        let b = dc(1, 1, 100, 5);
        assert_eq!(detect_break(&a, &b, 1024, 1), BatchBreak::OrderChange);
    }

    #[test]
    fn test_detect_break_buffer_full() {
        let a = dc(1, 1, 100, 0);
        let b = dc(1, 1, 100, 0);
        assert_eq!(detect_break(&a, &b, 10, 10), BatchBreak::BufferFull);
    }

    #[test]
    fn test_large_dynamic_batch() {
        let mut batcher = DrawCallBatcher::new();
        for i in 0..500 {
            batcher.submit(dc(i % 3, i % 2, 10, 0));
        }
        let (batches, stats) = batcher.batch();
        assert!(batches.len() <= 6); // 3 mats * 2 meshes
        assert_eq!(stats.input_draw_calls, 500);
        assert_eq!(stats.total_instances, 500);
    }

    #[test]
    fn test_static_batches_accessor() {
        let mut batcher = DrawCallBatcher::new();
        batcher.add_static_batch(1, 100, 50, identity());
        batcher.add_static_batch(2, 200, 100, identity());
        let sb = batcher.static_batches();
        assert_eq!(sb.len(), 2);
        assert_eq!(sb[0].material_id, 1);
        assert_eq!(sb[1].material_id, 2);
    }

    #[test]
    fn test_empty_batch() {
        let mut batcher = DrawCallBatcher::new();
        let (batches, stats) = batcher.batch();
        assert!(batches.is_empty());
        assert_eq!(stats.input_draw_calls, 0);
        assert_eq!(stats.output_batches, 0);
    }

    #[test]
    fn test_clear() {
        let mut batcher = DrawCallBatcher::new();
        batcher.submit(dc(1, 1, 100, 0));
        batcher.clear();
        assert_eq!(batcher.pending_count(), 0);
    }
}
