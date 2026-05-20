//! Render frame graph / render pass system.
//!
//! Define passes (shadow, gbuffer, lighting, post-process, UI) with
//! explicit resource dependencies (reads texture A, writes texture B).
//! Automatic pass ordering via topological sort. Resource lifetime
//! tracking. Pass culling (skip unused passes).

use std::collections::{HashMap, HashSet, VecDeque};

// ── Resource ────────────────────────────────────────────────────

/// A GPU resource (texture, buffer) tracked by the frame graph.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ResourceId(pub String);

impl ResourceId {
    pub fn new(name: &str) -> Self { Self(name.to_string()) }
}

/// Metadata about a resource.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceDesc {
    pub id: ResourceId,
    pub kind: ResourceKind,
    pub width: u32,
    pub height: u32,
    /// If true, the resource persists across frames (e.g., back buffer).
    pub persistent: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResourceKind {
    Texture2D,
    DepthBuffer,
    RenderTarget,
    StorageBuffer,
}

// ── Resource lifetime ───────────────────────────────────────────

/// Tracks when a resource is first written and last read.
#[derive(Debug, Clone, PartialEq)]
pub struct ResourceLifetime {
    pub resource: ResourceId,
    /// Index of the first pass that writes to this resource.
    pub first_write: usize,
    /// Index of the last pass that reads from this resource.
    pub last_read: Option<usize>,
}

// ── Render pass ─────────────────────────────────────────────────

type PassId = String;

/// A render pass with explicit resource dependencies.
#[derive(Debug, Clone, PartialEq)]
pub struct RenderPass {
    pub id: PassId,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
    pub enabled: bool,
    /// If true, this pass produces output consumed by the final present.
    pub is_output: bool,
}

impl RenderPass {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            reads: Vec::new(),
            writes: Vec::new(),
            enabled: true,
            is_output: false,
        }
    }

    pub fn reads(mut self, res: &str) -> Self {
        self.reads.push(ResourceId::new(res));
        self
    }

    pub fn writes(mut self, res: &str) -> Self {
        self.writes.push(ResourceId::new(res));
        self
    }

    pub fn output(mut self) -> Self {
        self.is_output = true;
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }
}

// ── Frame graph ─────────────────────────────────────────────────

/// A frame graph that manages render passes and their resource dependencies.
#[derive(Debug)]
pub struct FrameGraph {
    passes: Vec<RenderPass>,
    resources: HashMap<String, ResourceDesc>,
}

impl FrameGraph {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            resources: HashMap::new(),
        }
    }

    /// Register a resource that passes can read/write.
    pub fn add_resource(&mut self, desc: ResourceDesc) {
        self.resources.insert(desc.id.0.clone(), desc);
    }

    /// Add a render pass.
    pub fn add_pass(&mut self, pass: RenderPass) {
        self.passes.push(pass);
    }

    pub fn pass_count(&self) -> usize {
        self.passes.len()
    }

    pub fn resource_count(&self) -> usize {
        self.resources.len()
    }

    pub fn get_pass(&self, id: &str) -> Option<&RenderPass> {
        self.passes.iter().find(|p| p.id == id)
    }

    pub fn get_pass_mut(&mut self, id: &str) -> Option<&mut RenderPass> {
        self.passes.iter_mut().find(|p| p.id == id)
    }

    pub fn get_resource(&self, id: &str) -> Option<&ResourceDesc> {
        self.resources.get(id)
    }

    /// Enable or disable a pass by ID.
    pub fn set_pass_enabled(&mut self, id: &str, enabled: bool) -> bool {
        if let Some(p) = self.passes.iter_mut().find(|p| p.id == id) {
            p.enabled = enabled;
            true
        } else {
            false
        }
    }

    // ── Topological sort ────────────────────────────────────────

    /// Compute a valid execution order for enabled passes via topological
    /// sort. Returns the ordered pass IDs, or None if there is a cycle.
    pub fn compile(&self) -> Option<Vec<PassId>> {
        let enabled: Vec<&RenderPass> = self.passes.iter().filter(|p| p.enabled).collect();
        let pass_ids: Vec<&str> = enabled.iter().map(|p| p.id.as_str()).collect();
        let index_of = |id: &str| -> Option<usize> {
            pass_ids.iter().position(|pid| *pid == id)
        };

        let n = enabled.len();
        // Build adjacency: pass A writes resource R, pass B reads resource R => A -> B.
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut in_degree: Vec<usize> = vec![0; n];

        // Map resource -> writer pass index.
        let mut writers: HashMap<&str, Vec<usize>> = HashMap::new();
        for (i, pass) in enabled.iter().enumerate() {
            for w in &pass.writes {
                writers.entry(w.0.as_str()).or_default().push(i);
            }
        }

        for (i, pass) in enabled.iter().enumerate() {
            for r in &pass.reads {
                if let Some(writer_indices) = writers.get(r.0.as_str()) {
                    for &wi in writer_indices {
                        if wi != i {
                            adj[wi].push(i);
                            in_degree[i] += 1;
                        }
                    }
                }
            }
        }

        // Kahn's algorithm.
        let mut queue: VecDeque<usize> = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        let mut order: Vec<PassId> = Vec::new();
        while let Some(node) = queue.pop_front() {
            order.push(enabled[node].id.clone());
            for &next in &adj[node] {
                in_degree[next] -= 1;
                if in_degree[next] == 0 {
                    queue.push_back(next);
                }
            }
        }

        if order.len() == n {
            Some(order)
        } else {
            None // cycle detected
        }
    }

    // ── Pass culling ────────────────────────────────────────────

    /// Cull passes that do not contribute to any output pass.
    /// Returns the set of pass IDs that should execute.
    pub fn cull(&self) -> HashSet<PassId> {
        let enabled: Vec<&RenderPass> = self.passes.iter().filter(|p| p.enabled).collect();

        // Start from output passes and walk backwards through dependencies.
        let mut needed: HashSet<String> = HashSet::new();
        let mut stack: Vec<&str> = Vec::new();

        for pass in &enabled {
            if pass.is_output {
                needed.insert(pass.id.clone());
                stack.push(&pass.id);
            }
        }

        // Map resource -> writer pass IDs.
        let mut writers: HashMap<&str, Vec<&str>> = HashMap::new();
        for pass in &enabled {
            for w in &pass.writes {
                writers.entry(w.0.as_str()).or_default().push(&pass.id);
            }
        }

        while let Some(pid) = stack.pop() {
            let pass = match enabled.iter().find(|p| p.id == pid) {
                Some(p) => p,
                None => continue,
            };
            for r in &pass.reads {
                if let Some(w_ids) = writers.get(r.0.as_str()) {
                    for wid in w_ids {
                        if needed.insert(wid.to_string()) {
                            stack.push(wid);
                        }
                    }
                }
            }
        }

        needed
    }

    /// Compile with culling: only emit passes that contribute to output.
    pub fn compile_culled(&self) -> Option<Vec<PassId>> {
        let needed = self.cull();
        let compiled = self.compile()?;
        Some(compiled.into_iter().filter(|id| needed.contains(id)).collect())
    }

    // ── Resource lifetime analysis ──────────────────────────────

    /// Compute resource lifetimes given a compiled pass order.
    pub fn resource_lifetimes(&self, order: &[PassId]) -> Vec<ResourceLifetime> {
        let mut lifetimes: HashMap<String, ResourceLifetime> = HashMap::new();
        let pass_index = |id: &str| -> Option<usize> {
            order.iter().position(|pid| pid == id)
        };

        let enabled: Vec<&RenderPass> = self.passes.iter().filter(|p| p.enabled).collect();

        for pass in &enabled {
            let idx = match pass_index(&pass.id) {
                Some(i) => i,
                None => continue,
            };
            for w in &pass.writes {
                let lt = lifetimes.entry(w.0.clone()).or_insert(ResourceLifetime {
                    resource: w.clone(),
                    first_write: idx,
                    last_read: None,
                });
                lt.first_write = lt.first_write.min(idx);
            }
            for r in &pass.reads {
                let lt = lifetimes.entry(r.0.clone()).or_insert(ResourceLifetime {
                    resource: r.clone(),
                    first_write: idx,
                    last_read: None,
                });
                lt.last_read = Some(match lt.last_read {
                    Some(prev) => prev.max(idx),
                    None => idx,
                });
            }
        }

        // Collect and sort by first_write.
        let mut result: Vec<ResourceLifetime> = lifetimes.into_values().collect();
        result.sort_by_key(|lt| lt.first_write);
        result
    }

    /// Remove all passes and resources.
    pub fn clear(&mut self) {
        self.passes.clear();
        self.resources.clear();
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> FrameGraph {
        let mut fg = FrameGraph::new();
        fg.add_resource(ResourceDesc {
            id: ResourceId::new("depth"),
            kind: ResourceKind::DepthBuffer,
            width: 1920, height: 1080, persistent: false,
        });
        fg.add_resource(ResourceDesc {
            id: ResourceId::new("gbuffer"),
            kind: ResourceKind::RenderTarget,
            width: 1920, height: 1080, persistent: false,
        });
        fg.add_resource(ResourceDesc {
            id: ResourceId::new("shadow_map"),
            kind: ResourceKind::Texture2D,
            width: 2048, height: 2048, persistent: false,
        });
        fg.add_resource(ResourceDesc {
            id: ResourceId::new("hdr"),
            kind: ResourceKind::RenderTarget,
            width: 1920, height: 1080, persistent: false,
        });
        fg.add_resource(ResourceDesc {
            id: ResourceId::new("backbuffer"),
            kind: ResourceKind::RenderTarget,
            width: 1920, height: 1080, persistent: true,
        });

        fg.add_pass(RenderPass::new("shadow").writes("shadow_map"));
        fg.add_pass(RenderPass::new("gbuffer").reads("shadow_map").writes("gbuffer").writes("depth"));
        fg.add_pass(RenderPass::new("lighting").reads("gbuffer").reads("depth").writes("hdr"));
        fg.add_pass(RenderPass::new("postprocess").reads("hdr").writes("backbuffer").output());

        fg
    }

    #[test]
    fn test_add_pass() {
        let fg = make_graph();
        assert_eq!(fg.pass_count(), 4);
    }

    #[test]
    fn test_add_resource() {
        let fg = make_graph();
        assert_eq!(fg.resource_count(), 5);
    }

    #[test]
    fn test_get_pass() {
        let fg = make_graph();
        let p = fg.get_pass("shadow").unwrap();
        assert_eq!(p.id, "shadow");
        assert!(fg.get_pass("nonexistent").is_none());
    }

    #[test]
    fn test_get_resource() {
        let fg = make_graph();
        let r = fg.get_resource("depth").unwrap();
        assert_eq!(r.kind, ResourceKind::DepthBuffer);
    }

    #[test]
    fn test_compile_linear_pipeline() {
        let fg = make_graph();
        let order = fg.compile().unwrap();
        assert_eq!(order.len(), 4);
        // shadow must come before gbuffer.
        let si = order.iter().position(|id| id == "shadow").unwrap();
        let gi = order.iter().position(|id| id == "gbuffer").unwrap();
        let li = order.iter().position(|id| id == "lighting").unwrap();
        let pi = order.iter().position(|id| id == "postprocess").unwrap();
        assert!(si < gi);
        assert!(gi < li);
        assert!(li < pi);
    }

    #[test]
    fn test_compile_cycle_detection() {
        let mut fg = FrameGraph::new();
        fg.add_pass(RenderPass::new("a").reads("r1").writes("r2"));
        fg.add_pass(RenderPass::new("b").reads("r2").writes("r1"));
        assert!(fg.compile().is_none());
    }

    #[test]
    fn test_compile_empty() {
        let fg = FrameGraph::new();
        let order = fg.compile().unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn test_compile_single_pass() {
        let mut fg = FrameGraph::new();
        fg.add_pass(RenderPass::new("only").writes("out").output());
        let order = fg.compile().unwrap();
        assert_eq!(order, vec!["only"]);
    }

    #[test]
    fn test_disabled_pass_excluded() {
        let mut fg = make_graph();
        fg.set_pass_enabled("shadow", false);
        let order = fg.compile().unwrap();
        assert!(!order.contains(&"shadow".to_string()));
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn test_cull_removes_unused() {
        let mut fg = FrameGraph::new();
        fg.add_pass(RenderPass::new("shadow").writes("shadow_map"));
        fg.add_pass(RenderPass::new("main").reads("shadow_map").writes("hdr"));
        fg.add_pass(RenderPass::new("post").reads("hdr").writes("back").output());
        // Add a completely disconnected pass.
        fg.add_pass(RenderPass::new("debug_vis").writes("debug_tex"));
        let needed = fg.cull();
        assert!(needed.contains("shadow"));
        assert!(needed.contains("main"));
        assert!(needed.contains("post"));
        assert!(!needed.contains("debug_vis"));
    }

    #[test]
    fn test_cull_no_output() {
        let mut fg = FrameGraph::new();
        fg.add_pass(RenderPass::new("a").writes("r1"));
        // No output pass — nothing is needed.
        let needed = fg.cull();
        assert!(needed.is_empty());
    }

    #[test]
    fn test_compile_culled() {
        let mut fg = FrameGraph::new();
        fg.add_pass(RenderPass::new("shadow").writes("sm"));
        fg.add_pass(RenderPass::new("main").reads("sm").writes("hdr"));
        fg.add_pass(RenderPass::new("post").reads("hdr").writes("back").output());
        fg.add_pass(RenderPass::new("unused").writes("junk"));
        let order = fg.compile_culled().unwrap();
        assert_eq!(order.len(), 3);
        assert!(!order.contains(&"unused".to_string()));
    }

    #[test]
    fn test_resource_lifetimes() {
        let fg = make_graph();
        let order = fg.compile().unwrap();
        let lifetimes = fg.resource_lifetimes(&order);
        assert!(!lifetimes.is_empty());
        // shadow_map: written by shadow (index 0), read by gbuffer (index 1).
        let sm_lt = lifetimes.iter().find(|lt| lt.resource.0 == "shadow_map").unwrap();
        assert_eq!(sm_lt.first_write, 0);
        assert_eq!(sm_lt.last_read, Some(1));
    }

    #[test]
    fn test_resource_lifetime_no_readers() {
        let mut fg = FrameGraph::new();
        fg.add_pass(RenderPass::new("gen").writes("tex").output());
        let order = fg.compile().unwrap();
        let lifetimes = fg.resource_lifetimes(&order);
        let lt = &lifetimes[0];
        assert_eq!(lt.first_write, 0);
        assert!(lt.last_read.is_none());
    }

    #[test]
    fn test_parallel_passes() {
        let mut fg = FrameGraph::new();
        // Two passes that write independent resources, both read by a merge pass.
        fg.add_pass(RenderPass::new("gen_a").writes("a"));
        fg.add_pass(RenderPass::new("gen_b").writes("b"));
        fg.add_pass(RenderPass::new("merge").reads("a").reads("b").writes("out").output());
        let order = fg.compile().unwrap();
        assert_eq!(order.len(), 3);
        let mi = order.iter().position(|id| id == "merge").unwrap();
        let ai = order.iter().position(|id| id == "gen_a").unwrap();
        let bi = order.iter().position(|id| id == "gen_b").unwrap();
        assert!(ai < mi);
        assert!(bi < mi);
    }

    #[test]
    fn test_set_pass_enabled() {
        let mut fg = make_graph();
        assert!(fg.set_pass_enabled("shadow", false));
        assert!(!fg.get_pass("shadow").unwrap().enabled);
        assert!(!fg.set_pass_enabled("nonexistent", true));
    }

    #[test]
    fn test_clear() {
        let mut fg = make_graph();
        fg.clear();
        assert_eq!(fg.pass_count(), 0);
        assert_eq!(fg.resource_count(), 0);
    }

    #[test]
    fn test_render_pass_builder() {
        let pass = RenderPass::new("test")
            .reads("input")
            .writes("output")
            .output()
            .disabled();
        assert_eq!(pass.reads.len(), 1);
        assert_eq!(pass.writes.len(), 1);
        assert!(pass.is_output);
        assert!(!pass.enabled);
    }

    #[test]
    fn test_resource_id_equality() {
        let a = ResourceId::new("tex");
        let b = ResourceId::new("tex");
        assert_eq!(a, b);
    }

    #[test]
    fn test_get_pass_mut() {
        let mut fg = make_graph();
        fg.get_pass_mut("shadow").unwrap().is_output = true;
        assert!(fg.get_pass("shadow").unwrap().is_output);
    }

    #[test]
    fn test_diamond_dependency() {
        let mut fg = FrameGraph::new();
        fg.add_pass(RenderPass::new("source").writes("shared"));
        fg.add_pass(RenderPass::new("branch_a").reads("shared").writes("a_out"));
        fg.add_pass(RenderPass::new("branch_b").reads("shared").writes("b_out"));
        fg.add_pass(RenderPass::new("combine").reads("a_out").reads("b_out").writes("final").output());
        let order = fg.compile().unwrap();
        assert_eq!(order.len(), 4);
        let src = order.iter().position(|id| id == "source").unwrap();
        let ba = order.iter().position(|id| id == "branch_a").unwrap();
        let bb = order.iter().position(|id| id == "branch_b").unwrap();
        let comb = order.iter().position(|id| id == "combine").unwrap();
        assert!(src < ba);
        assert!(src < bb);
        assert!(ba < comb);
        assert!(bb < comb);
    }
}
