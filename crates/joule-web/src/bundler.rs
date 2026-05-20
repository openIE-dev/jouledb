//! Module bundler — dependency graph, topological sort, chunk splitting, and cache busting.
//!
//! Replaces Webpack / Rollup / esbuild bundling logic with a pure Rust model.
//! No filesystem access — operates on in-memory module descriptors.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Module ──────────────────────────────────────────────────────

/// A module in the bundle graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Module {
    pub id: u64,
    pub path: String,
    pub source: String,
    pub dependencies: Vec<u64>,
}

impl Module {
    pub fn new(id: u64, path: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            id,
            path: path.into(),
            source: source.into(),
            dependencies: Vec::new(),
        }
    }

    /// Content hash of the source for cache busting.
    pub fn content_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in self.source.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x100000001b3);
        }
        h
    }
}

// ── Chunk ───────────────────────────────────────────────────────

/// A chunk produced by the bundler (one per entry point + shared).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub name: String,
    pub module_ids: Vec<u64>,
    pub is_shared: bool,
}

impl Chunk {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            module_ids: Vec::new(),
            is_shared: false,
        }
    }

    /// Fingerprinted chunk filename.
    pub fn fingerprinted_name(&self, hash: u64) -> String {
        format!("{}.{:016x}.js", self.name, hash)
    }
}

// ── Bundle Manifest ─────────────────────────────────────────────

/// Maps chunk names to their constituent modules.
#[derive(Debug, Clone, Default)]
pub struct BundleManifest {
    pub chunks: Vec<Chunk>,
    pub module_index: HashMap<u64, usize>, // module_id -> chunk index
}

impl fmt::Display for BundleManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for chunk in &self.chunks {
            writeln!(f, "chunk {:?}: {} modules", chunk.name, chunk.module_ids.len())?;
        }
        Ok(())
    }
}

// ── Dependency Graph ────────────────────────────────────────────

/// The dependency graph and bundling engine.
#[derive(Debug, Clone, Default)]
pub struct DependencyGraph {
    modules: HashMap<u64, Module>,
}

/// Errors during bundling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BundleError {
    CircularDependency(Vec<u64>),
    MissingModule(u64),
}

impl fmt::Display for BundleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BundleError::CircularDependency(ids) => {
                write!(f, "circular dependency: {ids:?}")
            }
            BundleError::MissingModule(id) => {
                write!(f, "missing module: {id}")
            }
        }
    }
}

impl std::error::Error for BundleError {}

impl DependencyGraph {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a module to the graph.
    pub fn add_module(&mut self, module: Module) {
        self.modules.insert(module.id, module);
    }

    /// Get a module by ID.
    pub fn get_module(&self, id: u64) -> Option<&Module> {
        self.modules.get(&id)
    }

    /// All module IDs.
    pub fn module_ids(&self) -> Vec<u64> {
        let mut ids: Vec<u64> = self.modules.keys().copied().collect();
        ids.sort();
        ids
    }

    /// Detect circular dependencies. Returns the cycle path if found.
    pub fn detect_cycles(&self) -> Option<Vec<u64>> {
        let mut visited = HashSet::new();
        let mut on_stack = HashSet::new();
        let mut path = Vec::new();

        for &id in self.modules.keys() {
            if !visited.contains(&id) {
                if let Some(cycle) = self.dfs_cycle(id, &mut visited, &mut on_stack, &mut path) {
                    return Some(cycle);
                }
            }
        }
        None
    }

    fn dfs_cycle(
        &self,
        id: u64,
        visited: &mut HashSet<u64>,
        on_stack: &mut HashSet<u64>,
        path: &mut Vec<u64>,
    ) -> Option<Vec<u64>> {
        visited.insert(id);
        on_stack.insert(id);
        path.push(id);

        if let Some(module) = self.modules.get(&id) {
            for &dep in &module.dependencies {
                if !visited.contains(&dep) {
                    if let Some(cycle) = self.dfs_cycle(dep, visited, on_stack, path) {
                        return Some(cycle);
                    }
                } else if on_stack.contains(&dep) {
                    // Found cycle — extract it
                    let start = path.iter().position(|x| *x == dep).unwrap_or(0);
                    let mut cycle: Vec<u64> = path[start..].to_vec();
                    cycle.push(dep);
                    return Some(cycle);
                }
            }
        }

        path.pop();
        on_stack.remove(&id);
        None
    }

    /// Topological sort (Kahn's algorithm). Returns error on cycle.
    pub fn topological_sort(&self) -> Result<Vec<u64>, BundleError> {
        let mut in_degree: HashMap<u64, usize> = HashMap::new();
        for &id in self.modules.keys() {
            in_degree.entry(id).or_insert(0);
        }
        for module in self.modules.values() {
            for &dep in &module.dependencies {
                *in_degree.entry(dep).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<u64> = VecDeque::new();
        let mut sorted_ids: Vec<(u64, usize)> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(&id, &deg)| (id, deg))
            .collect();
        sorted_ids.sort_by_key(|&(id, _)| id);
        for (id, _) in sorted_ids {
            queue.push_back(id);
        }

        let mut order = Vec::new();
        while let Some(id) = queue.pop_front() {
            order.push(id);
            if let Some(module) = self.modules.get(&id) {
                let mut deps: Vec<u64> = module.dependencies.clone();
                deps.sort();
                for dep in deps {
                    if let Some(deg) = in_degree.get_mut(&dep) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(dep);
                        }
                    }
                }
            }
        }

        if order.len() != self.modules.len() {
            if let Some(cycle) = self.detect_cycles() {
                return Err(BundleError::CircularDependency(cycle));
            }
            // Fallback — shouldn't happen
            return Err(BundleError::CircularDependency(vec![]));
        }

        Ok(order)
    }

    /// Collect all transitive dependencies of a module.
    pub fn transitive_deps(&self, root: u64) -> HashSet<u64> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(id) = queue.pop_front() {
            if !visited.insert(id) {
                continue;
            }
            if let Some(module) = self.modules.get(&id) {
                for &dep in &module.dependencies {
                    queue.push_back(dep);
                }
            }
        }
        visited.remove(&root);
        visited
    }

    /// Split into chunks: one per entry point, plus shared.
    /// Modules referenced by 2+ entry points go into a shared chunk.
    pub fn split_chunks(&self, entry_points: &[u64]) -> BundleManifest {
        // Gather transitive deps per entry
        let mut entry_deps: Vec<HashSet<u64>> = Vec::new();
        for &ep in entry_points {
            let mut deps = self.transitive_deps(ep);
            deps.insert(ep);
            entry_deps.push(deps);
        }

        // Find shared modules (in 2+ entry chunks)
        let mut ref_count: HashMap<u64, usize> = HashMap::new();
        for deps in &entry_deps {
            for &id in deps {
                *ref_count.entry(id).or_insert(0) += 1;
            }
        }
        let shared: HashSet<u64> = ref_count
            .iter()
            .filter(|(_, count)| **count >= 2)
            .map(|(&id, _)| id)
            .collect();

        let mut manifest = BundleManifest::default();

        // Shared chunk first
        if !shared.is_empty() {
            let mut chunk = Chunk::new("shared");
            chunk.is_shared = true;
            let mut ids: Vec<u64> = shared.iter().copied().collect();
            ids.sort();
            chunk.module_ids = ids;
            let idx = manifest.chunks.len();
            for &mid in &chunk.module_ids {
                manifest.module_index.insert(mid, idx);
            }
            manifest.chunks.push(chunk);
        }

        // Per-entry chunks
        for (i, &ep) in entry_points.iter().enumerate() {
            let name = self
                .modules
                .get(&ep)
                .map(|m| {
                    m.path
                        .rsplit('/')
                        .next()
                        .unwrap_or(&m.path)
                        .trim_end_matches(".js")
                        .to_string()
                })
                .unwrap_or_else(|| format!("chunk{i}"));

            let mut chunk = Chunk::new(name);
            let mut ids: Vec<u64> = entry_deps[i]
                .iter()
                .filter(|id| !shared.contains(id))
                .copied()
                .collect();
            ids.sort();
            chunk.module_ids = ids;
            let idx = manifest.chunks.len();
            for &mid in &chunk.module_ids {
                manifest.module_index.insert(mid, idx);
            }
            manifest.chunks.push(chunk);
        }

        manifest
    }

    /// Compute a combined content hash for an entire chunk.
    pub fn chunk_content_hash(&self, chunk: &Chunk) -> u64 {
        let mut combined: u64 = 0;
        for &mid in &chunk.module_ids {
            if let Some(module) = self.modules.get(&mid) {
                combined = combined.wrapping_add(module.content_hash());
            }
        }
        combined
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> DependencyGraph {
        let mut g = DependencyGraph::new();
        let mut a = Module::new(1, "src/a.js", "import B");
        a.dependencies.push(2);
        a.dependencies.push(3);
        let mut b = Module::new(2, "src/b.js", "import C");
        b.dependencies.push(3);
        let c = Module::new(3, "src/c.js", "export const C = 1;");
        g.add_module(a);
        g.add_module(b);
        g.add_module(c);
        g
    }

    #[test]
    fn topological_sort_basic() {
        let g = make_graph();
        let order = g.topological_sort().unwrap();
        // A depends on B, C; B depends on C.
        // Topo order: roots first -> A(1) before B(2) before C(3) is valid
        let pos_a = order.iter().position(|x| *x == 1).unwrap();
        let pos_c = order.iter().position(|x| *x == 3).unwrap();
        assert!(pos_a < pos_c, "A should come before C");
    }

    #[test]
    fn circular_dependency_detected() {
        let mut g = DependencyGraph::new();
        let mut a = Module::new(1, "a.js", "");
        a.dependencies.push(2);
        let mut b = Module::new(2, "b.js", "");
        b.dependencies.push(1);
        g.add_module(a);
        g.add_module(b);
        let cycle = g.detect_cycles();
        assert!(cycle.is_some());
    }

    #[test]
    fn topological_sort_rejects_cycle() {
        let mut g = DependencyGraph::new();
        let mut a = Module::new(1, "a.js", "");
        a.dependencies.push(2);
        let mut b = Module::new(2, "b.js", "");
        b.dependencies.push(1);
        g.add_module(a);
        g.add_module(b);
        assert!(g.topological_sort().is_err());
    }

    #[test]
    fn transitive_deps() {
        let g = make_graph();
        let deps = g.transitive_deps(1);
        assert!(deps.contains(&2));
        assert!(deps.contains(&3));
        assert!(!deps.contains(&1));
    }

    #[test]
    fn chunk_splitting_shared_modules() {
        let g = make_graph();
        // Two entries: A(1) and B(2). Both reach C(3).
        let manifest = g.split_chunks(&[1, 2]);
        // C should be in shared chunk
        let shared = manifest.chunks.iter().find(|c| c.is_shared);
        assert!(shared.is_some());
        assert!(shared.unwrap().module_ids.contains(&3));
    }

    #[test]
    fn chunk_splitting_entry_exclusive() {
        let g = make_graph();
        let manifest = g.split_chunks(&[1, 2]);
        // Module 1 is only reachable from entry 1
        let shared = manifest.chunks.iter().find(|c| c.is_shared).unwrap();
        assert!(!shared.module_ids.contains(&1));
    }

    #[test]
    fn content_hash_deterministic() {
        let m = Module::new(1, "x.js", "console.log(42)");
        assert_eq!(m.content_hash(), m.content_hash());
    }

    #[test]
    fn content_hash_varies() {
        let m1 = Module::new(1, "x.js", "aaa");
        let m2 = Module::new(1, "x.js", "bbb");
        assert_ne!(m1.content_hash(), m2.content_hash());
    }

    #[test]
    fn fingerprinted_chunk_name() {
        let c = Chunk::new("main");
        let name = c.fingerprinted_name(0xdeadbeef);
        assert!(name.starts_with("main."));
        assert!(name.ends_with(".js"));
        assert!(name.contains("deadbeef"));
    }

    #[test]
    fn single_entry_no_shared() {
        let g = make_graph();
        let manifest = g.split_chunks(&[1]);
        // With only one entry, nothing is shared
        assert!(manifest.chunks.iter().all(|c| !c.is_shared));
    }

    #[test]
    fn no_cycles_in_dag() {
        let g = make_graph();
        assert!(g.detect_cycles().is_none());
    }

    #[test]
    fn empty_graph() {
        let g = DependencyGraph::new();
        let order = g.topological_sort().unwrap();
        assert!(order.is_empty());
    }

    #[test]
    fn manifest_display() {
        let g = make_graph();
        let manifest = g.split_chunks(&[1]);
        let s = format!("{manifest}");
        assert!(s.contains("chunk"));
    }
}
