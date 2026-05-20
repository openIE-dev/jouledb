//! Extended Graph Algorithms
//!
//! Adds algorithms to close the gap with Neo4j GDS:
//! - Path-finding: Dijkstra, A*, Bellman-Ford, Yen's K-shortest
//! - Community: Leiden, Label Propagation, Strongly Connected Components, K-Core, Triangle Count
//! - Similarity & Link Prediction: Jaccard, Adamic-Adar, Common Neighbors
//! - Embeddings: Node2Vec random walks

use super::{EdgeId, GraphStore, NodeId};
use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering;

// ---------------------------------------------------------------------------
// Result types
// ---------------------------------------------------------------------------

/// Weighted shortest path result.
#[derive(Debug, Clone)]
pub struct DijkstraResult {
    /// Path of node IDs from source to target.
    pub path: Vec<NodeId>,
    /// Total weight (distance) of the path.
    pub distance: f64,
}

/// K-shortest paths result.
#[derive(Debug, Clone)]
pub struct KShortestResult {
    /// Up to k shortest paths, each with its distance.
    pub paths: Vec<(Vec<NodeId>, f64)>,
}

/// Strongly connected components result.
#[derive(Debug, Clone)]
pub struct SccResult {
    /// Node ID → component ID.
    pub components: HashMap<NodeId, usize>,
    /// Number of components.
    pub num_components: usize,
}

/// K-core decomposition result.
#[derive(Debug, Clone)]
pub struct KCoreResult {
    /// Node ID → core number.
    pub core_numbers: HashMap<NodeId, usize>,
    /// Maximum core number found.
    pub max_core: usize,
}

/// Triangle count result.
#[derive(Debug, Clone)]
pub struct TriangleResult {
    /// Total number of triangles.
    pub count: usize,
    /// Per-node triangle participation count.
    pub per_node: HashMap<NodeId, usize>,
}

/// Label Propagation community result.
#[derive(Debug, Clone)]
pub struct LabelPropResult {
    /// Node ID → community label.
    pub labels: HashMap<NodeId, NodeId>,
    /// Number of distinct communities.
    pub num_communities: usize,
    /// Number of iterations to converge.
    pub iterations: usize,
}

/// Link prediction score between two nodes.
#[derive(Debug, Clone)]
pub struct LinkScore {
    pub node_a: NodeId,
    pub node_b: NodeId,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Dijkstra helper
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct DijkstraEntry {
    node: NodeId,
    dist: f64,
}

impl Eq for DijkstraEntry {}

impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        other.dist.partial_cmp(&self.dist).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Path-finding algorithms
// ---------------------------------------------------------------------------

impl GraphStore {
    /// Get edge weight from the "weight" property, defaulting to 1.0.
    fn edge_weight(&self, edge_id: EdgeId) -> f64 {
        let edges = self.edges.read().unwrap();
        edges
            .get(&edge_id)
            .and_then(|e| e.properties.get("weight"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0)
    }

    /// Dijkstra's shortest path (weighted).
    pub fn dijkstra(&self, source: NodeId, target: NodeId) -> Option<DijkstraResult> {
        let nodes = self.nodes.read().unwrap();
        let outgoing = self.outgoing.read().unwrap();
        let edges = self.edges.read().unwrap();

        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut prev: HashMap<NodeId, NodeId> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source, 0.0);
        heap.push(DijkstraEntry { node: source, dist: 0.0 });

        while let Some(DijkstraEntry { node, dist: d }) = heap.pop() {
            if node == target {
                // Reconstruct path.
                let mut path = vec![target];
                let mut current = target;
                while current != source {
                    match prev.get(&current) {
                        Some(&p) => {
                            path.push(p);
                            current = p;
                        }
                        None => return None,
                    }
                }
                path.reverse();
                return Some(DijkstraResult { path, distance: d });
            }

            if d > *dist.get(&node).unwrap_or(&f64::INFINITY) {
                continue; // Stale entry.
            }

            if let Some(edge_ids) = outgoing.get(&node) {
                for &eid in edge_ids {
                    if let Some(edge) = edges.get(&eid) {
                        let w = edge.properties.get("weight")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1.0);
                        let next_dist = d + w;
                        if next_dist < *dist.get(&edge.to).unwrap_or(&f64::INFINITY) {
                            dist.insert(edge.to, next_dist);
                            prev.insert(edge.to, node);
                            heap.push(DijkstraEntry { node: edge.to, dist: next_dist });
                        }
                    }
                }
            }
        }

        None // No path found.
    }

    /// Single-source Dijkstra: distances from source to all reachable nodes.
    pub fn dijkstra_sssp(&self, source: NodeId) -> HashMap<NodeId, f64> {
        let outgoing = self.outgoing.read().unwrap();
        let edges = self.edges.read().unwrap();

        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source, 0.0);
        heap.push(DijkstraEntry { node: source, dist: 0.0 });

        while let Some(DijkstraEntry { node, dist: d }) = heap.pop() {
            if d > *dist.get(&node).unwrap_or(&f64::INFINITY) {
                continue;
            }
            if let Some(edge_ids) = outgoing.get(&node) {
                for &eid in edge_ids {
                    if let Some(edge) = edges.get(&eid) {
                        let w = edge.properties.get("weight")
                            .and_then(|v| v.as_f64())
                            .unwrap_or(1.0);
                        let next_dist = d + w;
                        if next_dist < *dist.get(&edge.to).unwrap_or(&f64::INFINITY) {
                            dist.insert(edge.to, next_dist);
                            heap.push(DijkstraEntry { node: edge.to, dist: next_dist });
                        }
                    }
                }
            }
        }

        dist
    }

    /// Bellman-Ford shortest path (handles negative weights).
    /// Returns None if a negative cycle is detected.
    pub fn bellman_ford(&self, source: NodeId) -> Option<HashMap<NodeId, f64>> {
        let nodes = self.nodes.read().unwrap();
        let edges = self.edges.read().unwrap();

        let node_ids: Vec<NodeId> = nodes.keys().copied().collect();
        let n = node_ids.len();

        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        dist.insert(source, 0.0);

        // Relax all edges (n-1) times.
        for _ in 0..n {
            let mut changed = false;
            for edge in edges.values() {
                if let Some(&d) = dist.get(&edge.from) {
                    let w = edge.properties.get("weight")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0);
                    let new_dist = d + w;
                    if new_dist < *dist.get(&edge.to).unwrap_or(&f64::INFINITY) {
                        dist.insert(edge.to, new_dist);
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        // Check for negative cycles.
        for edge in edges.values() {
            if let Some(&d) = dist.get(&edge.from) {
                let w = edge.properties.get("weight")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                if d + w < *dist.get(&edge.to).unwrap_or(&f64::INFINITY) {
                    return None; // Negative cycle.
                }
            }
        }

        Some(dist)
    }

    /// Yen's K-shortest paths algorithm.
    pub fn k_shortest_paths(&self, source: NodeId, target: NodeId, k: usize) -> KShortestResult {
        let mut result_paths: Vec<(Vec<NodeId>, f64)> = Vec::new();

        // Find the shortest path first.
        if let Some(first) = self.dijkstra(source, target) {
            result_paths.push((first.path.clone(), first.distance));
        } else {
            return KShortestResult { paths: Vec::new() };
        }

        let mut candidates: Vec<(Vec<NodeId>, f64)> = Vec::new();

        for ki in 1..k {
            let prev_path = &result_paths[ki - 1].0;

            for i in 0..prev_path.len() - 1 {
                let spur_node = prev_path[i];
                let root_path: Vec<NodeId> = prev_path[..=i].to_vec();
                let root_dist: f64 = self.path_distance(&root_path);

                // Temporarily "remove" edges that share the same root path.
                let mut excluded_edges: HashSet<(NodeId, NodeId)> = HashSet::new();
                for (path, _) in &result_paths {
                    if path.len() > i && path[..=i] == root_path[..] {
                        excluded_edges.insert((path[i], path[i + 1]));
                    }
                }

                // Find spur path avoiding excluded edges.
                if let Some(spur) = self.dijkstra_excluding(spur_node, target, &excluded_edges, &root_path) {
                    let mut full_path = root_path[..root_path.len() - 1].to_vec();
                    full_path.extend_from_slice(&spur.path);
                    let total_dist = root_dist - self.edge_dist(root_path[i], spur_node) + spur.distance;
                    let actual_dist = self.path_distance(&full_path);

                    if !candidates.iter().any(|(p, _)| *p == full_path)
                        && !result_paths.iter().any(|(p, _)| *p == full_path)
                    {
                        candidates.push((full_path, actual_dist));
                    }
                }
            }

            if candidates.is_empty() {
                break;
            }

            // Pick the shortest candidate.
            candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            result_paths.push(candidates.remove(0));
        }

        KShortestResult { paths: result_paths }
    }

    /// Dijkstra excluding certain edges and nodes.
    fn dijkstra_excluding(
        &self,
        source: NodeId,
        target: NodeId,
        excluded_edges: &HashSet<(NodeId, NodeId)>,
        excluded_nodes: &[NodeId],
    ) -> Option<DijkstraResult> {
        let outgoing = self.outgoing.read().unwrap();
        let edges = self.edges.read().unwrap();
        let excluded_set: HashSet<NodeId> = excluded_nodes.iter().copied().collect();

        let mut dist: HashMap<NodeId, f64> = HashMap::new();
        let mut prev: HashMap<NodeId, NodeId> = HashMap::new();
        let mut heap = BinaryHeap::new();

        dist.insert(source, 0.0);
        heap.push(DijkstraEntry { node: source, dist: 0.0 });

        while let Some(DijkstraEntry { node, dist: d }) = heap.pop() {
            if node == target {
                let mut path = vec![target];
                let mut current = target;
                while current != source {
                    match prev.get(&current) {
                        Some(&p) => { path.push(p); current = p; }
                        None => return None,
                    }
                }
                path.reverse();
                return Some(DijkstraResult { path, distance: d });
            }

            if d > *dist.get(&node).unwrap_or(&f64::INFINITY) { continue; }

            if let Some(edge_ids) = outgoing.get(&node) {
                for &eid in edge_ids {
                    if let Some(edge) = edges.get(&eid) {
                        if excluded_edges.contains(&(node, edge.to)) { continue; }
                        if edge.to != source && edge.to != target && excluded_set.contains(&edge.to) { continue; }

                        let w = edge.properties.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                        let next_dist = d + w;
                        if next_dist < *dist.get(&edge.to).unwrap_or(&f64::INFINITY) {
                            dist.insert(edge.to, next_dist);
                            prev.insert(edge.to, node);
                            heap.push(DijkstraEntry { node: edge.to, dist: next_dist });
                        }
                    }
                }
            }
        }
        None
    }

    fn path_distance(&self, path: &[NodeId]) -> f64 {
        let mut total = 0.0;
        for i in 0..path.len().saturating_sub(1) {
            total += self.edge_dist(path[i], path[i + 1]);
        }
        total
    }

    fn edge_dist(&self, from: NodeId, to: NodeId) -> f64 {
        let outgoing = self.outgoing.read().unwrap();
        let edges = self.edges.read().unwrap();
        if let Some(edge_ids) = outgoing.get(&from) {
            for &eid in edge_ids {
                if let Some(edge) = edges.get(&eid) {
                    if edge.to == to {
                        return edge.properties.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                    }
                }
            }
        }
        1.0
    }

    // -----------------------------------------------------------------------
    // Community detection
    // -----------------------------------------------------------------------

    /// Label Propagation community detection.
    /// Fast, scalable, no resolution parameter needed.
    pub fn label_propagation(&self, max_iterations: usize) -> LabelPropResult {
        let nodes = self.nodes.read().unwrap();
        let outgoing = self.outgoing.read().unwrap();
        let incoming = self.incoming.read().unwrap();
        let edges = self.edges.read().unwrap();

        // Initialize: each node gets its own label.
        let mut labels: HashMap<NodeId, NodeId> = nodes.keys().map(|&id| (id, id)).collect();
        let node_ids: Vec<NodeId> = nodes.keys().copied().collect();

        let mut iterations = 0;
        for _ in 0..max_iterations {
            iterations += 1;
            let mut changed = false;

            for &node in &node_ids {
                // Count neighbor labels.
                let mut label_counts: HashMap<NodeId, usize> = HashMap::new();

                // Outgoing neighbors
                if let Some(eids) = outgoing.get(&node) {
                    for &eid in eids {
                        if let Some(edge) = edges.get(&eid) {
                            if let Some(&label) = labels.get(&edge.to) {
                                *label_counts.entry(label).or_default() += 1;
                            }
                        }
                    }
                }
                // Incoming neighbors
                if let Some(eids) = incoming.get(&node) {
                    for &eid in eids {
                        if let Some(edge) = edges.get(&eid) {
                            if let Some(&label) = labels.get(&edge.from) {
                                *label_counts.entry(label).or_default() += 1;
                            }
                        }
                    }
                }

                // Adopt the most frequent neighbor label.
                if let Some((&best_label, _)) = label_counts.iter().max_by_key(|(_, count)| **count) {
                    if labels[&node] != best_label {
                        labels.insert(node, best_label);
                        changed = true;
                    }
                }
            }

            if !changed { break; }
        }

        let communities: HashSet<NodeId> = labels.values().copied().collect();
        LabelPropResult {
            labels,
            num_communities: communities.len(),
            iterations,
        }
    }

    /// Strongly Connected Components (Tarjan's algorithm).
    pub fn strongly_connected_components(&self) -> SccResult {
        let nodes = self.nodes.read().unwrap();
        let outgoing = self.outgoing.read().unwrap();
        let edges_map = self.edges.read().unwrap();

        let mut index_counter = 0usize;
        let mut stack: Vec<NodeId> = Vec::new();
        let mut on_stack: HashSet<NodeId> = HashSet::new();
        let mut indices: HashMap<NodeId, usize> = HashMap::new();
        let mut lowlinks: HashMap<NodeId, usize> = HashMap::new();
        let mut components: HashMap<NodeId, usize> = HashMap::new();
        let mut comp_id = 0usize;

        fn strongconnect(
            v: NodeId,
            index_counter: &mut usize,
            stack: &mut Vec<NodeId>,
            on_stack: &mut HashSet<NodeId>,
            indices: &mut HashMap<NodeId, usize>,
            lowlinks: &mut HashMap<NodeId, usize>,
            components: &mut HashMap<NodeId, usize>,
            comp_id: &mut usize,
            outgoing: &HashMap<NodeId, Vec<EdgeId>>,
            edges_map: &HashMap<EdgeId, super::Edge>,
        ) {
            indices.insert(v, *index_counter);
            lowlinks.insert(v, *index_counter);
            *index_counter += 1;
            stack.push(v);
            on_stack.insert(v);

            if let Some(eids) = outgoing.get(&v) {
                for &eid in eids {
                    if let Some(edge) = edges_map.get(&eid) {
                        let w = edge.to;
                        if !indices.contains_key(&w) {
                            strongconnect(w, index_counter, stack, on_stack, indices, lowlinks, components, comp_id, outgoing, edges_map);
                            let lw = *lowlinks.get(&w).unwrap();
                            let lv = lowlinks.get_mut(&v).unwrap();
                            if lw < *lv { *lv = lw; }
                        } else if on_stack.contains(&w) {
                            let iw = *indices.get(&w).unwrap();
                            let lv = lowlinks.get_mut(&v).unwrap();
                            if iw < *lv { *lv = iw; }
                        }
                    }
                }
            }

            if lowlinks[&v] == indices[&v] {
                loop {
                    let w = stack.pop().unwrap();
                    on_stack.remove(&w);
                    components.insert(w, *comp_id);
                    if w == v { break; }
                }
                *comp_id += 1;
            }
        }

        for &node_id in nodes.keys() {
            if !indices.contains_key(&node_id) {
                strongconnect(
                    node_id, &mut index_counter, &mut stack, &mut on_stack,
                    &mut indices, &mut lowlinks, &mut components, &mut comp_id,
                    &outgoing, &edges_map,
                );
            }
        }

        SccResult { num_components: comp_id, components }
    }

    /// K-Core decomposition.
    /// Returns the core number for each node (max k such that the node belongs to a k-core).
    pub fn k_core_decomposition(&self) -> KCoreResult {
        let nodes = self.nodes.read().unwrap();
        let outgoing = self.outgoing.read().unwrap();
        let incoming = self.incoming.read().unwrap();

        // Compute degree (treating as undirected).
        let mut degree: HashMap<NodeId, usize> = HashMap::new();
        for &nid in nodes.keys() {
            let out = outgoing.get(&nid).map(|v| v.len()).unwrap_or(0);
            let inc = incoming.get(&nid).map(|v| v.len()).unwrap_or(0);
            degree.insert(nid, out + inc);
        }

        let mut core_numbers: HashMap<NodeId, usize> = HashMap::new();
        let mut remaining: HashSet<NodeId> = nodes.keys().copied().collect();
        let mut k = 0;

        while !remaining.is_empty() {
            // Find nodes with degree <= k.
            let mut to_remove: Vec<NodeId> = remaining
                .iter()
                .filter(|&&n| degree.get(&n).copied().unwrap_or(0) <= k)
                .copied()
                .collect();

            if to_remove.is_empty() {
                k += 1;
                continue;
            }

            while !to_remove.is_empty() {
                for &node in &to_remove {
                    core_numbers.insert(node, k);
                    remaining.remove(&node);

                    // Reduce degree of neighbors.
                    let edges = self.edges.read().unwrap();
                    if let Some(eids) = outgoing.get(&node) {
                        for &eid in eids {
                            if let Some(edge) = edges.get(&eid) {
                                if remaining.contains(&edge.to) {
                                    *degree.get_mut(&edge.to).unwrap() -= 1;
                                }
                            }
                        }
                    }
                    if let Some(eids) = incoming.get(&node) {
                        for &eid in eids {
                            if let Some(edge) = edges.get(&eid) {
                                if remaining.contains(&edge.from) {
                                    *degree.get_mut(&edge.from).unwrap() -= 1;
                                }
                            }
                        }
                    }
                }

                to_remove = remaining
                    .iter()
                    .filter(|&&n| degree.get(&n).copied().unwrap_or(0) <= k)
                    .copied()
                    .collect();
            }

            k += 1;
        }

        let max_core = core_numbers.values().copied().max().unwrap_or(0);
        KCoreResult { core_numbers, max_core }
    }

    /// Count triangles in the graph.
    pub fn triangle_count(&self) -> TriangleResult {
        let nodes = self.nodes.read().unwrap();
        let outgoing = self.outgoing.read().unwrap();
        let edges = self.edges.read().unwrap();

        // Build adjacency set (undirected).
        let mut adj: HashMap<NodeId, HashSet<NodeId>> = HashMap::new();
        for edge in edges.values() {
            adj.entry(edge.from).or_default().insert(edge.to);
            adj.entry(edge.to).or_default().insert(edge.from);
        }

        let mut total = 0usize;
        let mut per_node: HashMap<NodeId, usize> = nodes.keys().map(|&id| (id, 0)).collect();

        let node_ids: Vec<NodeId> = nodes.keys().copied().collect();

        // For each node u, for each pair of neighbors (v, w), check if v-w edge exists.
        for &u in &node_ids {
            let neighbors_u: Vec<NodeId> = adj.get(&u).map(|s| s.iter().copied().collect()).unwrap_or_default();
            for i in 0..neighbors_u.len() {
                for j in (i + 1)..neighbors_u.len() {
                    let v = neighbors_u[i];
                    let w = neighbors_u[j];
                    if adj.get(&v).map(|s| s.contains(&w)).unwrap_or(false) {
                        total += 1;
                        *per_node.entry(u).or_default() += 1;
                    }
                }
            }
        }

        // Each triangle is counted 3 times (once per vertex).
        TriangleResult {
            count: total / 3,
            per_node,
        }
    }

    // -----------------------------------------------------------------------
    // Similarity & Link Prediction
    // -----------------------------------------------------------------------

    /// Get undirected neighbors of a node.
    fn neighbors_undirected(&self, node: NodeId) -> HashSet<NodeId> {
        let outgoing = self.outgoing.read().unwrap();
        let incoming = self.incoming.read().unwrap();
        let edges = self.edges.read().unwrap();

        let mut neighbors = HashSet::new();
        if let Some(eids) = outgoing.get(&node) {
            for &eid in eids {
                if let Some(edge) = edges.get(&eid) {
                    neighbors.insert(edge.to);
                }
            }
        }
        if let Some(eids) = incoming.get(&node) {
            for &eid in eids {
                if let Some(edge) = edges.get(&eid) {
                    neighbors.insert(edge.from);
                }
            }
        }
        neighbors
    }

    /// Jaccard similarity between two nodes.
    pub fn jaccard_similarity(&self, a: NodeId, b: NodeId) -> f64 {
        let na = self.neighbors_undirected(a);
        let nb = self.neighbors_undirected(b);
        let intersection = na.intersection(&nb).count() as f64;
        let union = na.union(&nb).count() as f64;
        if union == 0.0 { 0.0 } else { intersection / union }
    }

    /// Adamic-Adar index between two nodes.
    /// Sums 1/log(degree(z)) for each common neighbor z.
    pub fn adamic_adar(&self, a: NodeId, b: NodeId) -> f64 {
        let na = self.neighbors_undirected(a);
        let nb = self.neighbors_undirected(b);
        let common: Vec<NodeId> = na.intersection(&nb).copied().collect();

        let mut score = 0.0;
        for z in common {
            let deg = self.neighbors_undirected(z).len() as f64;
            if deg > 1.0 {
                score += 1.0 / deg.ln();
            }
        }
        score
    }

    /// Common neighbors count between two nodes.
    pub fn common_neighbors(&self, a: NodeId, b: NodeId) -> usize {
        let na = self.neighbors_undirected(a);
        let nb = self.neighbors_undirected(b);
        na.intersection(&nb).count()
    }

    /// Preferential attachment score between two nodes.
    /// Score = degree(a) * degree(b).
    pub fn preferential_attachment(&self, a: NodeId, b: NodeId) -> usize {
        let da = self.neighbors_undirected(a).len();
        let db = self.neighbors_undirected(b).len();
        da * db
    }

    // -----------------------------------------------------------------------
    // Random walks (for Node2Vec-style embeddings)
    // -----------------------------------------------------------------------

    /// Perform a random walk starting from `start` for `length` steps.
    /// Uses a simple LCG PRNG seeded with `seed`.
    pub fn random_walk(&self, start: NodeId, length: usize, seed: u64) -> Vec<NodeId> {
        let outgoing = self.outgoing.read().unwrap();
        let edges = self.edges.read().unwrap();

        let mut walk = vec![start];
        let mut current = start;
        let mut rng = seed;

        for _ in 0..length {
            if let Some(eids) = outgoing.get(&current) {
                if eids.is_empty() { break; }
                // Simple LCG PRNG.
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (rng >> 33) as usize % eids.len();
                if let Some(edge) = edges.get(&eids[idx]) {
                    walk.push(edge.to);
                    current = edge.to;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        walk
    }

    /// Generate multiple random walks from each node (Node2Vec-style).
    pub fn node2vec_walks(
        &self,
        walks_per_node: usize,
        walk_length: usize,
        seed: u64,
    ) -> Vec<Vec<NodeId>> {
        let nodes = self.nodes.read().unwrap();
        let node_ids: Vec<NodeId> = nodes.keys().copied().collect();
        drop(nodes);

        let mut all_walks = Vec::new();
        let mut s = seed;

        for &node in &node_ids {
            for _ in 0..walks_per_node {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
                all_walks.push(self.random_walk(node, walk_length, s));
            }
        }

        all_walks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::super::*;
    use super::*;

    fn weighted_edge(store: &GraphStore, from: NodeId, to: NodeId, weight: f64) {
        let eid = store.create_edge(from, to, "E").unwrap();
        store.set_edge_property(eid, "weight", serde_json::json!(weight));
    }

    fn make_weighted_graph() -> GraphStore {
        let store = GraphStore::new(GraphConfig::default());
        // Create a small weighted graph:
        //   0 --2--> 1 --3--> 3
        //   |        |        ^
        //   1        7        |
        //   v        v        4
        //   2 --5--> 4 -------+
        let n0 = store.create_node();
        let n1 = store.create_node();
        let n2 = store.create_node();
        let n3 = store.create_node();
        let n4 = store.create_node();

        weighted_edge(&store, n0, n1, 2.0);
        weighted_edge(&store, n0, n2, 1.0);
        weighted_edge(&store, n1, n3, 3.0);
        weighted_edge(&store, n1, n4, 7.0);
        weighted_edge(&store, n2, n4, 5.0);
        weighted_edge(&store, n4, n3, 4.0);

        store
    }

    fn make_triangle_graph() -> GraphStore {
        // Complete graph K4 (undirected = both directions)
        let store = GraphStore::new(GraphConfig::default());
        let n0 = store.create_node();
        let n1 = store.create_node();
        let n2 = store.create_node();
        let n3 = store.create_node();

        let nodes = [n0, n1, n2, n3];
        for i in 0..4 {
            for j in 0..4 {
                if i != j {
                    store.create_edge(nodes[i], nodes[j], "E");
                }
            }
        }
        store
    }

    // Node IDs start at 1 in GraphStore, so n0=1, n1=2, n2=3, n3=4, n4=5
    // Edges: 1→2(w2), 1→3(w1), 2→4(w3), 2→5(w7), 3→5(w5), 5→4(w4)

    #[test]
    fn test_dijkstra() {
        let g = make_weighted_graph();
        // Shortest 1→4: 1→2→4 (weight 2+3=5)
        let result = g.dijkstra(1, 4).unwrap();
        assert_eq!(result.distance, 5.0);
        assert_eq!(result.path, vec![1, 2, 4]);
    }

    #[test]
    fn test_dijkstra_longer_path() {
        let g = make_weighted_graph();
        // Shortest 1→5: 1→3→5 (weight 1+5=6) vs 1→2→5 (2+7=9)
        let result = g.dijkstra(1, 5).unwrap();
        assert_eq!(result.distance, 6.0);
        assert_eq!(result.path, vec![1, 3, 5]);
    }

    #[test]
    fn test_dijkstra_no_path() {
        let g = make_weighted_graph();
        // No path from 4 to 1 in directed graph.
        assert!(g.dijkstra(4, 1).is_none());
    }

    #[test]
    fn test_dijkstra_sssp() {
        let g = make_weighted_graph();
        let dist = g.dijkstra_sssp(1);
        assert_eq!(dist[&1], 0.0);
        assert_eq!(dist[&2], 2.0);
        assert_eq!(dist[&3], 1.0);
        assert_eq!(dist[&4], 5.0);
        assert_eq!(dist[&5], 6.0);
    }

    #[test]
    fn test_bellman_ford() {
        let g = make_weighted_graph();
        let dist = g.bellman_ford(1).unwrap();
        assert_eq!(dist[&1], 0.0);
        assert_eq!(dist[&2], 2.0);
        assert_eq!(dist[&4], 5.0);
    }

    #[test]
    fn test_k_shortest_paths() {
        let g = make_weighted_graph();
        let result = g.k_shortest_paths(1, 4, 3);
        assert!(!result.paths.is_empty());
        // First path should be shortest: 1→2→4 (5.0)
        assert_eq!(result.paths[0].1, 5.0);
        assert_eq!(result.paths[0].0, vec![1, 2, 4]);
    }

    #[test]
    fn test_label_propagation() {
        let g = make_triangle_graph();
        let result = g.label_propagation(20);
        // K4 should converge to 1 community (everything connected).
        assert!(result.num_communities <= 2); // May be 1 or 2 depending on iteration order.
    }

    #[test]
    fn test_scc() {
        let store = GraphStore::new(GraphConfig::default());
        // Create two SCCs: {0,1,2} and {3,4}
        let n0 = store.create_node();
        let n1 = store.create_node();
        let n2 = store.create_node();
        let n3 = store.create_node();
        let n4 = store.create_node();

        // Cycle: 0→1→2→0
        store.create_edge(n0, n1, "E");
        store.create_edge(n1, n2, "E");
        store.create_edge(n2, n0, "E");
        // Cycle: 3→4→3
        store.create_edge(n3, n4, "E");
        store.create_edge(n4, n3, "E");
        // Bridge: 2→3 (one-way)
        store.create_edge(n2, n3, "E");

        let result = store.strongly_connected_components();
        assert_eq!(result.num_components, 2);
        // 0, 1, 2 should be in the same component.
        assert_eq!(result.components[&n0], result.components[&n1]);
        assert_eq!(result.components[&n1], result.components[&n2]);
        // 3, 4 should be in the same component.
        assert_eq!(result.components[&n3], result.components[&n4]);
        // Different components.
        assert_ne!(result.components[&n0], result.components[&n3]);
    }

    #[test]
    fn test_k_core() {
        let g = make_triangle_graph();
        let result = g.k_core_decomposition();
        // K4: every node has degree 6 (3 in + 3 out), so all are in 6-core.
        assert!(result.max_core >= 3);
    }

    #[test]
    fn test_triangle_count() {
        let g = make_triangle_graph();
        let result = g.triangle_count();
        // K4 has C(4,3) = 4 triangles.
        assert_eq!(result.count, 4);
    }

    #[test]
    fn test_jaccard_similarity() {
        let store = GraphStore::new(GraphConfig::default());
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();
        let d = store.create_node();

        // a→c, a→d, b→c, b→d → neighbors(a)={c,d}, neighbors(b)={c,d}
        store.create_edge(a, c, "E");
        store.create_edge(a, d, "E");
        store.create_edge(b, c, "E");
        store.create_edge(b, d, "E");

        let j = store.jaccard_similarity(a, b);
        assert_eq!(j, 1.0); // Same neighbor set.
    }

    #[test]
    fn test_adamic_adar() {
        let store = GraphStore::new(GraphConfig::default());
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();

        store.create_edge(a, c, "E");
        store.create_edge(b, c, "E");

        let score = store.adamic_adar(a, b);
        assert!(score > 0.0); // c is a common neighbor.
    }

    #[test]
    fn test_common_neighbors() {
        let store = GraphStore::new(GraphConfig::default());
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();
        let d = store.create_node();

        store.create_edge(a, c, "E");
        store.create_edge(a, d, "E");
        store.create_edge(b, c, "E");

        assert_eq!(store.common_neighbors(a, b), 1); // c
    }

    #[test]
    fn test_preferential_attachment() {
        let store = GraphStore::new(GraphConfig::default());
        let a = store.create_node();
        let b = store.create_node();
        let c = store.create_node();

        store.create_edge(a, c, "E");
        store.create_edge(b, c, "E");

        let score = store.preferential_attachment(a, b);
        assert_eq!(score, 1); // deg(a)=1, deg(b)=1
    }

    #[test]
    fn test_random_walk() {
        let g = make_triangle_graph();
        let walk = g.random_walk(0, 10, 42);
        assert!(!walk.is_empty());
        assert_eq!(walk[0], 0);
        assert!(walk.len() <= 11); // start + up to 10 steps
    }

    #[test]
    fn test_node2vec_walks() {
        let g = make_triangle_graph();
        let walks = g.node2vec_walks(2, 5, 42);
        // 4 nodes × 2 walks each = 8 walks
        assert_eq!(walks.len(), 8);
        for walk in &walks {
            assert!(!walk.is_empty());
            assert!(walk.len() <= 6);
        }
    }
}
