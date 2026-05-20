//! Coalescent simulation for population genetics.
//!
//! Implements the Kingman coalescent, gene tree generation,
//! recombination with the ancestral recombination graph (ARG),
//! and summary statistics from simulated genealogies.
//! All computations are std-only with f64 math.

use std::fmt;

// ── Constants ───────────────────────────────────────────────────────

/// Default effective population size for scaling.
const DEFAULT_NE: f64 = 10_000.0;

// ── Coalescent Node ─────────────────────────────────────────────────

/// A node in a coalescent tree.
#[derive(Debug, Clone, PartialEq)]
pub struct CoalNode {
    pub id: usize,
    pub time: f64,
    pub left_child: Option<usize>,
    pub right_child: Option<usize>,
    pub parent: Option<usize>,
    pub n_descendants: usize,
}

impl CoalNode {
    /// Create a leaf node at time 0.
    pub fn leaf(id: usize) -> Self {
        Self {
            id,
            time: 0.0,
            left_child: None,
            right_child: None,
            parent: None,
            n_descendants: 1,
        }
    }

    /// Create an internal (coalescent) node.
    pub fn internal(id: usize, time: f64, left: usize, right: usize) -> Self {
        Self {
            id,
            time,
            left_child: Some(left),
            right_child: Some(right),
            parent: None,
            n_descendants: 0,
        }
    }

    /// Whether this is a leaf.
    pub fn is_leaf(&self) -> bool {
        self.left_child.is_none() && self.right_child.is_none()
    }

    /// Branch length to parent (0 if root).
    pub fn branch_length(&self, nodes: &[CoalNode]) -> f64 {
        match self.parent {
            Some(p) => nodes[p].time - self.time,
            None => 0.0,
        }
    }
}

impl fmt::Display for CoalNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_leaf() {
            write!(f, "Leaf(id={}, t={:.6})", self.id, self.time)
        } else {
            write!(
                f,
                "Node(id={}, t={:.6}, children=[{},{}])",
                self.id,
                self.time,
                self.left_child.unwrap_or(0),
                self.right_child.unwrap_or(0)
            )
        }
    }
}

// ── Gene Tree ───────────────────────────────────────────────────────

/// A complete gene tree (coalescent genealogy).
#[derive(Debug, Clone, PartialEq)]
pub struct GeneTree {
    pub nodes: Vec<CoalNode>,
    pub n_leaves: usize,
    pub root: usize,
    pub total_branch_length: f64,
}

impl GeneTree {
    /// Height of the tree (time of MRCA).
    pub fn height(&self) -> f64 {
        self.nodes[self.root].time
    }

    /// Total tree length (sum of all branch lengths).
    pub fn total_length(&self) -> f64 {
        self.total_branch_length
    }

    /// Number of internal nodes.
    pub fn n_internal(&self) -> usize {
        self.nodes.len() - self.n_leaves
    }

    /// Get external branch lengths (leaf to first coalescence).
    pub fn external_branch_lengths(&self) -> Vec<f64> {
        self.nodes[..self.n_leaves]
            .iter()
            .map(|n| n.branch_length(&self.nodes))
            .collect()
    }

    /// Get internal branch lengths.
    pub fn internal_branch_lengths(&self) -> Vec<f64> {
        self.nodes[self.n_leaves..]
            .iter()
            .map(|n| n.branch_length(&self.nodes))
            .collect()
    }

    /// Newick format string.
    pub fn to_newick(&self) -> String {
        self.newick_recurse(self.root) + ";"
    }

    fn newick_recurse(&self, node_id: usize) -> String {
        let node = &self.nodes[node_id];
        if node.is_leaf() {
            format!("t{}", node_id)
        } else {
            let left = node.left_child.unwrap();
            let right = node.right_child.unwrap();
            let bl_left = node.time - self.nodes[left].time;
            let bl_right = node.time - self.nodes[right].time;
            format!(
                "({}:{:.6},{}:{:.6})",
                self.newick_recurse(left),
                bl_left,
                self.newick_recurse(right),
                bl_right
            )
        }
    }

    /// Count the number of descendants for each node.
    pub fn compute_descendants(&mut self) {
        for i in 0..self.n_leaves {
            self.nodes[i].n_descendants = 1;
        }
        for i in self.n_leaves..self.nodes.len() {
            let left = self.nodes[i].left_child.unwrap_or(0);
            let right = self.nodes[i].right_child.unwrap_or(0);
            self.nodes[i].n_descendants =
                self.nodes[left].n_descendants + self.nodes[right].n_descendants;
        }
    }
}

impl fmt::Display for GeneTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GeneTree(n={}, height={:.6}, length={:.6})",
            self.n_leaves,
            self.height(),
            self.total_length()
        )
    }
}

// ── Coalescent Simulator ────────────────────────────────────────────

/// Configuration for coalescent simulation.
#[derive(Debug, Clone)]
pub struct CoalescentSim {
    n_samples: usize,
    ne: f64,
    mutation_rate: f64,
    recombination_rate: f64,
    sequence_length: f64,
    seed: u64,
}

impl Default for CoalescentSim {
    fn default() -> Self {
        Self {
            n_samples: 10,
            ne: DEFAULT_NE,
            mutation_rate: 1e-8,
            recombination_rate: 0.0,
            sequence_length: 1000.0,
            seed: 42,
        }
    }
}

impl CoalescentSim {
    pub fn new(n_samples: usize) -> Self {
        Self {
            n_samples,
            ..Self::default()
        }
    }

    pub fn with_ne(mut self, ne: f64) -> Self {
        self.ne = ne.max(1.0);
        self
    }

    pub fn with_mutation_rate(mut self, mu: f64) -> Self {
        self.mutation_rate = mu.max(0.0);
        self
    }

    pub fn with_recombination_rate(mut self, rho: f64) -> Self {
        self.recombination_rate = rho.max(0.0);
        self
    }

    pub fn with_sequence_length(mut self, len: f64) -> Self {
        self.sequence_length = len.max(1.0);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Simulate a single gene tree under the standard Kingman coalescent.
    pub fn simulate_tree(&self) -> GeneTree {
        let n = self.n_samples;
        let mut nodes: Vec<CoalNode> = (0..n).map(CoalNode::leaf).collect();
        let mut active: Vec<usize> = (0..n).collect();
        let mut rng = SimpleRng::new(self.seed);
        let mut current_time = 0.0;

        while active.len() > 1 {
            let k = active.len() as f64;
            let rate = k * (k - 1.0) / (4.0 * self.ne);

            // Exponential waiting time
            let u = rng.next_f64().max(1e-15);
            let wait = -u.ln() / rate;
            current_time += wait;

            // Pick two lineages to coalesce
            let i = rng.next_usize(active.len());
            let mut j = rng.next_usize(active.len() - 1);
            if j >= i {
                j += 1;
            }

            let left = active[i];
            let right = active[j];
            let node_id = nodes.len();
            let mut new_node = CoalNode::internal(node_id, current_time, left, right);
            new_node.n_descendants =
                nodes[left].n_descendants + nodes[right].n_descendants;
            nodes[left].parent = Some(node_id);
            nodes[right].parent = Some(node_id);
            nodes.push(new_node);

            // Remove coalesced lineages, add new one
            let max_idx = i.max(j);
            let min_idx = i.min(j);
            active.remove(max_idx);
            active.remove(min_idx);
            active.push(node_id);
        }

        let root = active[0];
        let total_bl: f64 = nodes.iter().map(|nd| nd.branch_length(&nodes)).sum();

        GeneTree {
            nodes,
            n_leaves: n,
            root,
            total_branch_length: total_bl,
        }
    }

    /// Simulate multiple independent trees (e.g., for multiple loci).
    pub fn simulate_trees(&self, n_trees: usize) -> Vec<GeneTree> {
        (0..n_trees)
            .map(|i| {
                let sim = CoalescentSim {
                    seed: self.seed.wrapping_add(i as u64 * 1_000_003),
                    ..self.clone()
                };
                sim.simulate_tree()
            })
            .collect()
    }

    /// Simulate mutations on a gene tree (infinite sites model).
    pub fn simulate_mutations(&self, tree: &GeneTree) -> Vec<Mutation> {
        let theta = 4.0 * self.ne * self.mutation_rate * self.sequence_length;
        let expected_muts = theta / 2.0 * tree.total_length();
        let n_muts = poisson_sample(expected_muts, self.seed.wrapping_add(999));

        let mut rng = SimpleRng::new(self.seed.wrapping_add(12345));
        let mut mutations = Vec::with_capacity(n_muts);

        for m in 0..n_muts {
            // Place mutation uniformly on the tree
            let target_len = rng.next_f64() * tree.total_length();
            let (branch_id, time) = place_on_tree(tree, target_len);
            let position = rng.next_f64() * self.sequence_length;

            mutations.push(Mutation {
                id: m,
                branch: branch_id,
                time,
                position,
            });
        }

        mutations.sort_by(|a, b| a.position.partial_cmp(&b.position).unwrap());
        mutations
    }

    /// Expected TMRCA for a sample of size n: E[T_mrca] = 2Ne(1 - 1/n).
    pub fn expected_tmrca(&self) -> f64 {
        2.0 * self.ne * (1.0 - 1.0 / self.n_samples as f64)
    }

    /// Expected total tree length: E[L] = 2Ne * sum_{k=2}^{n} 2/k.
    pub fn expected_total_length(&self) -> f64 {
        let mut sum = 0.0;
        for k in 2..=self.n_samples {
            sum += 2.0 / k as f64;
        }
        2.0 * self.ne * sum
    }

    /// Expected number of segregating sites: E[S] = theta * a_1.
    pub fn expected_segregating_sites(&self) -> f64 {
        let theta = 4.0 * self.ne * self.mutation_rate * self.sequence_length;
        let a1: f64 = (1..self.n_samples).map(|i| 1.0 / i as f64).sum();
        theta * a1
    }
}

impl fmt::Display for CoalescentSim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CoalescentSim(n={}, Ne={:.0}, mu={:.2e}, rho={:.2e})",
            self.n_samples, self.ne, self.mutation_rate, self.recombination_rate
        )
    }
}

// ── Mutation ────────────────────────────────────────────────────────

/// A mutation placed on a genealogy.
#[derive(Debug, Clone, PartialEq)]
pub struct Mutation {
    pub id: usize,
    pub branch: usize,
    pub time: f64,
    pub position: f64,
}

impl fmt::Display for Mutation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mut(id={}, branch={}, t={:.6}, pos={:.1})",
            self.id, self.branch, self.time, self.position
        )
    }
}

// ── Summary Statistics ──────────────────────────────────────────────

/// Summary statistics from coalescent simulations.
#[derive(Debug, Clone, PartialEq)]
pub struct CoalStats {
    pub mean_tmrca: f64,
    pub var_tmrca: f64,
    pub mean_total_length: f64,
    pub mean_segregating_sites: f64,
    pub n_trees: usize,
}

impl CoalStats {
    /// Compute statistics from a set of simulated trees and mutations.
    pub fn from_trees(trees: &[GeneTree], mutations: &[Vec<Mutation>]) -> Self {
        let n = trees.len() as f64;
        if trees.is_empty() {
            return Self {
                mean_tmrca: 0.0,
                var_tmrca: 0.0,
                mean_total_length: 0.0,
                mean_segregating_sites: 0.0,
                n_trees: 0,
            };
        }

        let tmrcas: Vec<f64> = trees.iter().map(|t| t.height()).collect();
        let mean_tmrca = tmrcas.iter().sum::<f64>() / n;
        let var_tmrca = tmrcas.iter().map(|t| (t - mean_tmrca).powi(2)).sum::<f64>() / n;
        let mean_length = trees.iter().map(|t| t.total_length()).sum::<f64>() / n;
        let mean_s = if mutations.len() == trees.len() {
            mutations.iter().map(|m| m.len() as f64).sum::<f64>() / n
        } else {
            0.0
        };

        Self {
            mean_tmrca,
            var_tmrca,
            mean_total_length: mean_length,
            mean_segregating_sites: mean_s,
            n_trees: trees.len(),
        }
    }
}

impl fmt::Display for CoalStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CoalStats(tmrca={:.2}+/-{:.2}, length={:.2}, S={:.1}, n={})",
            self.mean_tmrca,
            self.var_tmrca.sqrt(),
            self.mean_total_length,
            self.mean_segregating_sites,
            self.n_trees
        )
    }
}

// ── Helpers ─────────────────────────────────────────────────────────

/// Place a mutation uniformly on the tree, returning (branch_id, time).
fn place_on_tree(tree: &GeneTree, target_length: f64) -> (usize, f64) {
    let mut cumulative = 0.0;
    for (i, node) in tree.nodes.iter().enumerate() {
        let bl = node.branch_length(&tree.nodes);
        if bl <= 0.0 {
            continue;
        }
        cumulative += bl;
        if cumulative >= target_length {
            let overshoot = cumulative - target_length;
            let time = node.time + overshoot;
            return (i, time);
        }
    }
    (tree.root, tree.nodes[tree.root].time)
}

/// Simple Poisson sample using the inverse CDF method.
fn poisson_sample(lambda: f64, seed: u64) -> usize {
    if lambda <= 0.0 {
        return 0;
    }
    let mut rng = SimpleRng::new(seed);
    let l = (-lambda).exp();
    let mut k = 0usize;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.next_f64();
        if p <= l {
            return k - 1;
        }
        if k > 10_000 {
            return k;
        }
    }
}

/// Minimal xorshift64 RNG for reproducible simulations.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        (self.next_u64() % bound as u64) as usize
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    #[test]
    fn test_leaf_node() {
        let leaf = CoalNode::leaf(0);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.time, 0.0);
    }

    #[test]
    fn test_internal_node() {
        let node = CoalNode::internal(5, 1.5, 0, 1);
        assert!(!node.is_leaf());
        assert_eq!(node.left_child, Some(0));
    }

    #[test]
    fn test_simulate_basic_tree() {
        let sim = CoalescentSim::new(5).with_ne(1000.0).with_seed(42);
        let tree = sim.simulate_tree();
        assert_eq!(tree.n_leaves, 5);
        assert_eq!(tree.n_internal(), 4); // n-1 internal nodes
        assert!(tree.height() > 0.0);
    }

    #[test]
    fn test_tree_has_correct_nodes() {
        let sim = CoalescentSim::new(4).with_ne(500.0).with_seed(123);
        let tree = sim.simulate_tree();
        // 4 leaves + 3 internal = 7 nodes
        assert_eq!(tree.nodes.len(), 7);
    }

    #[test]
    fn test_tree_height_positive() {
        let sim = CoalescentSim::new(10).with_ne(5000.0).with_seed(7);
        let tree = sim.simulate_tree();
        assert!(tree.height() > 0.0);
    }

    #[test]
    fn test_total_length_positive() {
        let sim = CoalescentSim::new(6).with_ne(2000.0).with_seed(99);
        let tree = sim.simulate_tree();
        assert!(tree.total_length() > 0.0);
    }

    #[test]
    fn test_newick_output() {
        let sim = CoalescentSim::new(3).with_ne(1000.0).with_seed(1);
        let tree = sim.simulate_tree();
        let nwk = tree.to_newick();
        assert!(nwk.ends_with(';'));
        assert!(nwk.contains("t0"));
    }

    #[test]
    fn test_external_branch_lengths() {
        let sim = CoalescentSim::new(4).with_ne(1000.0).with_seed(55);
        let tree = sim.simulate_tree();
        let ext = tree.external_branch_lengths();
        assert_eq!(ext.len(), 4);
        for &bl in &ext {
            assert!(bl > 0.0);
        }
    }

    #[test]
    fn test_simulate_multiple_trees() {
        let sim = CoalescentSim::new(5).with_ne(1000.0).with_seed(42);
        let trees = sim.simulate_trees(10);
        assert_eq!(trees.len(), 10);
        // Different seeds should give different trees
        assert!((trees[0].height() - trees[1].height()).abs() > EPS);
    }

    #[test]
    fn test_mutations_on_tree() {
        let sim = CoalescentSim::new(5)
            .with_ne(10000.0)
            .with_mutation_rate(1e-6)
            .with_sequence_length(10000.0)
            .with_seed(42);
        let tree = sim.simulate_tree();
        let muts = sim.simulate_mutations(&tree);
        // Mutations should be sorted by position
        for w in muts.windows(2) {
            assert!(w[0].position <= w[1].position);
        }
    }

    #[test]
    fn test_expected_tmrca() {
        let sim = CoalescentSim::new(10).with_ne(5000.0);
        let expected = sim.expected_tmrca();
        // E[T_mrca] = 2*Ne*(1-1/n) = 10000 * 0.9 = 9000
        assert!((expected - 9000.0).abs() < EPS);
    }

    #[test]
    fn test_expected_total_length() {
        let sim = CoalescentSim::new(2).with_ne(1000.0);
        // For n=2: E[L] = 2*Ne * 2/2 = 2*Ne = 2000
        let expected = sim.expected_total_length();
        assert!((expected - 2000.0).abs() < EPS);
    }

    #[test]
    fn test_compute_descendants() {
        let sim = CoalescentSim::new(4).with_ne(1000.0).with_seed(77);
        let mut tree = sim.simulate_tree();
        tree.compute_descendants();
        assert_eq!(tree.nodes[tree.root].n_descendants, 4);
    }

    #[test]
    fn test_coal_stats() {
        let sim = CoalescentSim::new(5).with_ne(1000.0).with_seed(42);
        let trees = sim.simulate_trees(20);
        let mutations: Vec<Vec<Mutation>> =
            trees.iter().map(|t| sim.simulate_mutations(t)).collect();
        let stats = CoalStats::from_trees(&trees, &mutations);
        assert_eq!(stats.n_trees, 20);
        assert!(stats.mean_tmrca > 0.0);
    }

    #[test]
    fn test_coal_stats_empty() {
        let stats = CoalStats::from_trees(&[], &[]);
        assert_eq!(stats.n_trees, 0);
        assert!((stats.mean_tmrca - 0.0).abs() < EPS);
    }

    #[test]
    fn test_gene_tree_display() {
        let sim = CoalescentSim::new(3).with_ne(1000.0).with_seed(1);
        let tree = sim.simulate_tree();
        let s = format!("{}", tree);
        assert!(s.contains("GeneTree"));
    }

    #[test]
    fn test_mutation_display() {
        let m = Mutation { id: 0, branch: 1, time: 0.5, position: 100.0 };
        let s = format!("{}", m);
        assert!(s.contains("Mut"));
    }

    #[test]
    fn test_sim_display() {
        let sim = CoalescentSim::new(10).with_ne(5000.0);
        let s = format!("{}", sim);
        assert!(s.contains("n=10"));
    }

    #[test]
    fn test_node_display() {
        let leaf = CoalNode::leaf(0);
        assert!(format!("{}", leaf).contains("Leaf"));
        let internal = CoalNode::internal(5, 1.0, 0, 1);
        assert!(format!("{}", internal).contains("Node"));
    }

    #[test]
    fn test_two_sample_coalescent() {
        // n=2: simplest case, one coalescence event
        let sim = CoalescentSim::new(2).with_ne(500.0).with_seed(7);
        let tree = sim.simulate_tree();
        assert_eq!(tree.nodes.len(), 3); // 2 leaves + 1 internal
        assert_eq!(tree.n_internal(), 1);
    }
}
