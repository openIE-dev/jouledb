//! Dominance analysis — dominator tree construction (Lengauer-Tarjan simplified),
//! immediate dominators, dominance frontier, post-dominators, dominance checking,
//! CFG-based analysis, loop detection via back edges.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Control flow graph ──────────────────────────────────────────────────────

/// A basic block identifier.
pub type BlockId = usize;

/// A control flow graph (CFG).
#[derive(Debug, Clone)]
pub struct Cfg {
    /// Number of blocks.
    pub num_blocks: usize,
    /// Successors for each block.
    pub successors: Vec<Vec<BlockId>>,
    /// Predecessors for each block (computed from successors).
    pub predecessors: Vec<Vec<BlockId>>,
    /// Block names for display.
    pub names: Vec<String>,
    /// Entry block.
    pub entry: BlockId,
    /// Exit blocks.
    pub exits: Vec<BlockId>,
}

impl Cfg {
    /// Create a CFG with the given number of blocks.
    pub fn new(num_blocks: usize) -> Self {
        let names: Vec<String> = (0..num_blocks).map(|i| format!("B{i}")).collect();
        Self {
            num_blocks,
            successors: vec![Vec::new(); num_blocks],
            predecessors: vec![Vec::new(); num_blocks],
            names,
            entry: 0,
            exits: Vec::new(),
        }
    }

    /// Add an edge from `src` to `dst`.
    pub fn add_edge(&mut self, src: BlockId, dst: BlockId) {
        if !self.successors[src].contains(&dst) {
            self.successors[src].push(dst);
        }
        if !self.predecessors[dst].contains(&src) {
            self.predecessors[dst].push(src);
        }
    }

    /// Set the entry block.
    pub fn set_entry(&mut self, block: BlockId) {
        self.entry = block;
    }

    /// Mark a block as an exit.
    pub fn add_exit(&mut self, block: BlockId) {
        if !self.exits.contains(&block) {
            self.exits.push(block);
        }
    }

    /// Set a block's name.
    pub fn set_name(&mut self, block: BlockId, name: &str) {
        self.names[block] = name.to_string();
    }

    /// Reverse postorder traversal from the entry.
    pub fn reverse_postorder(&self) -> Vec<BlockId> {
        let mut visited = vec![false; self.num_blocks];
        let mut order = Vec::new();
        self.rpo_visit(self.entry, &mut visited, &mut order);
        order.reverse();
        order
    }

    fn rpo_visit(&self, block: BlockId, visited: &mut Vec<bool>, order: &mut Vec<BlockId>) {
        if visited[block] {
            return;
        }
        visited[block] = true;
        for &succ in &self.successors[block] {
            self.rpo_visit(succ, visited, order);
        }
        order.push(block);
    }

    /// BFS from entry, returning reachable blocks.
    pub fn reachable_from_entry(&self) -> HashSet<BlockId> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(self.entry);
        visited.insert(self.entry);
        while let Some(block) = queue.pop_front() {
            for &succ in &self.successors[block] {
                if visited.insert(succ) {
                    queue.push_back(succ);
                }
            }
        }
        visited
    }

    /// Build the reverse CFG (for post-dominator computation).
    pub fn reverse(&self) -> Cfg {
        let mut rev = Cfg::new(self.num_blocks);
        for src in 0..self.num_blocks {
            for &dst in &self.successors[src] {
                rev.add_edge(dst, src);
            }
        }
        rev.names = self.names.clone();
        // Entry becomes exit and vice versa
        if let Some(&exit) = self.exits.first() {
            rev.entry = exit;
        }
        rev.exits = vec![self.entry];
        rev
    }
}

impl fmt::Display for Cfg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CFG ({} blocks, entry={}):", self.num_blocks, self.names[self.entry])?;
        for i in 0..self.num_blocks {
            let succs: Vec<&str> = self.successors[i]
                .iter()
                .map(|s| self.names[*s].as_str())
                .collect();
            writeln!(f, "  {} -> [{}]", self.names[i], succs.join(", "))?;
        }
        Ok(())
    }
}

// ── Dominator tree ──────────────────────────────────────────────────────────

/// The dominator tree computed from a CFG.
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// Immediate dominator for each block (entry has itself).
    idom: Vec<Option<BlockId>>,
    /// Children in the dominator tree.
    children: Vec<Vec<BlockId>>,
    /// Number of blocks.
    num_blocks: usize,
    /// Entry block.
    entry: BlockId,
}

impl DominatorTree {
    /// Compute dominators using the iterative algorithm
    /// (Cooper, Harvey, Kennedy — "A Simple, Fast Dominance Algorithm").
    pub fn compute(cfg: &Cfg) -> Self {
        let rpo = cfg.reverse_postorder();
        let num_blocks = cfg.num_blocks;

        // Map block -> RPO number
        let mut rpo_number = vec![0usize; num_blocks];
        for (i, &block) in rpo.iter().enumerate() {
            rpo_number[block] = i;
        }

        let mut idom: Vec<Option<BlockId>> = vec![None; num_blocks];
        idom[cfg.entry] = Some(cfg.entry);

        let intersect = |mut b1: BlockId, mut b2: BlockId, idom: &[Option<BlockId>]| -> BlockId {
            while b1 != b2 {
                while rpo_number[b1] > rpo_number[b2] {
                    b1 = idom[b1].unwrap_or(b1);
                }
                while rpo_number[b2] > rpo_number[b1] {
                    b2 = idom[b2].unwrap_or(b2);
                }
            }
            b1
        };

        let mut changed = true;
        while changed {
            changed = false;
            for &block in &rpo {
                if block == cfg.entry {
                    continue;
                }

                // Find first processed predecessor
                let mut new_idom: Option<BlockId> = None;
                for &pred in &cfg.predecessors[block] {
                    if idom[pred].is_some() {
                        new_idom = Some(match new_idom {
                            Some(current) => intersect(current, pred, &idom),
                            None => pred,
                        });
                    }
                }

                if new_idom != idom[block] {
                    idom[block] = new_idom;
                    changed = true;
                }
            }
        }

        // Build children
        let mut children = vec![Vec::new(); num_blocks];
        for block in 0..num_blocks {
            if let Some(parent) = idom[block] {
                if parent != block {
                    children[parent].push(block);
                }
            }
        }

        Self {
            idom,
            children,
            num_blocks,
            entry: cfg.entry,
        }
    }

    /// Get the immediate dominator of a block.
    pub fn idom(&self, block: BlockId) -> Option<BlockId> {
        self.idom[block]
    }

    /// Get children of a block in the dominator tree.
    pub fn children(&self, block: BlockId) -> &[BlockId] {
        &self.children[block]
    }

    /// Check if `a` dominates `b`.
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }
        let mut current = b;
        loop {
            match self.idom[current] {
                Some(parent) if parent != current => {
                    if parent == a {
                        return true;
                    }
                    current = parent;
                }
                _ => return false,
            }
        }
    }

    /// Check if `a` strictly dominates `b` (a dom b and a != b).
    pub fn strictly_dominates(&self, a: BlockId, b: BlockId) -> bool {
        a != b && self.dominates(a, b)
    }

    /// Compute the depth of a block in the dominator tree.
    pub fn depth(&self, block: BlockId) -> usize {
        let mut d = 0;
        let mut current = block;
        loop {
            match self.idom[current] {
                Some(parent) if parent != current => {
                    d += 1;
                    current = parent;
                }
                _ => return d,
            }
        }
    }

    /// Number of blocks in the tree.
    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    /// Print the dominator tree.
    pub fn display_tree(&self, names: &[String]) -> String {
        let mut out = String::new();
        self.display_subtree(self.entry, 0, names, &mut out);
        out
    }

    fn display_subtree(&self, block: BlockId, indent: usize, names: &[String], out: &mut String) {
        for _ in 0..indent {
            out.push_str("  ");
        }
        out.push_str(&names[block]);
        out.push('\n');
        for &child in &self.children[block] {
            self.display_subtree(child, indent + 1, names, out);
        }
    }
}

// ── Dominance frontier ──────────────────────────────────────────────────────

/// Dominance frontier: the set of blocks just outside the dominated region.
#[derive(Debug, Clone)]
pub struct DominanceFrontier {
    /// Frontier for each block.
    frontier: Vec<HashSet<BlockId>>,
}

impl DominanceFrontier {
    /// Compute the dominance frontier from a CFG and its dominator tree.
    pub fn compute(cfg: &Cfg, dom_tree: &DominatorTree) -> Self {
        let num_blocks = cfg.num_blocks;
        let mut frontier = vec![HashSet::new(); num_blocks];

        for block in 0..num_blocks {
            let preds = &cfg.predecessors[block];
            if preds.len() >= 2 {
                for &pred in preds {
                    let mut runner = pred;
                    // Walk up the dominator tree until we reach the immediate dominator
                    // of this join point
                    let idom_block = dom_tree.idom(block);
                    while Some(runner) != idom_block {
                        frontier[runner].insert(block);
                        match dom_tree.idom(runner) {
                            Some(parent) if parent != runner => runner = parent,
                            _ => break,
                        }
                    }
                }
            }
        }

        Self { frontier }
    }

    /// Get the dominance frontier of a block.
    pub fn frontier(&self, block: BlockId) -> &HashSet<BlockId> {
        &self.frontier[block]
    }

    /// Iterated dominance frontier of a set of blocks.
    pub fn iterated_frontier(&self, blocks: &HashSet<BlockId>) -> HashSet<BlockId> {
        let mut result = HashSet::new();
        let mut worklist: Vec<BlockId> = blocks.iter().copied().collect();
        let mut processed = HashSet::new();

        while let Some(block) = worklist.pop() {
            if !processed.insert(block) {
                continue;
            }
            for &frontier_block in &self.frontier[block] {
                if result.insert(frontier_block) {
                    worklist.push(frontier_block);
                }
            }
        }

        result
    }

    /// Whether a block is in the dominance frontier of another.
    pub fn is_in_frontier(&self, block: BlockId, of: BlockId) -> bool {
        self.frontier[of].contains(&block)
    }
}

// ── Post-dominator tree ─────────────────────────────────────────────────────

/// Post-dominator tree — computed by running dominance on the reverse CFG.
pub struct PostDominatorTree {
    inner: DominatorTree,
}

impl PostDominatorTree {
    /// Compute post-dominators from the CFG.
    pub fn compute(cfg: &Cfg) -> Self {
        let rev_cfg = cfg.reverse();
        Self {
            inner: DominatorTree::compute(&rev_cfg),
        }
    }

    /// Get the immediate post-dominator.
    pub fn ipdom(&self, block: BlockId) -> Option<BlockId> {
        self.inner.idom(block)
    }

    /// Check if `a` post-dominates `b`.
    pub fn post_dominates(&self, a: BlockId, b: BlockId) -> bool {
        self.inner.dominates(a, b)
    }
}

// ── Loop detection via back edges ───────────────────────────────────────────

/// A natural loop in the CFG.
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// Loop header block.
    pub header: BlockId,
    /// Back-edge source (the block that jumps back to the header).
    pub back_edge_source: BlockId,
    /// All blocks in the loop body (including header).
    pub body: HashSet<BlockId>,
    /// Nesting depth (0 = outermost).
    pub depth: usize,
}

impl NaturalLoop {
    /// Whether a block is in this loop.
    pub fn contains(&self, block: BlockId) -> bool {
        self.body.contains(&block)
    }

    /// Number of blocks in the loop.
    pub fn size(&self) -> usize {
        self.body.len()
    }
}

/// Detect all natural loops by finding back edges.
/// A back edge is an edge from `tail` to `head` where `head` dominates `tail`.
pub fn detect_loops(cfg: &Cfg, dom_tree: &DominatorTree) -> Vec<NaturalLoop> {
    let mut loops = Vec::new();

    // Find back edges
    for src in 0..cfg.num_blocks {
        for &dst in &cfg.successors[src] {
            if dom_tree.dominates(dst, src) {
                // Back edge: src -> dst
                // Compute the natural loop body
                let body = compute_loop_body(cfg, dst, src);
                loops.push(NaturalLoop {
                    header: dst,
                    back_edge_source: src,
                    body,
                    depth: 0,
                });
            }
        }
    }

    // Compute nesting depth
    compute_nesting_depth(&mut loops);

    loops
}

/// Compute the body of a natural loop given the header and back-edge source.
fn compute_loop_body(cfg: &Cfg, header: BlockId, back_edge_src: BlockId) -> HashSet<BlockId> {
    let mut body = HashSet::new();
    body.insert(header);

    if header == back_edge_src {
        return body;
    }

    body.insert(back_edge_src);
    let mut stack = vec![back_edge_src];

    while let Some(block) = stack.pop() {
        for &pred in &cfg.predecessors[block] {
            if body.insert(pred) {
                stack.push(pred);
            }
        }
    }

    body
}

/// Compute nesting depth for loops.
fn compute_nesting_depth(loops: &mut [NaturalLoop]) {
    let n = loops.len();
    for i in 0..n {
        let mut depth = 0;
        for j in 0..n {
            if i != j && loops[j].body.is_superset(&loops[i].body) {
                depth += 1;
            }
        }
        loops[i].depth = depth;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a diamond CFG:
    /// ```text
    ///     0 (entry)
    ///    / \
    ///   1   2
    ///    \ /
    ///     3 (exit)
    /// ```
    fn diamond_cfg() -> Cfg {
        let mut cfg = Cfg::new(4);
        cfg.add_edge(0, 1);
        cfg.add_edge(0, 2);
        cfg.add_edge(1, 3);
        cfg.add_edge(2, 3);
        cfg.set_entry(0);
        cfg.add_exit(3);
        cfg
    }

    /// Build a loop CFG:
    /// ```text
    ///   0 -> 1 -> 2
    ///        ^    |
    ///        +----+
    ///        1 -> 3 (exit)
    /// ```
    fn loop_cfg() -> Cfg {
        let mut cfg = Cfg::new(4);
        cfg.add_edge(0, 1);
        cfg.add_edge(1, 2);
        cfg.add_edge(2, 1); // back edge
        cfg.add_edge(1, 3);
        cfg.set_entry(0);
        cfg.add_exit(3);
        cfg
    }

    #[test]
    fn test_cfg_creation() {
        let cfg = diamond_cfg();
        assert_eq!(cfg.num_blocks, 4);
        assert_eq!(cfg.successors[0], vec![1, 2]);
        assert!(cfg.predecessors[3].contains(&1));
        assert!(cfg.predecessors[3].contains(&2));
    }

    #[test]
    fn test_reverse_postorder() {
        let cfg = diamond_cfg();
        let rpo = cfg.reverse_postorder();
        assert_eq!(rpo[0], 0); // entry first
        // 3 must come after both 1 and 2
        let pos3 = rpo.iter().position(|b| *b == 3).unwrap();
        let pos1 = rpo.iter().position(|b| *b == 1).unwrap();
        let pos2 = rpo.iter().position(|b| *b == 2).unwrap();
        assert!(pos3 > pos1);
        assert!(pos3 > pos2);
    }

    #[test]
    fn test_reachable() {
        let cfg = diamond_cfg();
        let reachable = cfg.reachable_from_entry();
        assert_eq!(reachable.len(), 4);
    }

    #[test]
    fn test_dominator_tree_diamond() {
        let cfg = diamond_cfg();
        let dom = DominatorTree::compute(&cfg);
        // 0 dominates everything
        assert!(dom.dominates(0, 0));
        assert!(dom.dominates(0, 1));
        assert!(dom.dominates(0, 2));
        assert!(dom.dominates(0, 3));
        // 1 does not dominate 2
        assert!(!dom.dominates(1, 2));
        // 3's idom is 0 (since both 1 and 2 lead to 3)
        assert_eq!(dom.idom(3), Some(0));
    }

    #[test]
    fn test_dominator_tree_loop() {
        let cfg = loop_cfg();
        let dom = DominatorTree::compute(&cfg);
        assert!(dom.dominates(0, 1));
        assert!(dom.dominates(1, 2));
        assert!(dom.dominates(1, 3));
        assert_eq!(dom.idom(1), Some(0));
        assert_eq!(dom.idom(2), Some(1));
    }

    #[test]
    fn test_strict_dominance() {
        let cfg = diamond_cfg();
        let dom = DominatorTree::compute(&cfg);
        assert!(dom.strictly_dominates(0, 1));
        assert!(!dom.strictly_dominates(0, 0));
    }

    #[test]
    fn test_depth() {
        let cfg = diamond_cfg();
        let dom = DominatorTree::compute(&cfg);
        assert_eq!(dom.depth(0), 0);
        assert_eq!(dom.depth(1), 1);
        assert_eq!(dom.depth(3), 1); // 3's idom is 0
    }

    #[test]
    fn test_dominance_frontier_diamond() {
        let cfg = diamond_cfg();
        let dom = DominatorTree::compute(&cfg);
        let df = DominanceFrontier::compute(&cfg, &dom);
        // Block 1's frontier should contain 3 (join point)
        assert!(df.is_in_frontier(3, 1));
        // Block 2's frontier should contain 3
        assert!(df.is_in_frontier(3, 2));
        // Entry's frontier should be empty
        assert!(df.frontier(0).is_empty());
    }

    #[test]
    fn test_iterated_dominance_frontier() {
        let cfg = diamond_cfg();
        let dom = DominatorTree::compute(&cfg);
        let df = DominanceFrontier::compute(&cfg, &dom);

        let mut blocks = HashSet::new();
        blocks.insert(1);
        blocks.insert(2);
        let idf = df.iterated_frontier(&blocks);
        assert!(idf.contains(&3));
    }

    #[test]
    fn test_post_dominator() {
        let cfg = diamond_cfg();
        let pdom = PostDominatorTree::compute(&cfg);
        // Block 3 post-dominates everything
        assert!(pdom.post_dominates(3, 0));
        assert!(pdom.post_dominates(3, 1));
        assert!(pdom.post_dominates(3, 2));
    }

    #[test]
    fn test_loop_detection() {
        let cfg = loop_cfg();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0].header, 1);
        assert_eq!(loops[0].back_edge_source, 2);
        assert!(loops[0].contains(1));
        assert!(loops[0].contains(2));
    }

    #[test]
    fn test_loop_size() {
        let cfg = loop_cfg();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops[0].size(), 2); // header + body
    }

    #[test]
    fn test_nested_loops() {
        // 0 -> 1 -> 2 -> 3 -> 1 (outer back edge)
        //           2 -> 2 (self-loop, inner)
        let mut cfg = Cfg::new(4);
        cfg.add_edge(0, 1);
        cfg.add_edge(1, 2);
        cfg.add_edge(2, 3);
        cfg.add_edge(3, 1); // outer loop back edge
        cfg.add_edge(2, 2); // self-loop
        cfg.set_entry(0);

        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert!(loops.len() >= 2);
    }

    #[test]
    fn test_no_loops() {
        let cfg = diamond_cfg();
        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert!(loops.is_empty());
    }

    #[test]
    fn test_cfg_display() {
        let cfg = diamond_cfg();
        let s = format!("{cfg}");
        assert!(s.contains("B0"));
        assert!(s.contains("B3"));
    }

    #[test]
    fn test_dominator_tree_display() {
        let cfg = diamond_cfg();
        let dom = DominatorTree::compute(&cfg);
        let tree = dom.display_tree(&cfg.names);
        assert!(tree.contains("B0"));
    }

    #[test]
    fn test_reverse_cfg() {
        let cfg = diamond_cfg();
        let rev = cfg.reverse();
        // In reverse, 3 should have edges to 1 and 2
        assert!(rev.successors[3].contains(&1));
        assert!(rev.successors[3].contains(&2));
    }
}
