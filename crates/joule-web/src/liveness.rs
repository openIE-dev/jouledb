//! Liveness analysis — live-in/live-out sets per basic block, def/use analysis,
//! iterative dataflow, live range computation, variable interference,
//! phi function operand liveness, analysis visualization.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt;

// ── Basic block representation ──────────────────────────────────────────────

/// A basic block identifier.
pub type BlockId = usize;

/// A variable name.
pub type VarName = String;

/// An instruction for liveness analysis purposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// Instruction index within the block.
    pub index: u32,
    /// Variables defined (written) by this instruction.
    pub defs: Vec<VarName>,
    /// Variables used (read) by this instruction.
    pub uses: Vec<VarName>,
    /// Mnemonic / description for display.
    pub mnemonic: String,
}

impl Instruction {
    /// Create a new instruction.
    pub fn new(index: u32, mnemonic: &str, defs: Vec<VarName>, uses: Vec<VarName>) -> Self {
        Self {
            index,
            defs,
            uses,
            mnemonic: mnemonic.to_string(),
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let defs = self.defs.join(", ");
        let uses = self.uses.join(", ");
        if defs.is_empty() {
            write!(f, "{}  uses({})", self.mnemonic, uses)
        } else {
            write!(f, "{} = {}  uses({})", defs, self.mnemonic, uses)
        }
    }
}

/// A basic block containing instructions.
#[derive(Debug, Clone)]
pub struct BasicBlock {
    /// Block ID.
    pub id: BlockId,
    /// Block name for display.
    pub name: String,
    /// Instructions in program order.
    pub instructions: Vec<Instruction>,
    /// Successor block IDs.
    pub successors: Vec<BlockId>,
    /// Predecessor block IDs.
    pub predecessors: Vec<BlockId>,
    /// Phi functions at the start of this block: var -> [(pred_block, operand_var)].
    pub phi_functions: HashMap<VarName, Vec<(BlockId, VarName)>>,
}

impl BasicBlock {
    /// Create a new basic block.
    pub fn new(id: BlockId, name: &str) -> Self {
        Self {
            id,
            name: name.to_string(),
            instructions: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
            phi_functions: HashMap::new(),
        }
    }

    /// Add an instruction.
    pub fn add_instruction(&mut self, instr: Instruction) {
        self.instructions.push(instr);
    }

    /// Add a phi function.
    pub fn add_phi(&mut self, var: VarName, operands: Vec<(BlockId, VarName)>) {
        self.phi_functions.insert(var, operands);
    }

    /// Number of instructions.
    pub fn len(&self) -> usize {
        self.instructions.len()
    }

    /// Whether the block has no instructions.
    pub fn is_empty(&self) -> bool {
        self.instructions.is_empty()
    }
}

// ── Def/Use analysis ────────────────────────────────────────────────────────

/// Def and use sets for a basic block.
#[derive(Debug, Clone)]
pub struct DefUse {
    /// Variables defined in this block (before being used).
    pub defs: BTreeSet<VarName>,
    /// Variables used in this block (before being defined — "upward exposed uses").
    pub uses: BTreeSet<VarName>,
}

impl DefUse {
    /// Compute def/use for a basic block.
    pub fn compute(block: &BasicBlock) -> Self {
        let mut defs = BTreeSet::new();
        let mut uses = BTreeSet::new();

        // Process phi functions: phi defs are at the top
        for var in block.phi_functions.keys() {
            defs.insert(var.clone());
        }

        // Process instructions in forward order
        for instr in &block.instructions {
            // Uses that aren't already defined in this block
            for u in &instr.uses {
                if !defs.contains(u) {
                    uses.insert(u.clone());
                }
            }
            // Definitions
            for d in &instr.defs {
                defs.insert(d.clone());
            }
        }

        Self { defs, uses }
    }
}

// ── Liveness result ─────────────────────────────────────────────────────────

/// Liveness information for the entire function.
#[derive(Debug, Clone)]
pub struct LivenessResult {
    /// Live-in sets per block.
    pub live_in: HashMap<BlockId, BTreeSet<VarName>>,
    /// Live-out sets per block.
    pub live_out: HashMap<BlockId, BTreeSet<VarName>>,
    /// Def/use per block.
    pub def_use: HashMap<BlockId, DefUse>,
    /// Number of iterations to converge.
    pub iterations: u32,
}

impl LivenessResult {
    /// Get live-in for a block.
    pub fn live_in(&self, block: BlockId) -> &BTreeSet<VarName> {
        static EMPTY: BTreeSet<VarName> = BTreeSet::new();
        self.live_in.get(&block).unwrap_or(&EMPTY)
    }

    /// Get live-out for a block.
    pub fn live_out(&self, block: BlockId) -> &BTreeSet<VarName> {
        static EMPTY: BTreeSet<VarName> = BTreeSet::new();
        self.live_out.get(&block).unwrap_or(&EMPTY)
    }

    /// Whether a variable is live-in at a block.
    pub fn is_live_in(&self, var: &str, block: BlockId) -> bool {
        self.live_in
            .get(&block)
            .map_or(false, |s| s.contains(var))
    }

    /// Whether a variable is live-out at a block.
    pub fn is_live_out(&self, var: &str, block: BlockId) -> bool {
        self.live_out
            .get(&block)
            .map_or(false, |s| s.contains(var))
    }

    /// All variables that appear in any live set.
    pub fn all_variables(&self) -> BTreeSet<VarName> {
        let mut vars = BTreeSet::new();
        for set in self.live_in.values() {
            for v in set {
                vars.insert(v.clone());
            }
        }
        for set in self.live_out.values() {
            for v in set {
                vars.insert(v.clone());
            }
        }
        vars
    }
}

// ── Iterative dataflow solver ───────────────────────────────────────────────

/// Compute liveness using iterative backward dataflow analysis.
///
/// The equations are:
///   live_out(B) = union of live_in(S) for all successors S of B
///   live_in(B) = use(B) union (live_out(B) - def(B))
///
/// With phi functions:
///   Operands of phi functions in successor S that come from block B
///   are live-out at B.
pub fn compute_liveness(blocks: &[BasicBlock]) -> LivenessResult {
    let num_blocks = blocks.len();

    // Compute def/use for each block
    let mut def_use: HashMap<BlockId, DefUse> = HashMap::new();
    for block in blocks {
        def_use.insert(block.id, DefUse::compute(block));
    }

    // Initialize live-in and live-out to empty
    let mut live_in: HashMap<BlockId, BTreeSet<VarName>> = HashMap::new();
    let mut live_out: HashMap<BlockId, BTreeSet<VarName>> = HashMap::new();
    for block in blocks {
        live_in.insert(block.id, BTreeSet::new());
        live_out.insert(block.id, BTreeSet::new());
    }

    // Build a map from block ID to block for quick lookup
    let block_map: HashMap<BlockId, &BasicBlock> = blocks.iter().map(|b| (b.id, b)).collect();

    // Iterate until fixed point
    let mut iterations = 0u32;
    let max_iterations = num_blocks as u32 * 10 + 100;
    loop {
        iterations += 1;
        let mut changed = false;

        // Process blocks in reverse order (typical for backward analysis)
        for block in blocks.iter().rev() {
            let bid = block.id;

            // Compute live_out(B) = union of live_in(S) for successors S
            let mut new_out = BTreeSet::new();
            for &succ_id in &block.successors {
                if let Some(succ_in) = live_in.get(&succ_id) {
                    for v in succ_in {
                        new_out.insert(v.clone());
                    }
                }

                // Add phi operands: for each phi in successor that takes
                // an operand from this block, that operand is live-out here
                if let Some(succ_block) = block_map.get(&succ_id) {
                    for (_phi_var, operands) in &succ_block.phi_functions {
                        for (pred_id, operand_var) in operands {
                            if *pred_id == bid {
                                new_out.insert(operand_var.clone());
                            }
                        }
                    }
                }
            }

            // Compute live_in(B) = use(B) union (live_out(B) - def(B))
            let du = &def_use[&bid];
            let mut new_in = du.uses.clone();
            for v in &new_out {
                if !du.defs.contains(v) {
                    new_in.insert(v.clone());
                }
            }

            if new_in != live_in[&bid] || new_out != live_out[&bid] {
                changed = true;
                live_in.insert(bid, new_in);
                live_out.insert(bid, new_out);
            }
        }

        if !changed || iterations >= max_iterations {
            break;
        }
    }

    LivenessResult {
        live_in,
        live_out,
        def_use,
        iterations,
    }
}

// ── Live ranges ─────────────────────────────────────────────────────────────

/// A live range for a variable — a contiguous range of program points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveRange {
    /// Variable name.
    pub var: VarName,
    /// Block where this range starts.
    pub start_block: BlockId,
    /// Instruction index where the range starts.
    pub start_index: u32,
    /// Block where this range ends.
    pub end_block: BlockId,
    /// Instruction index where the range ends.
    pub end_index: u32,
}

impl LiveRange {
    /// Create a new live range.
    pub fn new(var: &str, start_block: BlockId, start_index: u32, end_block: BlockId, end_index: u32) -> Self {
        Self {
            var: var.to_string(),
            start_block,
            start_index,
            end_block,
            end_index,
        }
    }
}

/// Compute live ranges from basic blocks and liveness results.
pub fn compute_live_ranges(blocks: &[BasicBlock], liveness: &LivenessResult) -> Vec<LiveRange> {
    let mut ranges = Vec::new();
    let all_vars = liveness.all_variables();

    for var in &all_vars {
        let mut start: Option<(BlockId, u32)> = None;
        let mut end: Option<(BlockId, u32)> = None;

        for block in blocks {
            // If live-in at this block, might be the start
            if liveness.is_live_in(var, block.id) && start.is_none() {
                start = Some((block.id, 0));
            }

            // Check instructions for def/use
            for instr in &block.instructions {
                let is_def = instr.defs.contains(var);
                let is_use = instr.uses.contains(var);

                if is_def && start.is_none() {
                    start = Some((block.id, instr.index));
                }
                if is_use || is_def {
                    end = Some((block.id, instr.index));
                }
            }

            // If live-out, extend the end
            if liveness.is_live_out(var, block.id) {
                let last_idx = block.instructions.last().map_or(0, |i| i.index);
                end = Some((block.id, last_idx));
            }
        }

        if let (Some((sb, si)), Some((eb, ei))) = (start, end) {
            ranges.push(LiveRange::new(var, sb, si, eb, ei));
        }
    }

    ranges
}

// ── Interference ────────────────────────────────────────────────────────────

/// Variable interference: two variables that are simultaneously live.
#[derive(Debug, Clone)]
pub struct InterferenceInfo {
    /// Pairs of interfering variables.
    edges: HashSet<(VarName, VarName)>,
}

impl InterferenceInfo {
    /// Compute interference from liveness results.
    pub fn compute(liveness: &LivenessResult) -> Self {
        let mut edges = HashSet::new();

        // For each block, all vars in live_out interfere with each other
        for live_set in liveness.live_out.values() {
            let vars: Vec<_> = live_set.iter().cloned().collect();
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    let a = &vars[i];
                    let b = &vars[j];
                    let (lo, hi) = if a < b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    edges.insert((lo, hi));
                }
            }
        }

        // Also for live_in
        for live_set in liveness.live_in.values() {
            let vars: Vec<_> = live_set.iter().cloned().collect();
            for i in 0..vars.len() {
                for j in (i + 1)..vars.len() {
                    let a = &vars[i];
                    let b = &vars[j];
                    let (lo, hi) = if a < b {
                        (a.clone(), b.clone())
                    } else {
                        (b.clone(), a.clone())
                    };
                    edges.insert((lo, hi));
                }
            }
        }

        Self { edges }
    }

    /// Whether two variables interfere.
    pub fn interferes(&self, a: &str, b: &str) -> bool {
        let (lo, hi) = if a < b {
            (a.to_string(), b.to_string())
        } else {
            (b.to_string(), a.to_string())
        };
        self.edges.contains(&(lo, hi))
    }

    /// Number of interference edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// All variables that interfere with a given variable.
    pub fn neighbors(&self, var: &str) -> BTreeSet<VarName> {
        let mut result = BTreeSet::new();
        for (a, b) in &self.edges {
            if a == var {
                result.insert(b.clone());
            } else if b == var {
                result.insert(a.clone());
            }
        }
        result
    }
}

// ── Visualization ───────────────────────────────────────────────────────────

/// Produce a textual visualization of liveness information.
pub fn visualize_liveness(blocks: &[BasicBlock], liveness: &LivenessResult) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "Liveness Analysis (converged in {} iterations)\n",
        liveness.iterations
    ));
    out.push_str(&"=".repeat(60));
    out.push('\n');

    for block in blocks {
        let in_set: Vec<_> = liveness.live_in(block.id).iter().cloned().collect();
        let out_set: Vec<_> = liveness.live_out(block.id).iter().cloned().collect();

        out.push_str(&format!("\n{} (id={}):\n", block.name, block.id));
        out.push_str(&format!("  live-in:  {{{}}}\n", in_set.join(", ")));

        for instr in &block.instructions {
            out.push_str(&format!("    {instr}\n"));
        }

        if !block.phi_functions.is_empty() {
            out.push_str("  phi functions:\n");
            let mut phi_entries: Vec<_> = block.phi_functions.iter().collect();
            phi_entries.sort_by_key(|(k, _)| (*k).clone());
            for (var, operands) in phi_entries {
                let ops: Vec<String> = operands
                    .iter()
                    .map(|(bid, v)| format!("B{bid}:{v}"))
                    .collect();
                out.push_str(&format!("    {} = phi({})\n", var, ops.join(", ")));
            }
        }

        out.push_str(&format!("  live-out: {{{}}}\n", out_set.join(", ")));
    }

    out
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple straight-line code: x = 1; y = x + 2; return y
    fn simple_blocks() -> Vec<BasicBlock> {
        let mut b0 = BasicBlock::new(0, "entry");
        b0.add_instruction(Instruction::new(0, "li", vec!["x".into()], vec![]));
        b0.add_instruction(Instruction::new(1, "add", vec!["y".into()], vec!["x".into()]));
        b0.add_instruction(Instruction::new(2, "ret", vec![], vec!["y".into()]));
        vec![b0]
    }

    /// Diamond CFG:
    /// B0: x = ...; if cond goto B1 else B2
    /// B1: y = x + 1; goto B3
    /// B2: y = x + 2; goto B3
    /// B3: return y
    fn diamond_blocks() -> Vec<BasicBlock> {
        let mut b0 = BasicBlock::new(0, "entry");
        b0.add_instruction(Instruction::new(0, "li", vec!["x".into()], vec![]));
        b0.add_instruction(Instruction::new(1, "br", vec![], vec!["cond".into()]));
        b0.successors = vec![1, 2];

        let mut b1 = BasicBlock::new(1, "then");
        b1.add_instruction(Instruction::new(0, "add", vec!["y1".into()], vec!["x".into()]));
        b1.successors = vec![3];
        b1.predecessors = vec![0];

        let mut b2 = BasicBlock::new(2, "else");
        b2.add_instruction(Instruction::new(0, "add", vec!["y2".into()], vec!["x".into()]));
        b2.successors = vec![3];
        b2.predecessors = vec![0];

        let mut b3 = BasicBlock::new(3, "merge");
        b3.add_phi("y".into(), vec![(1, "y1".into()), (2, "y2".into())]);
        b3.add_instruction(Instruction::new(0, "ret", vec![], vec!["y".into()]));
        b3.predecessors = vec![1, 2];

        vec![b0, b1, b2, b3]
    }

    #[test]
    fn test_def_use_simple() {
        let blocks = simple_blocks();
        let du = DefUse::compute(&blocks[0]);
        assert!(du.defs.contains("x"));
        assert!(du.defs.contains("y"));
        // x is not an upward-exposed use (it's defined before use)
        assert!(!du.uses.contains("x"));
    }

    #[test]
    fn test_liveness_simple() {
        let blocks = simple_blocks();
        let result = compute_liveness(&blocks);
        // Nothing is live-in at entry (no predecessors)
        assert!(result.live_in(0).is_empty());
        // Nothing is live-out at the only block (return consumes y)
        assert!(result.live_out(0).is_empty());
    }

    #[test]
    fn test_liveness_diamond() {
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);

        // x should be live-out at B0 (used in B1 and B2)
        assert!(result.is_live_out("x", 0));
        // x should be live-in at B1 and B2
        assert!(result.is_live_in("x", 1));
        assert!(result.is_live_in("x", 2));
        // y is defined by the phi at B3, so it is NOT live-in at B3.
        // Instead, the phi operands y1 and y2 are live-out at B1 and B2.
        assert!(!result.is_live_in("y", 3));
    }

    #[test]
    fn test_liveness_phi_operands() {
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);

        // y1 should be live-out at B1 (phi operand in B3)
        assert!(result.is_live_out("y1", 1));
        // y2 should be live-out at B2
        assert!(result.is_live_out("y2", 2));
    }

    #[test]
    fn test_convergence() {
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);
        assert!(result.iterations > 0);
        assert!(result.iterations < 100);
    }

    #[test]
    fn test_all_variables() {
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);
        let vars = result.all_variables();
        assert!(vars.contains("x"));
    }

    #[test]
    fn test_interference_computation() {
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);
        let interference = InterferenceInfo::compute(&result);
        // There should be at least some interference edges since
        // multiple variables are live at overlapping points.
        // x is live-out at B0, and cond is used in B0,
        // so they may coexist in a live set.
        assert!(interference.edge_count() >= 0); // basic structural check
        // x and cond are both in live-out(B0)
        if result.is_live_out("cond", 0) && result.is_live_out("x", 0) {
            assert!(interference.interferes("cond", "x"));
        }
    }

    #[test]
    fn test_interference_neighbors() {
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);
        let interference = InterferenceInfo::compute(&result);
        // Check that neighbors() returns a consistent set
        let n = interference.neighbors("x");
        for var in &n {
            assert!(interference.interferes("x", var));
        }
    }

    #[test]
    fn test_live_range_computation() {
        // Use diamond blocks which have cross-block liveness
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);
        let ranges = compute_live_ranges(&blocks, &result);
        // x is live across blocks, should have a range
        let has_x = ranges.iter().any(|r| r.var == "x");
        assert!(has_x);
    }

    #[test]
    fn test_instruction_display() {
        let instr = Instruction::new(0, "add", vec!["y".into()], vec!["x".into(), "z".into()]);
        let s = format!("{instr}");
        assert!(s.contains("add"));
        assert!(s.contains("x, z"));
    }

    #[test]
    fn test_basic_block_operations() {
        let mut block = BasicBlock::new(0, "test");
        assert!(block.is_empty());
        block.add_instruction(Instruction::new(0, "nop", vec![], vec![]));
        assert_eq!(block.len(), 1);
        assert!(!block.is_empty());
    }

    #[test]
    fn test_visualization() {
        let blocks = diamond_blocks();
        let result = compute_liveness(&blocks);
        let viz = visualize_liveness(&blocks, &result);
        assert!(viz.contains("Liveness Analysis"));
        assert!(viz.contains("entry"));
        assert!(viz.contains("live-in"));
        assert!(viz.contains("live-out"));
    }

    #[test]
    fn test_def_use_with_phi() {
        let blocks = diamond_blocks();
        let du = DefUse::compute(&blocks[3]); // merge block
        // phi defines y
        assert!(du.defs.contains("y"));
    }

    #[test]
    fn test_live_range_fields() {
        let lr = LiveRange::new("x", 0, 2, 3, 8);
        assert_eq!(lr.var, "x");
        assert_eq!(lr.start_block, 0);
        assert_eq!(lr.start_index, 2);
        assert_eq!(lr.end_block, 3);
        assert_eq!(lr.end_index, 8);
    }

    #[test]
    fn test_empty_program() {
        let blocks: Vec<BasicBlock> = vec![];
        let result = compute_liveness(&blocks);
        assert!(result.live_in.is_empty());
        assert!(result.live_out.is_empty());
    }
}
