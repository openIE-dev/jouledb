//! Intermediate representation builder — three-address code (TAC), basic blocks,
//! control flow graph, SSA construction (simple), dead code elimination,
//! constant folding, IR pretty printer.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── IR values ───────────────────────────────────────────────────────────────

/// An IR value (operand).
#[derive(Debug, Clone, PartialEq)]
pub enum IrValue {
    /// Integer constant.
    IntConst(i64),
    /// Float constant.
    FloatConst(f64),
    /// Boolean constant.
    BoolConst(bool),
    /// String constant.
    StrConst(String),
    /// Temporary / virtual register.
    Temp(u32),
    /// Named variable.
    Var(String),
    /// Void / no value.
    Void,
}

impl fmt::Display for IrValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IntConst(n) => write!(f, "{n}"),
            Self::FloatConst(n) => write!(f, "{n:.6}"),
            Self::BoolConst(b) => write!(f, "{b}"),
            Self::StrConst(s) => write!(f, "\"{s}\""),
            Self::Temp(id) => write!(f, "t{id}"),
            Self::Var(name) => write!(f, "{name}"),
            Self::Void => write!(f, "void"),
        }
    }
}

// ── IR operations ───────────────────────────────────────────────────────────

/// Binary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    And,
    Or,
    Shl,
    Shr,
    BitAnd,
    BitOr,
    BitXor,
}

impl fmt::Display for BinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "+",
            Self::Sub => "-",
            Self::Mul => "*",
            Self::Div => "/",
            Self::Mod => "%",
            Self::Eq => "==",
            Self::Ne => "!=",
            Self::Lt => "<",
            Self::Gt => ">",
            Self::Le => "<=",
            Self::Ge => ">=",
            Self::And => "&&",
            Self::Or => "||",
            Self::Shl => "<<",
            Self::Shr => ">>",
            Self::BitAnd => "&",
            Self::BitOr => "|",
            Self::BitXor => "^",
        };
        write!(f, "{s}")
    }
}

/// Unary operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
    BitNot,
}

impl fmt::Display for UnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Neg => write!(f, "-"),
            Self::Not => write!(f, "!"),
            Self::BitNot => write!(f, "~"),
        }
    }
}

// ── Three-address code instruction ──────────────────────────────────────────

/// A three-address code instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    /// dest = left op right
    BinOp {
        dest: IrValue,
        op: BinOp,
        left: IrValue,
        right: IrValue,
    },
    /// dest = op operand
    UnaryOp {
        dest: IrValue,
        op: UnaryOp,
        operand: IrValue,
    },
    /// dest = src (copy / move)
    Copy {
        dest: IrValue,
        src: IrValue,
    },
    /// Unconditional jump to block.
    Jump(usize),
    /// Conditional branch: if cond goto true_block else false_block.
    Branch {
        cond: IrValue,
        true_block: usize,
        false_block: usize,
    },
    /// Return a value.
    Return(IrValue),
    /// Call: dest = func(args...)
    Call {
        dest: IrValue,
        func: String,
        args: Vec<IrValue>,
    },
    /// Phi function (SSA): dest = phi(src1, src2, ...)
    Phi {
        dest: IrValue,
        sources: Vec<(usize, IrValue)>, // (block_id, value)
    },
    /// No-op (used as placeholder after DCE).
    Nop,
    /// Label (just a marker, not a real instruction).
    Label(String),
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BinOp { dest, op, left, right } => {
                write!(f, "  {dest} = {left} {op} {right}")
            }
            Self::UnaryOp { dest, op, operand } => {
                write!(f, "  {dest} = {op}{operand}")
            }
            Self::Copy { dest, src } => write!(f, "  {dest} = {src}"),
            Self::Jump(target) => write!(f, "  jump BB{target}"),
            Self::Branch { cond, true_block, false_block } => {
                write!(f, "  branch {cond}, BB{true_block}, BB{false_block}")
            }
            Self::Return(val) => write!(f, "  return {val}"),
            Self::Call { dest, func, args } => {
                write!(f, "  {dest} = call {func}(")?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{a}")?;
                }
                write!(f, ")")
            }
            Self::Phi { dest, sources } => {
                write!(f, "  {dest} = phi(")?;
                for (i, (blk, val)) in sources.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "BB{blk}:{val}")?;
                }
                write!(f, ")")
            }
            Self::Nop => write!(f, "  nop"),
            Self::Label(name) => write!(f, "{name}:"),
        }
    }
}

impl Instruction {
    /// Returns the destination (defined value), if any.
    pub fn dest(&self) -> Option<&IrValue> {
        match self {
            Self::BinOp { dest, .. }
            | Self::UnaryOp { dest, .. }
            | Self::Copy { dest, .. }
            | Self::Call { dest, .. }
            | Self::Phi { dest, .. } => Some(dest),
            _ => None,
        }
    }

    /// Returns all used values.
    pub fn uses(&self) -> Vec<&IrValue> {
        match self {
            Self::BinOp { left, right, .. } => vec![left, right],
            Self::UnaryOp { operand, .. } => vec![operand],
            Self::Copy { src, .. } => vec![src],
            Self::Branch { cond, .. } => vec![cond],
            Self::Return(val) => vec![val],
            Self::Call { args, .. } => args.iter().collect(),
            Self::Phi { sources, .. } => sources.iter().map(|(_, v)| v).collect(),
            _ => vec![],
        }
    }
}

// ── Basic block ─────────────────────────────────────────────────────────────

/// A basic block — a straight-line sequence of instructions with a single
/// entry point and a single exit (the last instruction is a terminator).
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub label: Option<String>,
    pub instructions: Vec<Instruction>,
    /// Successor block ids.
    pub successors: Vec<usize>,
    /// Predecessor block ids.
    pub predecessors: Vec<usize>,
}

impl BasicBlock {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            label: None,
            instructions: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
        }
    }

    pub fn with_label(id: usize, label: &str) -> Self {
        Self {
            id,
            label: Some(label.into()),
            instructions: Vec::new(),
            successors: Vec::new(),
            predecessors: Vec::new(),
        }
    }

    /// Return the terminator instruction (last), if any.
    pub fn terminator(&self) -> Option<&Instruction> {
        self.instructions.last().filter(|i| {
            matches!(i, Instruction::Jump(_) | Instruction::Branch { .. } | Instruction::Return(_))
        })
    }

    /// True if the block ends with a terminator.
    pub fn is_terminated(&self) -> bool {
        self.terminator().is_some()
    }
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref label) = self.label {
            writeln!(f, "BB{} ({label}):", self.id)?;
        } else {
            writeln!(f, "BB{}:", self.id)?;
        }
        for instr in &self.instructions {
            writeln!(f, "{instr}")?;
        }
        Ok(())
    }
}

// ── Control flow graph ──────────────────────────────────────────────────────

/// A control flow graph (collection of basic blocks).
#[derive(Debug, Clone)]
pub struct Cfg {
    pub blocks: Vec<BasicBlock>,
    pub entry: usize,
}

impl Cfg {
    pub fn new() -> Self {
        let entry = BasicBlock::new(0);
        Self {
            blocks: vec![entry],
            entry: 0,
        }
    }

    /// Add a new block and return its id.
    pub fn add_block(&mut self) -> usize {
        let id = self.blocks.len();
        self.blocks.push(BasicBlock::new(id));
        id
    }

    /// Add a new labelled block.
    pub fn add_labelled_block(&mut self, label: &str) -> usize {
        let id = self.blocks.len();
        self.blocks.push(BasicBlock::with_label(id, label));
        id
    }

    /// Add an edge (and update predecessor/successor lists).
    pub fn add_edge(&mut self, from: usize, to: usize) {
        if from < self.blocks.len() && to < self.blocks.len() {
            if !self.blocks[from].successors.contains(&to) {
                self.blocks[from].successors.push(to);
            }
            if !self.blocks[to].predecessors.contains(&from) {
                self.blocks[to].predecessors.push(from);
            }
        }
    }

    /// Recompute predecessor/successor lists from terminators.
    pub fn rebuild_edges(&mut self) {
        // Clear
        for blk in &mut self.blocks {
            blk.successors.clear();
            blk.predecessors.clear();
        }
        let mut edges = Vec::new();
        for blk in &self.blocks {
            if let Some(term) = blk.terminator() {
                match term {
                    Instruction::Jump(target) => {
                        edges.push((blk.id, *target));
                    }
                    Instruction::Branch { true_block, false_block, .. } => {
                        edges.push((blk.id, *true_block));
                        edges.push((blk.id, *false_block));
                    }
                    _ => {}
                }
            }
        }
        for (from, to) in edges {
            self.add_edge(from, to);
        }
    }

    /// Pretty print the CFG.
    pub fn pretty_print(&self) -> String {
        let mut out = String::new();
        for blk in &self.blocks {
            out.push_str(&format!("{blk}"));
        }
        out
    }

    /// Get the number of blocks.
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Compute a post-order traversal of blocks.
    pub fn post_order(&self) -> Vec<usize> {
        let mut visited = HashSet::new();
        let mut order = Vec::new();
        self.post_order_dfs(self.entry, &mut visited, &mut order);
        order
    }

    fn post_order_dfs(&self, block: usize, visited: &mut HashSet<usize>, order: &mut Vec<usize>) {
        if visited.contains(&block) || block >= self.blocks.len() {
            return;
        }
        visited.insert(block);
        for &succ in &self.blocks[block].successors {
            self.post_order_dfs(succ, visited, order);
        }
        order.push(block);
    }

    /// Compute reverse post-order (used for forward analyses).
    pub fn reverse_post_order(&self) -> Vec<usize> {
        let mut rpo = self.post_order();
        rpo.reverse();
        rpo
    }

    /// Find all blocks reachable from entry.
    pub fn reachable_blocks(&self) -> HashSet<usize> {
        let mut visited = HashSet::new();
        let mut worklist = VecDeque::new();
        worklist.push_back(self.entry);
        while let Some(b) = worklist.pop_front() {
            if !visited.insert(b) {
                continue;
            }
            if b < self.blocks.len() {
                for &s in &self.blocks[b].successors {
                    worklist.push_back(s);
                }
            }
        }
        visited
    }
}

impl Default for Cfg {
    fn default() -> Self {
        Self::new()
    }
}

// ── IR Builder ──────────────────────────────────────────────────────────────

/// Builder for constructing IR (three-address code within a CFG).
pub struct IrBuilder {
    pub cfg: Cfg,
    current_block: usize,
    next_temp: u32,
}

impl IrBuilder {
    pub fn new() -> Self {
        Self {
            cfg: Cfg::new(),
            current_block: 0,
            next_temp: 0,
        }
    }

    /// Allocate a fresh temporary.
    pub fn fresh_temp(&mut self) -> IrValue {
        let t = self.next_temp;
        self.next_temp += 1;
        IrValue::Temp(t)
    }

    /// Get the current block id.
    pub fn current_block_id(&self) -> usize {
        self.current_block
    }

    /// Switch to emitting instructions into a different block.
    pub fn set_current_block(&mut self, block: usize) {
        self.current_block = block;
    }

    /// Create a new block and switch to it.
    pub fn new_block(&mut self) -> usize {
        let id = self.cfg.add_block();
        self.current_block = id;
        id
    }

    /// Create a new labelled block.
    pub fn new_labelled_block(&mut self, label: &str) -> usize {
        let id = self.cfg.add_labelled_block(label);
        self.current_block = id;
        id
    }

    /// Emit a binary operation.
    pub fn emit_binop(&mut self, op: BinOp, left: IrValue, right: IrValue) -> IrValue {
        let dest = self.fresh_temp();
        self.emit(Instruction::BinOp {
            dest: dest.clone(),
            op,
            left,
            right,
        });
        dest
    }

    /// Emit a unary operation.
    pub fn emit_unaryop(&mut self, op: UnaryOp, operand: IrValue) -> IrValue {
        let dest = self.fresh_temp();
        self.emit(Instruction::UnaryOp {
            dest: dest.clone(),
            op,
            operand,
        });
        dest
    }

    /// Emit a copy instruction.
    pub fn emit_copy(&mut self, src: IrValue) -> IrValue {
        let dest = self.fresh_temp();
        self.emit(Instruction::Copy {
            dest: dest.clone(),
            src,
        });
        dest
    }

    /// Emit a call.
    pub fn emit_call(&mut self, func: &str, args: Vec<IrValue>) -> IrValue {
        let dest = self.fresh_temp();
        self.emit(Instruction::Call {
            dest: dest.clone(),
            func: func.into(),
            args,
        });
        dest
    }

    /// Emit an unconditional jump.
    pub fn emit_jump(&mut self, target: usize) {
        self.emit(Instruction::Jump(target));
        self.cfg.add_edge(self.current_block, target);
    }

    /// Emit a conditional branch.
    pub fn emit_branch(&mut self, cond: IrValue, true_block: usize, false_block: usize) {
        self.emit(Instruction::Branch {
            cond,
            true_block,
            false_block,
        });
        self.cfg.add_edge(self.current_block, true_block);
        self.cfg.add_edge(self.current_block, false_block);
    }

    /// Emit a return.
    pub fn emit_return(&mut self, val: IrValue) {
        self.emit(Instruction::Return(val));
    }

    /// Emit a phi node.
    pub fn emit_phi(&mut self, sources: Vec<(usize, IrValue)>) -> IrValue {
        let dest = self.fresh_temp();
        self.emit(Instruction::Phi {
            dest: dest.clone(),
            sources,
        });
        dest
    }

    /// Emit a raw instruction into the current block.
    pub fn emit(&mut self, instr: Instruction) {
        if self.current_block < self.cfg.blocks.len() {
            self.cfg.blocks[self.current_block].instructions.push(instr);
        }
    }

    /// Build the final CFG.
    pub fn build(self) -> Cfg {
        self.cfg
    }
}

impl Default for IrBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Constant folding ────────────────────────────────────────────────────────

/// Perform constant folding on all instructions in the CFG.
pub fn constant_fold(cfg: &mut Cfg) {
    for blk in &mut cfg.blocks {
        for instr in &mut blk.instructions {
            *instr = fold_instruction(instr);
        }
    }
}

fn fold_instruction(instr: &Instruction) -> Instruction {
    match instr {
        Instruction::BinOp { dest, op, left, right } => {
            if let Some(folded) = fold_binop(*op, left, right) {
                Instruction::Copy {
                    dest: dest.clone(),
                    src: folded,
                }
            } else {
                instr.clone()
            }
        }
        Instruction::UnaryOp { dest, op, operand } => {
            if let Some(folded) = fold_unaryop(*op, operand) {
                Instruction::Copy {
                    dest: dest.clone(),
                    src: folded,
                }
            } else {
                instr.clone()
            }
        }
        _ => instr.clone(),
    }
}

fn fold_binop(op: BinOp, left: &IrValue, right: &IrValue) -> Option<IrValue> {
    match (left, right) {
        (IrValue::IntConst(a), IrValue::IntConst(b)) => {
            let result = match op {
                BinOp::Add => IrValue::IntConst(a.wrapping_add(*b)),
                BinOp::Sub => IrValue::IntConst(a.wrapping_sub(*b)),
                BinOp::Mul => IrValue::IntConst(a.wrapping_mul(*b)),
                BinOp::Div => {
                    if *b == 0 {
                        return None;
                    }
                    IrValue::IntConst(a.wrapping_div(*b))
                }
                BinOp::Mod => {
                    if *b == 0 {
                        return None;
                    }
                    IrValue::IntConst(a.wrapping_rem(*b))
                }
                BinOp::Eq => IrValue::BoolConst(a == b),
                BinOp::Ne => IrValue::BoolConst(a != b),
                BinOp::Lt => IrValue::BoolConst(a < b),
                BinOp::Gt => IrValue::BoolConst(a > b),
                BinOp::Le => IrValue::BoolConst(a <= b),
                BinOp::Ge => IrValue::BoolConst(a >= b),
                _ => return None,
            };
            Some(result)
        }
        (IrValue::FloatConst(a), IrValue::FloatConst(b)) => {
            let result = match op {
                BinOp::Add => IrValue::FloatConst(a + b),
                BinOp::Sub => IrValue::FloatConst(a - b),
                BinOp::Mul => IrValue::FloatConst(a * b),
                BinOp::Div => {
                    if *b == 0.0 {
                        return None;
                    }
                    IrValue::FloatConst(a / b)
                }
                _ => return None,
            };
            Some(result)
        }
        (IrValue::BoolConst(a), IrValue::BoolConst(b)) => {
            let result = match op {
                BinOp::And => IrValue::BoolConst(*a && *b),
                BinOp::Or => IrValue::BoolConst(*a || *b),
                BinOp::Eq => IrValue::BoolConst(a == b),
                BinOp::Ne => IrValue::BoolConst(a != b),
                _ => return None,
            };
            Some(result)
        }
        _ => None,
    }
}

fn fold_unaryop(op: UnaryOp, operand: &IrValue) -> Option<IrValue> {
    match (op, operand) {
        (UnaryOp::Neg, IrValue::IntConst(n)) => Some(IrValue::IntConst(-n)),
        (UnaryOp::Neg, IrValue::FloatConst(f)) => Some(IrValue::FloatConst(-f)),
        (UnaryOp::Not, IrValue::BoolConst(b)) => Some(IrValue::BoolConst(!b)),
        _ => None,
    }
}

// ── Dead code elimination ───────────────────────────────────────────────────

/// Remove instructions whose results are never used (within a single block).
pub fn eliminate_dead_code(cfg: &mut Cfg) {
    // Collect all used values across the whole CFG
    let mut used: HashSet<String> = HashSet::new();
    for blk in &cfg.blocks {
        for instr in &blk.instructions {
            for u in instr.uses() {
                used.insert(format!("{u}"));
            }
        }
    }

    // Mark instructions whose dest is never used (and that have no side effects)
    for blk in &mut cfg.blocks {
        let mut new_instrs = Vec::new();
        for instr in &blk.instructions {
            let keep = match instr {
                Instruction::BinOp { dest, .. }
                | Instruction::UnaryOp { dest, .. }
                | Instruction::Copy { dest, .. } => {
                    used.contains(&format!("{dest}"))
                }
                Instruction::Nop => false,
                _ => true, // keep jumps, branches, returns, calls, phis
            };
            if keep {
                new_instrs.push(instr.clone());
            }
        }
        blk.instructions = new_instrs;
    }
}

/// Remove unreachable blocks from the CFG.
pub fn eliminate_unreachable_blocks(cfg: &mut Cfg) {
    let reachable = cfg.reachable_blocks();
    for blk in &mut cfg.blocks {
        if !reachable.contains(&blk.id) {
            blk.instructions.clear();
            blk.successors.clear();
            blk.predecessors.clear();
        }
    }
}

// ── Simple SSA construction ─────────────────────────────────────────────────

/// Insert phi nodes at dominance frontiers for a simple CFG.
/// This is a simplified version — it inserts phi nodes at all join points
/// for variables that are defined in multiple blocks.
pub fn insert_phi_nodes(cfg: &mut Cfg) {
    // Find which variables/temps are defined in each block
    let mut defs_in_block: HashMap<usize, HashSet<String>> = HashMap::new();
    for blk in &cfg.blocks {
        let mut defs = HashSet::new();
        for instr in &blk.instructions {
            if let Some(dest) = instr.dest() {
                defs.insert(format!("{dest}"));
            }
        }
        defs_in_block.insert(blk.id, defs);
    }

    // For each block with multiple predecessors, insert phi nodes for all
    // variables defined in any predecessor.
    let block_count = cfg.blocks.len();
    for i in 0..block_count {
        let preds = cfg.blocks[i].predecessors.clone();
        if preds.len() < 2 {
            continue;
        }

        let mut need_phi: HashSet<String> = HashSet::new();
        for pred in &preds {
            if let Some(defs) = defs_in_block.get(pred) {
                for d in defs {
                    need_phi.insert(d.clone());
                }
            }
        }

        let mut phi_instrs = Vec::new();
        for var_name in &need_phi {
            let dest = if var_name.starts_with('t') {
                if let Ok(n) = var_name[1..].parse::<u32>() {
                    IrValue::Temp(n)
                } else {
                    IrValue::Var(var_name.clone())
                }
            } else {
                IrValue::Var(var_name.clone())
            };

            let sources: Vec<(usize, IrValue)> = preds
                .iter()
                .map(|p| (*p, dest.clone()))
                .collect();
            phi_instrs.push(Instruction::Phi {
                dest,
                sources,
            });
        }

        // Prepend phi nodes to the block
        let mut new_instrs = phi_instrs;
        new_instrs.extend(cfg.blocks[i].instructions.clone());
        cfg.blocks[i].instructions = new_instrs;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fresh_temp() {
        let mut builder = IrBuilder::new();
        let t0 = builder.fresh_temp();
        let t1 = builder.fresh_temp();
        assert_eq!(t0, IrValue::Temp(0));
        assert_eq!(t1, IrValue::Temp(1));
    }

    #[test]
    fn test_emit_binop() {
        let mut builder = IrBuilder::new();
        let result = builder.emit_binop(
            BinOp::Add,
            IrValue::IntConst(1),
            IrValue::IntConst(2),
        );
        assert_eq!(result, IrValue::Temp(0));
        assert_eq!(builder.cfg.blocks[0].instructions.len(), 1);
    }

    #[test]
    fn test_emit_unaryop() {
        let mut builder = IrBuilder::new();
        let result = builder.emit_unaryop(UnaryOp::Neg, IrValue::IntConst(5));
        assert_eq!(result, IrValue::Temp(0));
    }

    #[test]
    fn test_emit_copy() {
        let mut builder = IrBuilder::new();
        let result = builder.emit_copy(IrValue::IntConst(42));
        assert_eq!(result, IrValue::Temp(0));
    }

    #[test]
    fn test_emit_call() {
        let mut builder = IrBuilder::new();
        let result = builder.emit_call("printf", vec![IrValue::StrConst("hello".into())]);
        assert_eq!(result, IrValue::Temp(0));
    }

    #[test]
    fn test_emit_jump() {
        let mut builder = IrBuilder::new();
        let b1 = builder.new_block();
        builder.set_current_block(0);
        builder.emit_jump(b1);
        assert!(builder.cfg.blocks[0].successors.contains(&b1));
    }

    #[test]
    fn test_emit_branch() {
        let mut builder = IrBuilder::new();
        let b_true = builder.cfg.add_block();
        let b_false = builder.cfg.add_block();
        builder.set_current_block(0);
        builder.emit_branch(IrValue::BoolConst(true), b_true, b_false);
        assert!(builder.cfg.blocks[0].successors.contains(&b_true));
        assert!(builder.cfg.blocks[0].successors.contains(&b_false));
    }

    #[test]
    fn test_basic_block_terminator() {
        let mut blk = BasicBlock::new(0);
        assert!(!blk.is_terminated());
        blk.instructions.push(Instruction::Jump(1));
        assert!(blk.is_terminated());
    }

    #[test]
    fn test_cfg_post_order() {
        let mut builder = IrBuilder::new();
        let b1 = builder.cfg.add_block();
        let b2 = builder.cfg.add_block();
        builder.cfg.add_edge(0, b1);
        builder.cfg.add_edge(b1, b2);
        let po = builder.cfg.post_order();
        // b2 should come before b1 in post-order
        let pos_b2 = po.iter().position(|x| *x == b2).unwrap();
        let pos_b1 = po.iter().position(|x| *x == b1).unwrap();
        assert!(pos_b2 < pos_b1);
    }

    #[test]
    fn test_cfg_reverse_post_order() {
        let mut builder = IrBuilder::new();
        let b1 = builder.cfg.add_block();
        builder.cfg.add_edge(0, b1);
        let rpo = builder.cfg.reverse_post_order();
        assert_eq!(rpo[0], 0); // entry first
    }

    #[test]
    fn test_constant_fold_int() {
        let mut builder = IrBuilder::new();
        builder.emit_binop(BinOp::Add, IrValue::IntConst(3), IrValue::IntConst(4));
        let mut cfg = builder.build();
        constant_fold(&mut cfg);
        // The binop should have been folded to a copy
        assert!(matches!(
            &cfg.blocks[0].instructions[0],
            Instruction::Copy { src: IrValue::IntConst(7), .. }
        ));
    }

    #[test]
    fn test_constant_fold_float() {
        let mut builder = IrBuilder::new();
        builder.emit_binop(BinOp::Mul, IrValue::FloatConst(2.0), IrValue::FloatConst(3.0));
        let mut cfg = builder.build();
        constant_fold(&mut cfg);
        assert!(matches!(
            &cfg.blocks[0].instructions[0],
            Instruction::Copy { src: IrValue::FloatConst(v), .. } if (*v - 6.0).abs() < 1e-10
        ));
    }

    #[test]
    fn test_constant_fold_bool() {
        let mut builder = IrBuilder::new();
        builder.emit_binop(BinOp::And, IrValue::BoolConst(true), IrValue::BoolConst(false));
        let mut cfg = builder.build();
        constant_fold(&mut cfg);
        assert!(matches!(
            &cfg.blocks[0].instructions[0],
            Instruction::Copy { src: IrValue::BoolConst(false), .. }
        ));
    }

    #[test]
    fn test_constant_fold_neg() {
        let mut builder = IrBuilder::new();
        builder.emit_unaryop(UnaryOp::Neg, IrValue::IntConst(5));
        let mut cfg = builder.build();
        constant_fold(&mut cfg);
        assert!(matches!(
            &cfg.blocks[0].instructions[0],
            Instruction::Copy { src: IrValue::IntConst(-5), .. }
        ));
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut builder = IrBuilder::new();
        // This result is never used
        builder.emit_binop(BinOp::Add, IrValue::IntConst(1), IrValue::IntConst(2));
        // This one is used in the return
        let r = builder.emit_binop(BinOp::Mul, IrValue::IntConst(3), IrValue::IntConst(4));
        builder.emit_return(r);
        let mut cfg = builder.build();
        let before = cfg.blocks[0].instructions.len();
        eliminate_dead_code(&mut cfg);
        let after = cfg.blocks[0].instructions.len();
        assert!(after < before);
    }

    #[test]
    fn test_reachable_blocks() {
        let mut cfg = Cfg::new();
        let b1 = cfg.add_block();
        let _b2 = cfg.add_block(); // unreachable
        cfg.add_edge(0, b1);
        let reachable = cfg.reachable_blocks();
        assert!(reachable.contains(&0));
        assert!(reachable.contains(&b1));
        assert!(!reachable.contains(&_b2));
    }

    #[test]
    fn test_pretty_print() {
        let mut builder = IrBuilder::new();
        builder.emit_binop(BinOp::Add, IrValue::IntConst(1), IrValue::IntConst(2));
        let cfg = builder.build();
        let text = cfg.pretty_print();
        assert!(text.contains("BB0:"));
        assert!(text.contains("+"));
    }

    #[test]
    fn test_instruction_display() {
        let instr = Instruction::BinOp {
            dest: IrValue::Temp(0),
            op: BinOp::Add,
            left: IrValue::IntConst(1),
            right: IrValue::IntConst(2),
        };
        let s = format!("{instr}");
        assert!(s.contains("t0 = 1 + 2"));
    }

    #[test]
    fn test_rebuild_edges() {
        let mut builder = IrBuilder::new();
        let b1 = builder.cfg.add_block();
        builder.set_current_block(0);
        builder.emit(Instruction::Jump(b1));
        builder.cfg.rebuild_edges();
        assert!(builder.cfg.blocks[0].successors.contains(&b1));
        assert!(builder.cfg.blocks[b1].predecessors.contains(&0));
    }

    #[test]
    fn test_phi_insertion() {
        let mut cfg = Cfg::new();
        let b1 = cfg.add_block();
        let b2 = cfg.add_block();
        // b1 and b2 converge on a join block
        let join = cfg.add_block();
        cfg.add_edge(0, b1);
        cfg.add_edge(0, b2);
        cfg.add_edge(b1, join);
        cfg.add_edge(b2, join);

        // Define t0 in b1 and b2
        cfg.blocks[b1].instructions.push(Instruction::Copy {
            dest: IrValue::Temp(0),
            src: IrValue::IntConst(1),
        });
        cfg.blocks[b2].instructions.push(Instruction::Copy {
            dest: IrValue::Temp(0),
            src: IrValue::IntConst(2),
        });

        insert_phi_nodes(&mut cfg);

        // The join block should now have a phi node
        assert!(cfg.blocks[join]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Phi { .. })));
    }

    #[test]
    fn test_ir_value_display() {
        assert_eq!(format!("{}", IrValue::IntConst(42)), "42");
        assert_eq!(format!("{}", IrValue::Temp(3)), "t3");
        assert_eq!(format!("{}", IrValue::Var("x".into())), "x");
        assert_eq!(format!("{}", IrValue::Void), "void");
    }

    #[test]
    fn test_instruction_dest_and_uses() {
        let instr = Instruction::BinOp {
            dest: IrValue::Temp(0),
            op: BinOp::Add,
            left: IrValue::Temp(1),
            right: IrValue::Temp(2),
        };
        assert_eq!(instr.dest(), Some(&IrValue::Temp(0)));
        assert_eq!(instr.uses().len(), 2);
    }

    #[test]
    fn test_labelled_block() {
        let mut builder = IrBuilder::new();
        let b = builder.new_labelled_block("loop_header");
        assert_eq!(builder.cfg.blocks[b].label.as_deref(), Some("loop_header"));
    }
}
