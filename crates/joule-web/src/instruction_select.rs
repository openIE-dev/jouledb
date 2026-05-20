//! Instruction selection — pattern matching on IR trees, tiling algorithm,
//! instruction patterns, cost-based selection, target instruction set definition,
//! lowering rules, selected instruction stream.

use std::collections::HashMap;
use std::fmt;

// ── IR tree nodes ───────────────────────────────────────────────────────────

/// An IR tree node representing an expression to be lowered.
#[derive(Debug, Clone, PartialEq)]
pub enum IrNode {
    /// Integer constant.
    IntConst(i64),
    /// Virtual register read.
    Reg(u32),
    /// Binary operation.
    BinOp {
        op: IrBinOp,
        left: Box<IrNode>,
        right: Box<IrNode>,
    },
    /// Unary operation.
    UnaryOp {
        op: IrUnaryOp,
        operand: Box<IrNode>,
    },
    /// Load from memory (base + offset).
    Load {
        base: Box<IrNode>,
        offset: i64,
    },
    /// Store to memory.
    Store {
        base: Box<IrNode>,
        offset: i64,
        value: Box<IrNode>,
    },
    /// Function call.
    Call {
        name: String,
        args: Vec<IrNode>,
    },
    /// Conditional branch.
    CondBranch {
        cond: Box<IrNode>,
        true_label: String,
        false_label: String,
    },
    /// Unconditional jump.
    Jump(String),
    /// Label definition.
    Label(String),
    /// Return.
    Return(Option<Box<IrNode>>),
}

impl IrNode {
    /// Count total nodes in the tree.
    pub fn node_count(&self) -> usize {
        match self {
            Self::IntConst(_) | Self::Reg(_) | Self::Jump(_) | Self::Label(_) => 1,
            Self::BinOp { left, right, .. } => 1 + left.node_count() + right.node_count(),
            Self::UnaryOp { operand, .. } => 1 + operand.node_count(),
            Self::Load { base, .. } => 1 + base.node_count(),
            Self::Store { base, value, .. } => 1 + base.node_count() + value.node_count(),
            Self::Call { args, .. } => 1 + args.iter().map(|a| a.node_count()).sum::<usize>(),
            Self::CondBranch { cond, .. } => 1 + cond.node_count(),
            Self::Return(Some(v)) => 1 + v.node_count(),
            Self::Return(None) => 1,
        }
    }
}

/// Binary operations in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrBinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Xor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

impl fmt::Display for IrBinOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Mod => "mod",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::Eq => "eq",
            Self::Ne => "ne",
            Self::Lt => "lt",
            Self::Le => "le",
            Self::Gt => "gt",
            Self::Ge => "ge",
        };
        write!(f, "{s}")
    }
}

/// Unary operations in the IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrUnaryOp {
    Neg,
    Not,
    BitNot,
}

impl fmt::Display for IrUnaryOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Neg => "neg",
            Self::Not => "not",
            Self::BitNot => "bitnot",
        };
        write!(f, "{s}")
    }
}

// ── Target instruction set ──────────────────────────────────────────────────

/// An operand in a target instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum TargetOperand {
    /// Register reference.
    Register(u32),
    /// Immediate value.
    Immediate(i64),
    /// Memory reference: base register + offset.
    Memory { base: u32, offset: i64 },
    /// Label reference.
    Label(String),
}

impl fmt::Display for TargetOperand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register(r) => write!(f, "r{r}"),
            Self::Immediate(v) => write!(f, "#{v}"),
            Self::Memory { base, offset } => {
                if *offset >= 0 {
                    write!(f, "[r{base}+{offset}]")
                } else {
                    write!(f, "[r{base}{offset}]")
                }
            }
            Self::Label(l) => write!(f, "{l}"),
        }
    }
}

/// A selected target instruction.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectedInstr {
    /// Instruction mnemonic.
    pub mnemonic: String,
    /// Destination operand (if any).
    pub dest: Option<TargetOperand>,
    /// Source operands.
    pub sources: Vec<TargetOperand>,
    /// Cost (in abstract cycles).
    pub cost: u32,
    /// Comment for debugging.
    pub comment: String,
}

impl SelectedInstr {
    /// Create a new selected instruction.
    pub fn new(mnemonic: &str, dest: Option<TargetOperand>, sources: Vec<TargetOperand>, cost: u32) -> Self {
        Self {
            mnemonic: mnemonic.to_string(),
            dest,
            sources,
            cost,
            comment: String::new(),
        }
    }

    /// Add a comment.
    pub fn with_comment(mut self, comment: &str) -> Self {
        self.comment = comment.to_string();
        self
    }
}

impl fmt::Display for SelectedInstr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.mnemonic)?;
        if let Some(dest) = &self.dest {
            write!(f, " {dest}")?;
        }
        for (i, src) in self.sources.iter().enumerate() {
            if i == 0 && self.dest.is_some() {
                write!(f, ", {src}")?;
            } else if i == 0 {
                write!(f, " {src}")?;
            } else {
                write!(f, ", {src}")?;
            }
        }
        if !self.comment.is_empty() {
            write!(f, "  ; {}", self.comment)?;
        }
        Ok(())
    }
}

// ── Instruction patterns ────────────────────────────────────────────────────

/// A pattern that matches an IR subtree and produces target instructions.
#[derive(Debug, Clone)]
pub struct InstrPattern {
    /// Pattern name for debugging.
    pub name: String,
    /// Cost of this pattern (lower is better).
    pub cost: u32,
    /// The kind of IR node this pattern matches.
    pub match_kind: PatternKind,
}

/// What kind of IR node a pattern matches.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PatternKind {
    /// Match a binary op with a specific operator.
    BinOp(IrBinOp),
    /// Match binary op with immediate on the right.
    BinOpImm(IrBinOp),
    /// Match a load.
    Load,
    /// Match a store.
    Store,
    /// Match a constant.
    Const,
    /// Match a register.
    Reg,
    /// Match a call.
    Call,
    /// Match a conditional branch.
    CondBranch,
    /// Match a jump.
    Jump,
    /// Match a return.
    Return,
    /// Match a unary op.
    UnaryOp(IrUnaryOp),
    /// Fused multiply-add: a + (b * c).
    FusedMulAdd,
    /// Load with base+offset where offset is from add-immediate.
    LoadIndexed,
}

impl InstrPattern {
    /// Create a new pattern.
    pub fn new(name: &str, cost: u32, match_kind: PatternKind) -> Self {
        Self {
            name: name.to_string(),
            cost,
            match_kind,
        }
    }

    /// Try to match this pattern against an IR node.
    pub fn matches(&self, node: &IrNode) -> bool {
        match (&self.match_kind, node) {
            (PatternKind::BinOp(pop), IrNode::BinOp { op, .. }) => pop == op,
            (PatternKind::BinOpImm(pop), IrNode::BinOp { op, right, .. }) => {
                pop == op && matches!(**right, IrNode::IntConst(_))
            }
            (PatternKind::Load, IrNode::Load { .. }) => true,
            (PatternKind::Store, IrNode::Store { .. }) => true,
            (PatternKind::Const, IrNode::IntConst(_)) => true,
            (PatternKind::Reg, IrNode::Reg(_)) => true,
            (PatternKind::Call, IrNode::Call { .. }) => true,
            (PatternKind::CondBranch, IrNode::CondBranch { .. }) => true,
            (PatternKind::Jump, IrNode::Jump(_)) => true,
            (PatternKind::Return, IrNode::Return(_)) => true,
            (PatternKind::UnaryOp(puo), IrNode::UnaryOp { op, .. }) => puo == op,
            (PatternKind::FusedMulAdd, IrNode::BinOp { op: IrBinOp::Add, left: _, right }) => {
                matches!(**right, IrNode::BinOp { op: IrBinOp::Mul, .. })
            }
            (PatternKind::LoadIndexed, IrNode::Load { base, .. }) => {
                matches!(**base, IrNode::BinOp { op: IrBinOp::Add, .. })
            }
            _ => false,
        }
    }
}

// ── Target instruction set definition ───────────────────────────────────────

/// A target ISA definition with available patterns.
#[derive(Debug, Clone)]
pub struct TargetIsa {
    /// Name of the target.
    pub name: String,
    /// Available patterns, sorted by priority.
    pub patterns: Vec<InstrPattern>,
    /// Number of general-purpose registers.
    pub gp_registers: u32,
}

impl TargetIsa {
    /// Create a simple RISC-like target.
    pub fn simple_risc() -> Self {
        let patterns = vec![
            // Fused patterns (higher priority = listed first, lower cost)
            InstrPattern::new("fma", 2, PatternKind::FusedMulAdd),
            InstrPattern::new("load_indexed", 2, PatternKind::LoadIndexed),
            // Immediate variants
            InstrPattern::new("addi", 1, PatternKind::BinOpImm(IrBinOp::Add)),
            InstrPattern::new("subi", 1, PatternKind::BinOpImm(IrBinOp::Sub)),
            InstrPattern::new("muli", 2, PatternKind::BinOpImm(IrBinOp::Mul)),
            InstrPattern::new("andi", 1, PatternKind::BinOpImm(IrBinOp::And)),
            InstrPattern::new("ori", 1, PatternKind::BinOpImm(IrBinOp::Or)),
            InstrPattern::new("shli", 1, PatternKind::BinOpImm(IrBinOp::Shl)),
            InstrPattern::new("shri", 1, PatternKind::BinOpImm(IrBinOp::Shr)),
            // Register-register
            InstrPattern::new("add", 1, PatternKind::BinOp(IrBinOp::Add)),
            InstrPattern::new("sub", 1, PatternKind::BinOp(IrBinOp::Sub)),
            InstrPattern::new("mul", 3, PatternKind::BinOp(IrBinOp::Mul)),
            InstrPattern::new("div", 10, PatternKind::BinOp(IrBinOp::Div)),
            InstrPattern::new("mod", 10, PatternKind::BinOp(IrBinOp::Mod)),
            InstrPattern::new("and", 1, PatternKind::BinOp(IrBinOp::And)),
            InstrPattern::new("or", 1, PatternKind::BinOp(IrBinOp::Or)),
            InstrPattern::new("xor", 1, PatternKind::BinOp(IrBinOp::Xor)),
            InstrPattern::new("shl", 1, PatternKind::BinOp(IrBinOp::Shl)),
            InstrPattern::new("shr", 1, PatternKind::BinOp(IrBinOp::Shr)),
            InstrPattern::new("cmp_eq", 1, PatternKind::BinOp(IrBinOp::Eq)),
            InstrPattern::new("cmp_ne", 1, PatternKind::BinOp(IrBinOp::Ne)),
            InstrPattern::new("cmp_lt", 1, PatternKind::BinOp(IrBinOp::Lt)),
            InstrPattern::new("cmp_le", 1, PatternKind::BinOp(IrBinOp::Le)),
            InstrPattern::new("cmp_gt", 1, PatternKind::BinOp(IrBinOp::Gt)),
            InstrPattern::new("cmp_ge", 1, PatternKind::BinOp(IrBinOp::Ge)),
            // Unary
            InstrPattern::new("neg", 1, PatternKind::UnaryOp(IrUnaryOp::Neg)),
            InstrPattern::new("not", 1, PatternKind::UnaryOp(IrUnaryOp::Not)),
            InstrPattern::new("bitnot", 1, PatternKind::UnaryOp(IrUnaryOp::BitNot)),
            // Memory
            InstrPattern::new("load", 3, PatternKind::Load),
            InstrPattern::new("store", 3, PatternKind::Store),
            // Other
            InstrPattern::new("li", 1, PatternKind::Const),
            InstrPattern::new("mov", 0, PatternKind::Reg),
            InstrPattern::new("call", 5, PatternKind::Call),
            InstrPattern::new("br", 2, PatternKind::CondBranch),
            InstrPattern::new("jmp", 1, PatternKind::Jump),
            InstrPattern::new("ret", 1, PatternKind::Return),
        ];
        Self {
            name: "SimpleRISC".to_string(),
            patterns,
            gp_registers: 16,
        }
    }

    /// Find the best matching pattern for a node.
    pub fn best_match(&self, node: &IrNode) -> Option<&InstrPattern> {
        self.patterns
            .iter()
            .filter(|p| p.matches(node))
            .min_by_key(|p| p.cost)
    }
}

// ── Selection statistics ────────────────────────────────────────────────────

/// Statistics from instruction selection.
#[derive(Debug, Clone, Default)]
pub struct SelectionStats {
    /// Total IR nodes processed.
    pub ir_nodes: u32,
    /// Total target instructions emitted.
    pub instructions_emitted: u32,
    /// Total cost of emitted instructions.
    pub total_cost: u64,
    /// Patterns used: pattern name -> count.
    pub pattern_usage: HashMap<String, u32>,
    /// Number of nodes that had no matching pattern.
    pub unmatched: u32,
}

impl SelectionStats {
    /// Average cost per instruction.
    pub fn avg_cost(&self) -> f64 {
        if self.instructions_emitted == 0 {
            return 0.0;
        }
        self.total_cost as f64 / self.instructions_emitted as f64
    }

    /// Record a pattern use.
    fn record_pattern(&mut self, name: &str, cost: u32) {
        *self.pattern_usage.entry(name.to_string()).or_insert(0) += 1;
        self.instructions_emitted += 1;
        self.total_cost += cost as u64;
    }
}

// ── Instruction selector ────────────────────────────────────────────────────

/// The instruction selector — tiles IR trees into target instructions using
/// a greedy maximal-munch approach.
pub struct InstructionSelector {
    /// Target ISA.
    isa: TargetIsa,
    /// Emitted instructions.
    output: Vec<SelectedInstr>,
    /// Next virtual register.
    next_reg: u32,
    /// Statistics.
    stats: SelectionStats,
}

impl InstructionSelector {
    /// Create a new selector for the given target.
    pub fn new(isa: TargetIsa) -> Self {
        Self {
            isa,
            output: Vec::new(),
            next_reg: 0,
            stats: SelectionStats::default(),
        }
    }

    /// Allocate a fresh virtual register.
    fn fresh_reg(&mut self) -> u32 {
        let r = self.next_reg;
        self.next_reg += 1;
        r
    }

    /// Select instructions for a list of IR trees (statements).
    pub fn select_all(&mut self, trees: &[IrNode]) -> Vec<SelectedInstr> {
        self.output.clear();
        self.stats = SelectionStats::default();
        self.next_reg = 0;

        for tree in trees {
            self.select_tree(tree);
        }

        self.output.clone()
    }

    /// Look up the best pattern for a node, returning owned (name, cost, match_kind).
    fn lookup_pattern(&self, node: &IrNode) -> Option<(String, u32, PatternKind)> {
        self.isa.best_match(node).map(|p| (p.name.clone(), p.cost, p.match_kind.clone()))
    }

    /// Select instructions for a single IR tree.
    /// Returns the register holding the result (if any).
    fn select_tree(&mut self, node: &IrNode) -> Option<u32> {
        self.stats.ir_nodes += 1;

        match node {
            IrNode::IntConst(val) => {
                let dest = self.fresh_reg();
                self.emit("li", Some(TargetOperand::Register(dest)), vec![TargetOperand::Immediate(*val)], 1, "li");
                Some(dest)
            }

            IrNode::Reg(r) => Some(*r),

            IrNode::BinOp { op, left, right } => {
                // Check for fused multiply-add: a + (b * c)
                // FMA is checked structurally (not via best_match) because it
                // spans two IR nodes and should be preferred over separate add+mul.
                if *op == IrBinOp::Add {
                    if let IrNode::BinOp { op: IrBinOp::Mul, left: ml, right: mr } = right.as_ref() {
                        // Check whether the ISA has an fma pattern at all
                        let has_fma = self.isa.patterns.iter()
                            .any(|p| p.match_kind == PatternKind::FusedMulAdd);
                        if has_fma {
                            let fma_cost = self.isa.patterns.iter()
                                .find(|p| p.match_kind == PatternKind::FusedMulAdd)
                                .map_or(2, |p| p.cost);
                            let ra = self.select_tree(left)?;
                            let rb = self.select_tree(ml)?;
                            let rc = self.select_tree(mr)?;
                            let dest = self.fresh_reg();
                            self.emit(
                                "fma",
                                Some(TargetOperand::Register(dest)),
                                vec![
                                    TargetOperand::Register(ra),
                                    TargetOperand::Register(rb),
                                    TargetOperand::Register(rc),
                                ],
                                fma_cost,
                                "fma",
                            );
                            return Some(dest);
                        }
                    }
                }

                // Check for immediate variant
                if let IrNode::IntConst(imm) = right.as_ref() {
                    let pat_info = self.lookup_pattern(node);
                    if let Some((pat_name, pat_cost, PatternKind::BinOpImm(_))) = pat_info {
                        let lr = self.select_tree(left)?;
                        let dest = self.fresh_reg();
                        self.emit(
                            &pat_name,
                            Some(TargetOperand::Register(dest)),
                            vec![TargetOperand::Register(lr), TargetOperand::Immediate(*imm)],
                            pat_cost,
                            &pat_name,
                        );
                        return Some(dest);
                    }
                }

                // General register-register
                let pat_info = self.lookup_pattern(node);
                let lr = self.select_tree(left)?;
                let rr = self.select_tree(right)?;
                let dest = self.fresh_reg();
                let (name, cost) = pat_info
                    .map(|(n, c, _)| (n, c))
                    .unwrap_or_else(|| (format!("{op}"), 1));
                self.emit(
                    &name,
                    Some(TargetOperand::Register(dest)),
                    vec![TargetOperand::Register(lr), TargetOperand::Register(rr)],
                    cost,
                    &name,
                );
                Some(dest)
            }

            IrNode::UnaryOp { op, operand } => {
                let pat_info = self.lookup_pattern(node);
                let r = self.select_tree(operand)?;
                let dest = self.fresh_reg();
                let (name, cost) = pat_info
                    .map(|(n, c, _)| (n, c))
                    .unwrap_or_else(|| (format!("{op}"), 1));
                self.emit(
                    &name,
                    Some(TargetOperand::Register(dest)),
                    vec![TargetOperand::Register(r)],
                    cost,
                    &name,
                );
                Some(dest)
            }

            IrNode::Load { base, offset } => {
                let br = self.select_tree(base)?;
                let dest = self.fresh_reg();
                self.emit(
                    "load",
                    Some(TargetOperand::Register(dest)),
                    vec![TargetOperand::Memory { base: br, offset: *offset }],
                    3,
                    "load",
                );
                Some(dest)
            }

            IrNode::Store { base, offset, value } => {
                let br = self.select_tree(base)?;
                let vr = self.select_tree(value)?;
                self.emit(
                    "store",
                    None,
                    vec![
                        TargetOperand::Memory { base: br, offset: *offset },
                        TargetOperand::Register(vr),
                    ],
                    3,
                    "store",
                );
                None
            }

            IrNode::Call { name, args } => {
                let mut arg_regs = Vec::new();
                for arg in args {
                    if let Some(r) = self.select_tree(arg) {
                        arg_regs.push(TargetOperand::Register(r));
                    }
                }
                let dest = self.fresh_reg();
                let mut sources = vec![TargetOperand::Label(name.clone())];
                sources.extend(arg_regs);
                self.emit("call", Some(TargetOperand::Register(dest)), sources, 5, "call");
                Some(dest)
            }

            IrNode::CondBranch { cond, true_label, false_label } => {
                let cr = self.select_tree(cond)?;
                self.emit(
                    "br",
                    None,
                    vec![
                        TargetOperand::Register(cr),
                        TargetOperand::Label(true_label.clone()),
                        TargetOperand::Label(false_label.clone()),
                    ],
                    2,
                    "br",
                );
                None
            }

            IrNode::Jump(label) => {
                self.emit("jmp", None, vec![TargetOperand::Label(label.clone())], 1, "jmp");
                None
            }

            IrNode::Label(label) => {
                self.emit(&format!("{label}:"), None, vec![], 0, "label");
                None
            }

            IrNode::Return(val) => {
                if let Some(v) = val {
                    let r = self.select_tree(v)?;
                    self.emit("ret", None, vec![TargetOperand::Register(r)], 1, "ret");
                } else {
                    self.emit("ret", None, vec![], 1, "ret");
                }
                None
            }
        }
    }

    /// Emit a target instruction.
    fn emit(&mut self, mnemonic: &str, dest: Option<TargetOperand>, sources: Vec<TargetOperand>, cost: u32, pattern_name: &str) {
        self.output.push(SelectedInstr::new(mnemonic, dest, sources, cost));
        self.stats.record_pattern(pattern_name, cost);
    }

    /// Get selection statistics.
    pub fn statistics(&self) -> &SelectionStats {
        &self.stats
    }

    /// Get the emitted instruction stream.
    pub fn output(&self) -> &[SelectedInstr] {
        &self.output
    }
}

// ── Lowering rules ──────────────────────────────────────────────────────────

/// A lowering rule transforms an IR construct into a sequence of target instructions.
#[derive(Debug, Clone)]
pub struct LoweringRule {
    /// Rule name.
    pub name: String,
    /// Source pattern kind.
    pub source: PatternKind,
    /// Number of instructions this rule emits.
    pub emission_count: u32,
    /// Total cost of emitted instructions.
    pub total_cost: u32,
}

impl LoweringRule {
    /// Create a new lowering rule.
    pub fn new(name: &str, source: PatternKind, emission_count: u32, total_cost: u32) -> Self {
        Self {
            name: name.to_string(),
            source,
            emission_count,
            total_cost,
        }
    }
}

/// A set of lowering rules for a target.
#[derive(Debug, Clone)]
pub struct LoweringRuleSet {
    rules: Vec<LoweringRule>,
}

impl LoweringRuleSet {
    /// Create an empty rule set.
    pub fn new() -> Self {
        Self { rules: Vec::new() }
    }

    /// Add a rule.
    pub fn add_rule(&mut self, rule: LoweringRule) {
        self.rules.push(rule);
    }

    /// Find the best rule for a given pattern kind.
    pub fn best_rule(&self, kind: &PatternKind) -> Option<&LoweringRule> {
        self.rules
            .iter()
            .filter(|r| r.source == *kind)
            .min_by_key(|r| r.total_cost)
    }

    /// Number of rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Whether empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }
}

impl Default for LoweringRuleSet {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_add(left: IrNode, right: IrNode) -> IrNode {
        IrNode::BinOp {
            op: IrBinOp::Add,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    fn make_mul(left: IrNode, right: IrNode) -> IrNode {
        IrNode::BinOp {
            op: IrBinOp::Mul,
            left: Box::new(left),
            right: Box::new(right),
        }
    }

    #[test]
    fn test_select_constant() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let instrs = sel.select_all(&[IrNode::IntConst(42)]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].mnemonic, "li");
        assert_eq!(instrs[0].sources, vec![TargetOperand::Immediate(42)]);
    }

    #[test]
    fn test_select_add_registers() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        // Use two register operands so the immediate pattern doesn't match
        let tree = make_add(IrNode::Reg(0), IrNode::Reg(1));
        let instrs = sel.select_all(&[tree]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].mnemonic, "add");
    }

    #[test]
    fn test_select_add_immediate() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = make_add(IrNode::Reg(0), IrNode::IntConst(5));
        let instrs = sel.select_all(&[tree]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].mnemonic, "addi");
    }

    #[test]
    fn test_select_fused_multiply_add() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        // a + (b * c)
        let tree = make_add(
            IrNode::Reg(0),
            make_mul(IrNode::Reg(1), IrNode::Reg(2)),
        );
        let instrs = sel.select_all(&[tree]);
        // Should use fma instead of mul + add
        let has_fma = instrs.iter().any(|i| i.mnemonic == "fma");
        assert!(has_fma);
    }

    #[test]
    fn test_select_load() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = IrNode::Load {
            base: Box::new(IrNode::Reg(0)),
            offset: 8,
        };
        let instrs = sel.select_all(&[tree]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].mnemonic, "load");
    }

    #[test]
    fn test_select_store() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = IrNode::Store {
            base: Box::new(IrNode::Reg(0)),
            offset: 0,
            value: Box::new(IrNode::IntConst(99)),
        };
        let instrs = sel.select_all(&[tree]);
        // li for the value, then store
        assert_eq!(instrs.len(), 2);
        assert_eq!(instrs[1].mnemonic, "store");
    }

    #[test]
    fn test_select_call() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = IrNode::Call {
            name: "foo".to_string(),
            args: vec![IrNode::IntConst(1), IrNode::IntConst(2)],
        };
        let instrs = sel.select_all(&[tree]);
        // li, li, call
        assert_eq!(instrs.len(), 3);
        assert_eq!(instrs[2].mnemonic, "call");
    }

    #[test]
    fn test_select_cond_branch() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = IrNode::CondBranch {
            cond: Box::new(IrNode::Reg(0)),
            true_label: "then".to_string(),
            false_label: "else".to_string(),
        };
        let instrs = sel.select_all(&[tree]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].mnemonic, "br");
    }

    #[test]
    fn test_select_return_with_value() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = IrNode::Return(Some(Box::new(IrNode::IntConst(0))));
        let instrs = sel.select_all(&[tree]);
        assert_eq!(instrs.len(), 2); // li + ret
        assert_eq!(instrs[1].mnemonic, "ret");
    }

    #[test]
    fn test_select_return_void() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = IrNode::Return(None);
        let instrs = sel.select_all(&[tree]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].mnemonic, "ret");
    }

    #[test]
    fn test_pattern_matching() {
        let pat = InstrPattern::new("addi", 1, PatternKind::BinOpImm(IrBinOp::Add));
        let node = make_add(IrNode::Reg(0), IrNode::IntConst(5));
        assert!(pat.matches(&node));

        // Doesn't match if right is not immediate
        let node2 = make_add(IrNode::Reg(0), IrNode::Reg(1));
        assert!(!pat.matches(&node2));
    }

    #[test]
    fn test_selection_statistics() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        sel.select_all(&[
            make_add(IrNode::IntConst(1), IrNode::IntConst(2)),
            IrNode::Return(None),
        ]);
        let stats = sel.statistics();
        assert!(stats.instructions_emitted > 0);
        assert!(stats.total_cost > 0);
    }

    #[test]
    fn test_node_count() {
        let tree = make_add(
            IrNode::IntConst(1),
            make_mul(IrNode::Reg(0), IrNode::IntConst(2)),
        );
        assert_eq!(tree.node_count(), 5);
    }

    #[test]
    fn test_selected_instr_display() {
        let instr = SelectedInstr::new(
            "add",
            Some(TargetOperand::Register(0)),
            vec![TargetOperand::Register(1), TargetOperand::Register(2)],
            1,
        );
        let s = format!("{instr}");
        assert!(s.contains("add"));
        assert!(s.contains("r0"));
        assert!(s.contains("r1"));
    }

    #[test]
    fn test_lowering_rule_set() {
        let mut rules = LoweringRuleSet::new();
        rules.add_rule(LoweringRule::new("add_rr", PatternKind::BinOp(IrBinOp::Add), 1, 1));
        rules.add_rule(LoweringRule::new("add_ri", PatternKind::BinOpImm(IrBinOp::Add), 1, 1));
        assert_eq!(rules.len(), 2);
        assert!(!rules.is_empty());
        assert!(rules.best_rule(&PatternKind::BinOp(IrBinOp::Add)).is_some());
    }

    #[test]
    fn test_target_operand_display() {
        assert_eq!(format!("{}", TargetOperand::Register(3)), "r3");
        assert_eq!(format!("{}", TargetOperand::Immediate(42)), "#42");
        assert_eq!(
            format!("{}", TargetOperand::Memory { base: 1, offset: 8 }),
            "[r1+8]"
        );
        assert_eq!(
            format!("{}", TargetOperand::Memory { base: 1, offset: -4 }),
            "[r1-4]"
        );
    }

    #[test]
    fn test_select_unary_neg() {
        let isa = TargetIsa::simple_risc();
        let mut sel = InstructionSelector::new(isa);
        let tree = IrNode::UnaryOp {
            op: IrUnaryOp::Neg,
            operand: Box::new(IrNode::Reg(0)),
        };
        let instrs = sel.select_all(&[tree]);
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0].mnemonic, "neg");
    }

    #[test]
    fn test_avg_cost() {
        let mut stats = SelectionStats::default();
        stats.instructions_emitted = 4;
        stats.total_cost = 12;
        assert!((stats.avg_cost() - 3.0).abs() < f64::EPSILON);
    }
}
