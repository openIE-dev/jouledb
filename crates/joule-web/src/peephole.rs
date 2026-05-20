//! Peephole optimizer — sliding window over instructions, pattern matching rules,
//! strength reduction (mul by power of 2 to shift), identity elimination
//! (add 0, mul 1), redundant load elimination, rule statistics.

use std::collections::HashMap;
use std::fmt;

// ── Instruction representation ──────────────────────────────────────────────

/// An operand in the peephole optimizer's IR.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operand {
    /// Register.
    Reg(u32),
    /// Immediate integer.
    Imm(i64),
    /// Memory: base register + offset.
    Mem { base: u32, offset: i64 },
    /// Label.
    Label(String),
}

impl fmt::Display for Operand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Reg(r) => write!(f, "r{r}"),
            Self::Imm(v) => write!(f, "#{v}"),
            Self::Mem { base, offset } => {
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

/// An instruction for peephole optimization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instr {
    /// Opcode.
    pub opcode: Opcode,
    /// Destination operand (if any).
    pub dest: Option<Operand>,
    /// Source operands.
    pub sources: Vec<Operand>,
}

impl Instr {
    /// Create a new instruction.
    pub fn new(opcode: Opcode, dest: Option<Operand>, sources: Vec<Operand>) -> Self {
        Self {
            opcode,
            dest,
            sources,
        }
    }

    /// Whether this is a no-op.
    pub fn is_nop(&self) -> bool {
        self.opcode == Opcode::Nop
    }
}

impl fmt::Display for Instr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.opcode)?;
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
        Ok(())
    }
}

/// Opcodes recognized by the peephole optimizer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    Add,
    Sub,
    Mul,
    Div,
    Shl,
    Shr,
    And,
    Or,
    Xor,
    Mov,
    Load,
    Store,
    Nop,
    Neg,
    Not,
    Cmp,
    Jmp,
    Ret,
    Li,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Add => "add",
            Self::Sub => "sub",
            Self::Mul => "mul",
            Self::Div => "div",
            Self::Shl => "shl",
            Self::Shr => "shr",
            Self::And => "and",
            Self::Or => "or",
            Self::Xor => "xor",
            Self::Mov => "mov",
            Self::Load => "load",
            Self::Store => "store",
            Self::Nop => "nop",
            Self::Neg => "neg",
            Self::Not => "not",
            Self::Cmp => "cmp",
            Self::Jmp => "jmp",
            Self::Ret => "ret",
            Self::Li => "li",
        };
        write!(f, "{s}")
    }
}

// ── Peephole rules ──────────────────────────────────────────────────────────

/// The name of a peephole optimization rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RuleName {
    /// x + 0 -> x (or 0 + x -> x).
    AddZero,
    /// x - 0 -> x.
    SubZero,
    /// x * 1 -> x (or 1 * x -> x).
    MulOne,
    /// x * 0 -> 0 (or 0 * x -> 0).
    MulZero,
    /// x * (power of 2) -> x << log2.
    MulPowerOfTwo,
    /// x / 1 -> x.
    DivOne,
    /// x & 0 -> 0.
    AndZero,
    /// x | 0 -> x.
    OrZero,
    /// x ^ 0 -> x.
    XorZero,
    /// x ^ x -> 0.
    XorSelf,
    /// x - x -> 0.
    SubSelf,
    /// mov x, x -> nop.
    MovSelf,
    /// Double negation: neg(neg(x)) -> x.
    DoubleNeg,
    /// Redundant load after store to same location.
    RedundantLoad,
    /// Store then load to same address with same value.
    StoreLoadSame,
    /// x << 0 -> x.
    ShlZero,
    /// x >> 0 -> x.
    ShrZero,
}

impl fmt::Display for RuleName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::AddZero => "add_zero",
            Self::SubZero => "sub_zero",
            Self::MulOne => "mul_one",
            Self::MulZero => "mul_zero",
            Self::MulPowerOfTwo => "mul_pow2",
            Self::DivOne => "div_one",
            Self::AndZero => "and_zero",
            Self::OrZero => "or_zero",
            Self::XorZero => "xor_zero",
            Self::XorSelf => "xor_self",
            Self::SubSelf => "sub_self",
            Self::MovSelf => "mov_self",
            Self::DoubleNeg => "double_neg",
            Self::RedundantLoad => "redundant_load",
            Self::StoreLoadSame => "store_load_same",
            Self::ShlZero => "shl_zero",
            Self::ShrZero => "shr_zero",
        };
        write!(f, "{s}")
    }
}

// ── Rule statistics ─────────────────────────────────────────────────────────

/// Statistics about which rules were applied.
#[derive(Debug, Clone, Default)]
pub struct RuleStats {
    /// Count per rule.
    pub counts: HashMap<RuleName, u32>,
    /// Total instructions before optimization.
    pub before_count: u32,
    /// Total instructions after optimization.
    pub after_count: u32,
}

impl RuleStats {
    /// Record a rule application.
    fn record(&mut self, rule: RuleName) {
        *self.counts.entry(rule).or_insert(0) += 1;
    }

    /// Total number of rule applications.
    pub fn total_applications(&self) -> u32 {
        self.counts.values().sum()
    }

    /// Instructions eliminated.
    pub fn eliminated(&self) -> u32 {
        self.before_count.saturating_sub(self.after_count)
    }

    /// Reduction ratio.
    pub fn reduction_ratio(&self) -> f64 {
        if self.before_count == 0 {
            return 0.0;
        }
        self.eliminated() as f64 / self.before_count as f64
    }
}

// ── Peephole optimizer ──────────────────────────────────────────────────────

/// Check if a value is a power of 2 and return its log2.
fn log2_if_power_of_two(val: i64) -> Option<u32> {
    if val <= 0 {
        return None;
    }
    let v = val as u64;
    if v & (v - 1) == 0 {
        Some(v.trailing_zeros())
    } else {
        None
    }
}

/// The peephole optimizer.
pub struct PeepholeOptimizer {
    /// Window size for pattern matching.
    window_size: usize,
    /// Statistics.
    stats: RuleStats,
    /// Which rules are enabled.
    enabled_rules: HashSet<RuleName>,
}

use std::collections::HashSet;

impl PeepholeOptimizer {
    /// Create a new optimizer with all rules enabled.
    pub fn new() -> Self {
        let mut enabled = HashSet::new();
        for rule in &[
            RuleName::AddZero,
            RuleName::SubZero,
            RuleName::MulOne,
            RuleName::MulZero,
            RuleName::MulPowerOfTwo,
            RuleName::DivOne,
            RuleName::AndZero,
            RuleName::OrZero,
            RuleName::XorZero,
            RuleName::XorSelf,
            RuleName::SubSelf,
            RuleName::MovSelf,
            RuleName::DoubleNeg,
            RuleName::RedundantLoad,
            RuleName::StoreLoadSame,
            RuleName::ShlZero,
            RuleName::ShrZero,
        ] {
            enabled.insert(*rule);
        }
        Self {
            window_size: 2,
            stats: RuleStats::default(),
            enabled_rules: enabled,
        }
    }

    /// Set the window size.
    pub fn set_window_size(&mut self, size: usize) {
        self.window_size = size.max(1);
    }

    /// Disable a specific rule.
    pub fn disable_rule(&mut self, rule: RuleName) {
        self.enabled_rules.remove(&rule);
    }

    /// Enable a specific rule.
    pub fn enable_rule(&mut self, rule: RuleName) {
        self.enabled_rules.insert(rule);
    }

    /// Optimize a sequence of instructions.
    pub fn optimize(&mut self, instructions: &[Instr]) -> Vec<Instr> {
        self.stats = RuleStats::default();
        self.stats.before_count = instructions.len() as u32;

        let mut result: Vec<Instr> = instructions.to_vec();
        let mut changed = true;

        // Iterate until no more changes (max 10 passes to prevent infinite loops)
        let mut pass = 0;
        while changed && pass < 10 {
            changed = false;
            pass += 1;

            // Single-instruction patterns
            let mut i = 0;
            while i < result.len() {
                if let Some(replacement) = self.try_single_pattern(&result[i]) {
                    result[i] = replacement;
                    changed = true;
                }
                i += 1;
            }

            // Two-instruction patterns (window = 2)
            if self.window_size >= 2 {
                let mut i = 0;
                while i + 1 < result.len() {
                    if let Some(replacements) = self.try_pair_pattern(&result[i], &result[i + 1]) {
                        // Replace the pair
                        result.splice(i..i + 2, replacements);
                        changed = true;
                        // Don't advance — re-check at same position
                    } else {
                        i += 1;
                    }
                }
            }

            // Remove nops
            let before = result.len();
            result.retain(|instr| !instr.is_nop());
            if result.len() < before {
                changed = true;
            }
        }

        self.stats.after_count = result.len() as u32;
        result
    }

    /// Try to optimize a single instruction.
    fn try_single_pattern(&mut self, instr: &Instr) -> Option<Instr> {
        match instr.opcode {
            // add rd, rs, #0 -> mov rd, rs
            Opcode::Add => {
                if self.enabled_rules.contains(&RuleName::AddZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::AddZero);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                    if let Some(Operand::Imm(0)) = instr.sources.first() {
                        if instr.sources.len() > 1 {
                            self.stats.record(RuleName::AddZero);
                            return Some(Instr::new(
                                Opcode::Mov,
                                instr.dest.clone(),
                                vec![instr.sources[1].clone()],
                            ));
                        }
                    }
                }
            }

            // sub rd, rs, #0 -> mov rd, rs
            Opcode::Sub => {
                if self.enabled_rules.contains(&RuleName::SubZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::SubZero);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                }
                // sub rd, rs, rs -> li rd, #0
                if self.enabled_rules.contains(&RuleName::SubSelf) {
                    if instr.sources.len() >= 2 && instr.sources[0] == instr.sources[1] {
                        self.stats.record(RuleName::SubSelf);
                        return Some(Instr::new(
                            Opcode::Li,
                            instr.dest.clone(),
                            vec![Operand::Imm(0)],
                        ));
                    }
                }
            }

            // mul rd, rs, #1 -> mov rd, rs
            Opcode::Mul => {
                if self.enabled_rules.contains(&RuleName::MulOne) {
                    if let Some(Operand::Imm(1)) = instr.sources.get(1) {
                        self.stats.record(RuleName::MulOne);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                    if let Some(Operand::Imm(1)) = instr.sources.first() {
                        if instr.sources.len() > 1 {
                            self.stats.record(RuleName::MulOne);
                            return Some(Instr::new(
                                Opcode::Mov,
                                instr.dest.clone(),
                                vec![instr.sources[1].clone()],
                            ));
                        }
                    }
                }
                // mul rd, rs, #0 -> li rd, #0
                if self.enabled_rules.contains(&RuleName::MulZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::MulZero);
                        return Some(Instr::new(
                            Opcode::Li,
                            instr.dest.clone(),
                            vec![Operand::Imm(0)],
                        ));
                    }
                    if let Some(Operand::Imm(0)) = instr.sources.first() {
                        self.stats.record(RuleName::MulZero);
                        return Some(Instr::new(
                            Opcode::Li,
                            instr.dest.clone(),
                            vec![Operand::Imm(0)],
                        ));
                    }
                }
                // mul rd, rs, #(power of 2) -> shl rd, rs, #log2
                if self.enabled_rules.contains(&RuleName::MulPowerOfTwo) {
                    if let Some(Operand::Imm(val)) = instr.sources.get(1) {
                        if *val > 1 {
                            if let Some(shift) = log2_if_power_of_two(*val) {
                                self.stats.record(RuleName::MulPowerOfTwo);
                                return Some(Instr::new(
                                    Opcode::Shl,
                                    instr.dest.clone(),
                                    vec![instr.sources[0].clone(), Operand::Imm(shift as i64)],
                                ));
                            }
                        }
                    }
                }
            }

            // div rd, rs, #1 -> mov rd, rs
            Opcode::Div => {
                if self.enabled_rules.contains(&RuleName::DivOne) {
                    if let Some(Operand::Imm(1)) = instr.sources.get(1) {
                        self.stats.record(RuleName::DivOne);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                }
            }

            // and rd, rs, #0 -> li rd, #0
            Opcode::And => {
                if self.enabled_rules.contains(&RuleName::AndZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::AndZero);
                        return Some(Instr::new(
                            Opcode::Li,
                            instr.dest.clone(),
                            vec![Operand::Imm(0)],
                        ));
                    }
                }
            }

            // or rd, rs, #0 -> mov rd, rs
            Opcode::Or => {
                if self.enabled_rules.contains(&RuleName::OrZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::OrZero);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                }
            }

            // xor rd, rs, #0 -> mov rd, rs
            Opcode::Xor => {
                if self.enabled_rules.contains(&RuleName::XorZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::XorZero);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                }
                // xor rd, rs, rs -> li rd, #0
                if self.enabled_rules.contains(&RuleName::XorSelf) {
                    if instr.sources.len() >= 2 && instr.sources[0] == instr.sources[1] {
                        self.stats.record(RuleName::XorSelf);
                        return Some(Instr::new(
                            Opcode::Li,
                            instr.dest.clone(),
                            vec![Operand::Imm(0)],
                        ));
                    }
                }
            }

            // mov rd, rd -> nop
            Opcode::Mov => {
                if self.enabled_rules.contains(&RuleName::MovSelf) {
                    if let (Some(dest), Some(src)) = (&instr.dest, instr.sources.first()) {
                        if dest == src {
                            self.stats.record(RuleName::MovSelf);
                            return Some(Instr::new(Opcode::Nop, None, vec![]));
                        }
                    }
                }
            }

            // shl rd, rs, #0 -> mov rd, rs
            Opcode::Shl => {
                if self.enabled_rules.contains(&RuleName::ShlZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::ShlZero);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                }
            }

            // shr rd, rs, #0 -> mov rd, rs
            Opcode::Shr => {
                if self.enabled_rules.contains(&RuleName::ShrZero) {
                    if let Some(Operand::Imm(0)) = instr.sources.get(1) {
                        self.stats.record(RuleName::ShrZero);
                        return Some(Instr::new(
                            Opcode::Mov,
                            instr.dest.clone(),
                            vec![instr.sources[0].clone()],
                        ));
                    }
                }
            }

            _ => {}
        }

        None
    }

    /// Try to optimize a pair of instructions.
    fn try_pair_pattern(&mut self, first: &Instr, second: &Instr) -> Option<Vec<Instr>> {
        // neg(neg(x)) -> mov x
        if self.enabled_rules.contains(&RuleName::DoubleNeg) {
            if first.opcode == Opcode::Neg && second.opcode == Opcode::Neg {
                if let (Some(first_dest), Some(second_src)) =
                    (&first.dest, second.sources.first())
                {
                    if first_dest == second_src {
                        self.stats.record(RuleName::DoubleNeg);
                        return Some(vec![Instr::new(
                            Opcode::Mov,
                            second.dest.clone(),
                            first.sources.clone(),
                        )]);
                    }
                }
            }
        }

        // store [addr], val ; load rd, [addr] -> store [addr], val ; mov rd, val
        if self.enabled_rules.contains(&RuleName::StoreLoadSame) {
            if first.opcode == Opcode::Store && second.opcode == Opcode::Load {
                // store sources: [addr], val   load sources: [addr]
                if let (Some(store_addr), Some(load_addr)) =
                    (first.sources.first(), second.sources.first())
                {
                    if store_addr == load_addr && first.sources.len() >= 2 {
                        self.stats.record(RuleName::StoreLoadSame);
                        return Some(vec![
                            first.clone(),
                            Instr::new(
                                Opcode::Mov,
                                second.dest.clone(),
                                vec![first.sources[1].clone()],
                            ),
                        ]);
                    }
                }
            }
        }

        None
    }

    /// Get statistics from the last optimization run.
    pub fn statistics(&self) -> &RuleStats {
        &self.stats
    }
}

impl Default for PeepholeOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn add_imm(dest: u32, src: u32, imm: i64) -> Instr {
        Instr::new(
            Opcode::Add,
            Some(Operand::Reg(dest)),
            vec![Operand::Reg(src), Operand::Imm(imm)],
        )
    }

    fn mul_imm(dest: u32, src: u32, imm: i64) -> Instr {
        Instr::new(
            Opcode::Mul,
            Some(Operand::Reg(dest)),
            vec![Operand::Reg(src), Operand::Imm(imm)],
        )
    }

    #[test]
    fn test_add_zero_elimination() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![add_imm(0, 1, 0)];
        let result = opt.optimize(&instrs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, Opcode::Mov);
    }

    #[test]
    fn test_mul_one_elimination() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![mul_imm(0, 1, 1)];
        let result = opt.optimize(&instrs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, Opcode::Mov);
    }

    #[test]
    fn test_mul_zero_elimination() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![mul_imm(0, 1, 0)];
        let result = opt.optimize(&instrs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, Opcode::Li);
        assert_eq!(result[0].sources, vec![Operand::Imm(0)]);
    }

    #[test]
    fn test_strength_reduction_mul_power_of_two() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![mul_imm(0, 1, 8)]; // *8 -> <<3
        let result = opt.optimize(&instrs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, Opcode::Shl);
        assert_eq!(result[0].sources[1], Operand::Imm(3));
    }

    #[test]
    fn test_strength_reduction_mul_16() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![mul_imm(0, 1, 16)]; // *16 -> <<4
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Shl);
        assert_eq!(result[0].sources[1], Operand::Imm(4));
    }

    #[test]
    fn test_no_strength_reduction_for_non_power_of_two() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![mul_imm(0, 1, 7)]; // 7 is not a power of 2
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Mul);
    }

    #[test]
    fn test_sub_self() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![Instr::new(
            Opcode::Sub,
            Some(Operand::Reg(0)),
            vec![Operand::Reg(1), Operand::Reg(1)],
        )];
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Li);
        assert_eq!(result[0].sources, vec![Operand::Imm(0)]);
    }

    #[test]
    fn test_xor_self() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![Instr::new(
            Opcode::Xor,
            Some(Operand::Reg(0)),
            vec![Operand::Reg(1), Operand::Reg(1)],
        )];
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Li);
    }

    #[test]
    fn test_mov_self_elimination() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![Instr::new(
            Opcode::Mov,
            Some(Operand::Reg(3)),
            vec![Operand::Reg(3)],
        )];
        let result = opt.optimize(&instrs);
        assert!(result.is_empty()); // nop gets removed
    }

    #[test]
    fn test_double_neg() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![
            Instr::new(Opcode::Neg, Some(Operand::Reg(1)), vec![Operand::Reg(0)]),
            Instr::new(Opcode::Neg, Some(Operand::Reg(2)), vec![Operand::Reg(1)]),
        ];
        let result = opt.optimize(&instrs);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].opcode, Opcode::Mov);
    }

    #[test]
    fn test_store_load_same() {
        let mut opt = PeepholeOptimizer::new();
        let addr = Operand::Mem { base: 0, offset: 8 };
        let instrs = vec![
            Instr::new(Opcode::Store, None, vec![addr.clone(), Operand::Reg(1)]),
            Instr::new(Opcode::Load, Some(Operand::Reg(2)), vec![addr]),
        ];
        let result = opt.optimize(&instrs);
        assert_eq!(result.len(), 2);
        assert_eq!(result[1].opcode, Opcode::Mov);
    }

    #[test]
    fn test_div_one() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![Instr::new(
            Opcode::Div,
            Some(Operand::Reg(0)),
            vec![Operand::Reg(1), Operand::Imm(1)],
        )];
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Mov);
    }

    #[test]
    fn test_and_zero() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![Instr::new(
            Opcode::And,
            Some(Operand::Reg(0)),
            vec![Operand::Reg(1), Operand::Imm(0)],
        )];
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Li);
    }

    #[test]
    fn test_or_zero() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![Instr::new(
            Opcode::Or,
            Some(Operand::Reg(0)),
            vec![Operand::Reg(1), Operand::Imm(0)],
        )];
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Mov);
    }

    #[test]
    fn test_shl_zero() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![Instr::new(
            Opcode::Shl,
            Some(Operand::Reg(0)),
            vec![Operand::Reg(1), Operand::Imm(0)],
        )];
        let result = opt.optimize(&instrs);
        assert_eq!(result[0].opcode, Opcode::Mov);
    }

    #[test]
    fn test_statistics() {
        let mut opt = PeepholeOptimizer::new();
        let instrs = vec![
            add_imm(0, 1, 0),
            mul_imm(2, 3, 1),
            mul_imm(4, 5, 8),
        ];
        opt.optimize(&instrs);
        let stats = opt.statistics();
        assert_eq!(stats.total_applications(), 3);
        assert_eq!(stats.before_count, 3);
    }

    #[test]
    fn test_disable_rule() {
        let mut opt = PeepholeOptimizer::new();
        opt.disable_rule(RuleName::AddZero);
        let instrs = vec![add_imm(0, 1, 0)];
        let result = opt.optimize(&instrs);
        // Rule disabled, so add stays
        assert_eq!(result[0].opcode, Opcode::Add);
    }

    #[test]
    fn test_log2_power_of_two() {
        assert_eq!(log2_if_power_of_two(1), Some(0));
        assert_eq!(log2_if_power_of_two(2), Some(1));
        assert_eq!(log2_if_power_of_two(4), Some(2));
        assert_eq!(log2_if_power_of_two(1024), Some(10));
        assert_eq!(log2_if_power_of_two(3), None);
        assert_eq!(log2_if_power_of_two(0), None);
        assert_eq!(log2_if_power_of_two(-1), None);
    }

    #[test]
    fn test_reduction_ratio() {
        let mut stats = RuleStats::default();
        stats.before_count = 10;
        stats.after_count = 7;
        assert!((stats.reduction_ratio() - 0.3).abs() < f64::EPSILON);
    }
}
