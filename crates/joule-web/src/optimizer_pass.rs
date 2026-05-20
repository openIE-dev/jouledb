//! Optimization passes — pass manager, constant propagation, copy propagation,
//! dead code elimination, common subexpression elimination, loop invariant
//! code motion, pass ordering, optimization level (O0/O1/O2).

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── IR representation for optimization ──────────────────────────────────────

/// A value in the IR.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Integer constant.
    Int(i64),
    /// Float constant.
    Float(f64),
    /// Boolean constant.
    Bool(bool),
    /// Variable reference.
    Var(String),
    /// Undefined / unknown.
    Undef,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Var(v) => write!(f, "{v}"),
            Self::Undef => write!(f, "undef"),
        }
    }
}

/// Binary operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    And,
    Or,
    Shl,
    Shr,
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
            Self::And => "&&",
            Self::Or => "||",
            Self::Shl => "<<",
            Self::Shr => ">>",
        };
        write!(f, "{s}")
    }
}

/// A single IR statement.
#[derive(Debug, Clone, PartialEq)]
pub enum Stmt {
    /// Assignment: dest = value.
    Assign {
        dest: String,
        value: Value,
    },
    /// Binary operation: dest = left op right.
    BinOp {
        dest: String,
        op: BinOp,
        left: Value,
        right: Value,
    },
    /// Copy: dest = source.
    Copy {
        dest: String,
        source: String,
    },
    /// Branch if condition is true.
    BranchIf {
        cond: Value,
        target: String,
    },
    /// Unconditional jump.
    Jump(String),
    /// Label.
    Label(String),
    /// Return a value.
    Return(Value),
    /// No-op (placeholder for deleted instructions).
    Nop,
    /// Call: dest = function(args...).
    Call {
        dest: String,
        function: String,
        args: Vec<Value>,
    },
}

impl Stmt {
    /// Get the variable defined by this statement (if any).
    pub fn defined_var(&self) -> Option<&str> {
        match self {
            Self::Assign { dest, .. }
            | Self::BinOp { dest, .. }
            | Self::Copy { dest, .. }
            | Self::Call { dest, .. } => Some(dest),
            _ => None,
        }
    }

    /// Get all variables used (read) by this statement.
    pub fn used_vars(&self) -> Vec<String> {
        let mut vars = Vec::new();
        let collect_value = |v: &Value, vs: &mut Vec<String>| {
            if let Value::Var(name) = v {
                vs.push(name.clone());
            }
        };
        match self {
            Self::Assign { value, .. } => collect_value(value, &mut vars),
            Self::BinOp { left, right, .. } => {
                collect_value(left, &mut vars);
                collect_value(right, &mut vars);
            }
            Self::Copy { source, .. } => vars.push(source.clone()),
            Self::BranchIf { cond, .. } => collect_value(cond, &mut vars),
            Self::Return(v) => collect_value(v, &mut vars),
            Self::Call { args, .. } => {
                for a in args {
                    collect_value(a, &mut vars);
                }
            }
            Self::Jump(_) | Self::Label(_) | Self::Nop => {}
        }
        vars
    }

    /// Whether this is a Nop.
    pub fn is_nop(&self) -> bool {
        matches!(self, Self::Nop)
    }
}

// ── Optimization level ──────────────────────────────────────────────────────

/// Optimization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OptLevel {
    /// No optimizations.
    O0,
    /// Basic optimizations (constant prop, copy prop, DCE).
    O1,
    /// Aggressive optimizations (CSE, LICM, multiple passes).
    O2,
}

impl fmt::Display for OptLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::O0 => "O0",
            Self::O1 => "O1",
            Self::O2 => "O2",
        };
        write!(f, "{s}")
    }
}

// ── Pass trait and results ──────────────────────────────────────────────────

/// Statistics from a single pass run.
#[derive(Debug, Clone, Default)]
pub struct PassStats {
    /// Number of statements modified.
    pub modified: u32,
    /// Number of statements removed.
    pub removed: u32,
    /// Number of statements added.
    pub added: u32,
}

impl PassStats {
    /// Whether any changes were made.
    pub fn changed(&self) -> bool {
        self.modified > 0 || self.removed > 0 || self.added > 0
    }
}

/// Named pass result.
#[derive(Debug, Clone)]
pub struct PassResult {
    /// Pass name.
    pub name: String,
    /// Statistics.
    pub stats: PassStats,
}

// ── Constant propagation ────────────────────────────────────────────────────

/// Constant propagation: replace variables with known constant values.
pub fn constant_propagation(stmts: &mut Vec<Stmt>) -> PassStats {
    let mut stats = PassStats::default();
    let mut constants: HashMap<String, Value> = HashMap::new();

    for i in 0..stmts.len() {
        // First, substitute known constants into uses
        let stmt = stmts[i].clone();
        let new_stmt = substitute_constants(&stmt, &constants);
        if new_stmt != stmt {
            stmts[i] = new_stmt.clone();
            stats.modified += 1;
        }

        // Then, try to evaluate constant expressions
        let folded = constant_fold(&stmts[i]);
        if folded != stmts[i] {
            stmts[i] = folded.clone();
            stats.modified += 1;
        }

        // Track constant definitions
        match &stmts[i] {
            Stmt::Assign { dest, value } if !matches!(value, Value::Var(_) | Value::Undef) => {
                constants.insert(dest.clone(), value.clone());
            }
            Stmt::BinOp { dest, op: _, left, right: _ } => {
                // If the result is a constant after folding
                if let Stmt::Assign { value, .. } = &stmts[i] {
                    if !matches!(value, Value::Var(_) | Value::Undef) {
                        constants.insert(dest.clone(), value.clone());
                    }
                }
                // If left is the only remaining value (identity fold already happened)
                if matches!(left, Value::Int(_) | Value::Float(_) | Value::Bool(_)) {
                    // Check if folded to an assign
                }
            }
            _ => {
                // If a variable is redefined with unknown value, remove from constants
                if let Some(dest) = stmts[i].defined_var() {
                    if !matches!(&stmts[i], Stmt::Assign { value, .. } if !matches!(value, Value::Var(_) | Value::Undef))
                    {
                        constants.remove(dest);
                    }
                }
            }
        }

        // Labels and branches invalidate all constants (conservative)
        if matches!(&stmts[i], Stmt::Label(_) | Stmt::Jump(_) | Stmt::BranchIf { .. }) {
            constants.clear();
        }
    }

    stats
}

/// Substitute known constant values into a statement.
fn substitute_constants(stmt: &Stmt, constants: &HashMap<String, Value>) -> Stmt {
    let sub = |v: &Value| -> Value {
        if let Value::Var(name) = v {
            if let Some(c) = constants.get(name) {
                return c.clone();
            }
        }
        v.clone()
    };

    match stmt {
        Stmt::Assign { dest, value } => Stmt::Assign {
            dest: dest.clone(),
            value: sub(value),
        },
        Stmt::BinOp { dest, op, left, right } => Stmt::BinOp {
            dest: dest.clone(),
            op: *op,
            left: sub(left),
            right: sub(right),
        },
        Stmt::BranchIf { cond, target } => Stmt::BranchIf {
            cond: sub(cond),
            target: target.clone(),
        },
        Stmt::Return(v) => Stmt::Return(sub(v)),
        Stmt::Call { dest, function, args } => Stmt::Call {
            dest: dest.clone(),
            function: function.clone(),
            args: args.iter().map(|a| sub(a)).collect(),
        },
        other => other.clone(),
    }
}

/// Constant fold a statement if possible.
fn constant_fold(stmt: &Stmt) -> Stmt {
    if let Stmt::BinOp { dest, op, left, right } = stmt {
        if let (Value::Int(l), Value::Int(r)) = (left, right) {
            let result = match op {
                BinOp::Add => Some(Value::Int(l.wrapping_add(*r))),
                BinOp::Sub => Some(Value::Int(l.wrapping_sub(*r))),
                BinOp::Mul => Some(Value::Int(l.wrapping_mul(*r))),
                BinOp::Div if *r != 0 => Some(Value::Int(l.wrapping_div(*r))),
                BinOp::Mod if *r != 0 => Some(Value::Int(l.wrapping_rem(*r))),
                BinOp::Eq => Some(Value::Bool(l == r)),
                BinOp::Ne => Some(Value::Bool(l != r)),
                BinOp::Lt => Some(Value::Bool(l < r)),
                BinOp::Gt => Some(Value::Bool(l > r)),
                BinOp::Shl => Some(Value::Int(l.wrapping_shl(*r as u32))),
                BinOp::Shr => Some(Value::Int(l.wrapping_shr(*r as u32))),
                _ => None,
            };
            if let Some(val) = result {
                return Stmt::Assign {
                    dest: dest.clone(),
                    value: val,
                };
            }
        }
        if let (Value::Bool(l), Value::Bool(r)) = (left, right) {
            let result = match op {
                BinOp::And => Some(Value::Bool(*l && *r)),
                BinOp::Or => Some(Value::Bool(*l || *r)),
                BinOp::Eq => Some(Value::Bool(l == r)),
                BinOp::Ne => Some(Value::Bool(l != r)),
                _ => None,
            };
            if let Some(val) = result {
                return Stmt::Assign {
                    dest: dest.clone(),
                    value: val,
                };
            }
        }
    }
    stmt.clone()
}

// ── Copy propagation ────────────────────────────────────────────────────────

/// Copy propagation: replace uses of `x` with `y` when `x = y` is known.
pub fn copy_propagation(stmts: &mut Vec<Stmt>) -> PassStats {
    let mut stats = PassStats::default();
    let mut copies: HashMap<String, String> = HashMap::new();

    for i in 0..stmts.len() {
        // Substitute copies in uses
        let stmt = stmts[i].clone();
        let new_stmt = substitute_copies(&stmt, &copies);
        if new_stmt != stmt {
            stmts[i] = new_stmt;
            stats.modified += 1;
        }

        // Track copies
        if let Stmt::Copy { dest, source } = &stmts[i] {
            // Resolve chains: if source is itself a copy, follow it
            let resolved = resolve_copy(source, &copies);
            copies.insert(dest.clone(), resolved);
        } else if let Some(def) = stmts[i].defined_var() {
            // Non-copy definition kills the copy
            copies.remove(def);
        }

        // Labels invalidate copies
        if matches!(&stmts[i], Stmt::Label(_) | Stmt::Jump(_) | Stmt::BranchIf { .. }) {
            copies.clear();
        }
    }

    stats
}

/// Resolve copy chains.
fn resolve_copy(name: &str, copies: &HashMap<String, String>) -> String {
    let mut current = name.to_string();
    let mut visited = HashSet::new();
    while let Some(src) = copies.get(&current) {
        if !visited.insert(current.clone()) {
            break; // cycle
        }
        current = src.clone();
    }
    current
}

/// Substitute copies into a statement.
fn substitute_copies(stmt: &Stmt, copies: &HashMap<String, String>) -> Stmt {
    let sub = |v: &Value| -> Value {
        if let Value::Var(name) = v {
            let resolved = resolve_copy(name, copies);
            if resolved != *name {
                return Value::Var(resolved);
            }
        }
        v.clone()
    };

    match stmt {
        Stmt::Assign { dest, value } => Stmt::Assign {
            dest: dest.clone(),
            value: sub(value),
        },
        Stmt::BinOp { dest, op, left, right } => Stmt::BinOp {
            dest: dest.clone(),
            op: *op,
            left: sub(left),
            right: sub(right),
        },
        Stmt::Copy { dest, source } => {
            let resolved = resolve_copy(source, copies);
            Stmt::Copy {
                dest: dest.clone(),
                source: resolved,
            }
        }
        Stmt::BranchIf { cond, target } => Stmt::BranchIf {
            cond: sub(cond),
            target: target.clone(),
        },
        Stmt::Return(v) => Stmt::Return(sub(v)),
        Stmt::Call { dest, function, args } => Stmt::Call {
            dest: dest.clone(),
            function: function.clone(),
            args: args.iter().map(|a| sub(a)).collect(),
        },
        other => other.clone(),
    }
}

// ── Dead code elimination ───────────────────────────────────────────────────

/// Dead code elimination: remove statements whose defined variables are never used.
pub fn dead_code_elimination(stmts: &mut Vec<Stmt>) -> PassStats {
    let mut stats = PassStats::default();

    // Build use set — all variables that are read anywhere
    let mut used: HashSet<String> = HashSet::new();
    for stmt in stmts.iter() {
        for var in stmt.used_vars() {
            used.insert(var);
        }
    }

    // Remove definitions of variables never used (except calls which may have side effects)
    for stmt in stmts.iter_mut() {
        let dominated_by_call = matches!(stmt, Stmt::Call { .. });
        if dominated_by_call {
            continue;
        }
        if let Some(def) = stmt.defined_var() {
            if !used.contains(def) {
                *stmt = Stmt::Nop;
                stats.removed += 1;
            }
        }
    }

    // Remove nops
    stmts.retain(|s| !s.is_nop());

    stats
}

// ── Common subexpression elimination ────────────────────────────────────────

/// Key for CSE — an expression that can be looked up.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct CseKey {
    op: BinOp,
    left: String,
    right: String,
}

/// Common subexpression elimination: reuse previously computed expressions.
pub fn common_subexpression_elimination(stmts: &mut Vec<Stmt>) -> PassStats {
    let mut stats = PassStats::default();
    let mut available: HashMap<CseKey, String> = HashMap::new();

    for i in 0..stmts.len() {
        // Labels/jumps invalidate everything
        if matches!(&stmts[i], Stmt::Label(_) | Stmt::Jump(_) | Stmt::BranchIf { .. }) {
            available.clear();
            continue;
        }

        // Non-BinOp definitions kill availability of that result variable
        if !matches!(&stmts[i], Stmt::BinOp { .. }) {
            if let Some(def) = stmts[i].defined_var() {
                let def_str = def.to_string();
                available.retain(|_, v| *v != def_str);
            }
            continue;
        }

        if let Stmt::BinOp { dest, op, left, right } = &stmts[i] {
            let key = CseKey {
                op: *op,
                left: format!("{left}"),
                right: format!("{right}"),
            };

            // Check if any operand in the key was killed (the result var was redefined)
            if let Some(existing) = available.get(&key) {
                // Replace with a copy from the existing result
                stmts[i] = Stmt::Copy {
                    dest: dest.clone(),
                    source: existing.clone(),
                };
                stats.modified += 1;
            } else {
                // Kill any availability whose result is this dest (redefinition)
                let def_str = dest.to_string();
                available.retain(|_, v| *v != def_str);
                available.insert(key, dest.clone());
            }
        }
    }

    stats
}

// ── Loop invariant code motion ──────────────────────────────────────────────

/// Simple loop detection: find back-edges from labels to jumps.
/// Returns (header_label_index, back_edge_jump_index) pairs.
fn find_loops(stmts: &[Stmt]) -> Vec<(usize, usize)> {
    let mut labels: HashMap<String, usize> = HashMap::new();
    let mut loops = Vec::new();

    // First pass: find label positions
    for (i, stmt) in stmts.iter().enumerate() {
        if let Stmt::Label(name) = stmt {
            labels.insert(name.clone(), i);
        }
    }

    // Second pass: find back-edges (jumps to earlier labels)
    for (i, stmt) in stmts.iter().enumerate() {
        let target = match stmt {
            Stmt::Jump(t) => Some(t),
            Stmt::BranchIf { target: t, .. } => Some(t),
            _ => None,
        };
        if let Some(t) = target {
            if let Some(&label_pos) = labels.get(t) {
                if label_pos <= i {
                    loops.push((label_pos, i));
                }
            }
        }
    }

    loops
}

/// Loop invariant code motion: move computations whose operands don't change
/// inside the loop to before the loop header.
pub fn loop_invariant_code_motion(stmts: &mut Vec<Stmt>) -> PassStats {
    let mut stats = PassStats::default();
    let loops = find_loops(stmts);

    for (header, back_edge) in loops {
        // Find variables defined inside the loop
        let mut loop_defs: HashSet<String> = HashSet::new();
        for stmt in &stmts[header..=back_edge] {
            if let Some(def) = stmt.defined_var() {
                loop_defs.insert(def.to_string());
            }
        }

        // Find invariant statements: all used vars are NOT defined in the loop
        let mut to_hoist: Vec<usize> = Vec::new();
        for i in (header + 1)..back_edge {
            let uses = stmts[i].used_vars();
            let is_invariant = !uses.is_empty()
                && uses.iter().all(|u| !loop_defs.contains(u))
                && stmts[i].defined_var().is_some()
                && !matches!(&stmts[i], Stmt::Call { .. }); // Don't hoist calls

            if is_invariant {
                to_hoist.push(i);
            }
        }

        // Hoist: extract and place before the header
        // Process in reverse to keep indices valid
        let mut hoisted_stmts = Vec::new();
        for &idx in to_hoist.iter().rev() {
            let stmt = stmts.remove(idx);
            hoisted_stmts.push(stmt);
            stats.modified += 1;
        }
        hoisted_stmts.reverse();

        // Insert before header (find header in the now-modified vec)
        let new_header = stmts.iter().position(|s| {
            if let Stmt::Label(name) = s {
                if let Stmt::Label(orig) = &stmts.get(header).cloned().unwrap_or(Stmt::Nop) {
                    return name == orig;
                }
            }
            false
        });

        let insert_pos = new_header.unwrap_or(0);
        for (j, stmt) in hoisted_stmts.into_iter().enumerate() {
            stmts.insert(insert_pos + j, stmt);
        }
    }

    stats
}

// ── Pass manager ────────────────────────────────────────────────────────────

/// A named optimization pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PassKind {
    ConstantPropagation,
    CopyPropagation,
    DeadCodeElimination,
    CommonSubexprElimination,
    LoopInvariantCodeMotion,
}

impl fmt::Display for PassKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::ConstantPropagation => "ConstProp",
            Self::CopyPropagation => "CopyProp",
            Self::DeadCodeElimination => "DCE",
            Self::CommonSubexprElimination => "CSE",
            Self::LoopInvariantCodeMotion => "LICM",
        };
        write!(f, "{s}")
    }
}

/// The pass manager runs optimization passes in order.
pub struct PassManager {
    /// Ordered list of passes to run.
    passes: Vec<PassKind>,
    /// Maximum iterations for fixed-point passes.
    max_iterations: u32,
    /// Results from the last run.
    results: Vec<PassResult>,
}

impl PassManager {
    /// Create a pass manager for the given optimization level.
    pub fn for_level(level: OptLevel) -> Self {
        let passes = match level {
            OptLevel::O0 => vec![],
            OptLevel::O1 => vec![
                PassKind::ConstantPropagation,
                PassKind::CopyPropagation,
                PassKind::DeadCodeElimination,
            ],
            OptLevel::O2 => vec![
                PassKind::ConstantPropagation,
                PassKind::CopyPropagation,
                PassKind::CommonSubexprElimination,
                PassKind::LoopInvariantCodeMotion,
                PassKind::DeadCodeElimination,
                // Second round
                PassKind::ConstantPropagation,
                PassKind::DeadCodeElimination,
            ],
        };
        Self {
            passes,
            max_iterations: 3,
            results: Vec::new(),
        }
    }

    /// Create a custom pass manager.
    pub fn custom(passes: Vec<PassKind>) -> Self {
        Self {
            passes,
            max_iterations: 3,
            results: Vec::new(),
        }
    }

    /// Set max iterations for fixed-point convergence.
    pub fn set_max_iterations(&mut self, n: u32) {
        self.max_iterations = n;
    }

    /// Run all passes on the given statements.
    pub fn run(&mut self, stmts: &mut Vec<Stmt>) -> &[PassResult] {
        self.results.clear();

        for pass in &self.passes.clone() {
            let stats = run_pass(*pass, stmts);
            self.results.push(PassResult {
                name: format!("{pass}"),
                stats,
            });
        }

        &self.results
    }

    /// Run passes to fixed point (until no changes).
    pub fn run_to_fixed_point(&mut self, stmts: &mut Vec<Stmt>) -> u32 {
        let mut iterations = 0;
        loop {
            iterations += 1;
            let before_len = stmts.len();
            let before = stmts.clone();

            self.run(stmts);

            if stmts.len() == before_len && *stmts == before {
                break;
            }
            if iterations >= self.max_iterations {
                break;
            }
        }
        iterations
    }

    /// Get the results from the last run.
    pub fn results(&self) -> &[PassResult] {
        &self.results
    }

    /// Get the pass list.
    pub fn passes(&self) -> &[PassKind] {
        &self.passes
    }
}

/// Run a single pass.
fn run_pass(pass: PassKind, stmts: &mut Vec<Stmt>) -> PassStats {
    match pass {
        PassKind::ConstantPropagation => constant_propagation(stmts),
        PassKind::CopyPropagation => copy_propagation(stmts),
        PassKind::DeadCodeElimination => dead_code_elimination(stmts),
        PassKind::CommonSubexprElimination => common_subexpression_elimination(stmts),
        PassKind::LoopInvariantCodeMotion => loop_invariant_code_motion(stmts),
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_propagation_basic() {
        let mut stmts = vec![
            Stmt::Assign { dest: "x".into(), value: Value::Int(10) },
            Stmt::BinOp {
                dest: "y".into(),
                op: BinOp::Add,
                left: Value::Var("x".into()),
                right: Value::Int(5),
            },
        ];
        let stats = constant_propagation(&mut stmts);
        assert!(stats.changed());
        // x should be substituted, then folded: y = 15
        assert_eq!(stmts[1], Stmt::Assign { dest: "y".into(), value: Value::Int(15) });
    }

    #[test]
    fn test_constant_fold_arithmetic() {
        let stmt = Stmt::BinOp {
            dest: "z".into(),
            op: BinOp::Mul,
            left: Value::Int(6),
            right: Value::Int(7),
        };
        let folded = constant_fold(&stmt);
        assert_eq!(folded, Stmt::Assign { dest: "z".into(), value: Value::Int(42) });
    }

    #[test]
    fn test_constant_fold_comparison() {
        let stmt = Stmt::BinOp {
            dest: "c".into(),
            op: BinOp::Lt,
            left: Value::Int(3),
            right: Value::Int(5),
        };
        let folded = constant_fold(&stmt);
        assert_eq!(folded, Stmt::Assign { dest: "c".into(), value: Value::Bool(true) });
    }

    #[test]
    fn test_constant_fold_div_by_zero_no_fold() {
        let stmt = Stmt::BinOp {
            dest: "z".into(),
            op: BinOp::Div,
            left: Value::Int(10),
            right: Value::Int(0),
        };
        let folded = constant_fold(&stmt);
        assert_eq!(folded, stmt); // Not folded
    }

    #[test]
    fn test_copy_propagation() {
        let mut stmts = vec![
            Stmt::Copy { dest: "b".into(), source: "a".into() },
            Stmt::BinOp {
                dest: "c".into(),
                op: BinOp::Add,
                left: Value::Var("b".into()),
                right: Value::Int(1),
            },
        ];
        let stats = copy_propagation(&mut stmts);
        assert!(stats.changed());
        // b should be replaced by a
        if let Stmt::BinOp { left, .. } = &stmts[1] {
            assert_eq!(*left, Value::Var("a".into()));
        } else {
            panic!("expected BinOp");
        }
    }

    #[test]
    fn test_copy_propagation_chain() {
        let mut stmts = vec![
            Stmt::Copy { dest: "b".into(), source: "a".into() },
            Stmt::Copy { dest: "c".into(), source: "b".into() },
            Stmt::Return(Value::Var("c".into())),
        ];
        copy_propagation(&mut stmts);
        // c should resolve through chain to a
        assert_eq!(stmts[2], Stmt::Return(Value::Var("a".into())));
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut stmts = vec![
            Stmt::Assign { dest: "x".into(), value: Value::Int(10) },
            Stmt::Assign { dest: "y".into(), value: Value::Int(20) },
            Stmt::Return(Value::Var("x".into())),
        ];
        let stats = dead_code_elimination(&mut stmts);
        assert_eq!(stats.removed, 1);
        // y should be removed since it's never used
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_dce_keeps_used_vars() {
        let mut stmts = vec![
            Stmt::Assign { dest: "x".into(), value: Value::Int(10) },
            Stmt::Return(Value::Var("x".into())),
        ];
        let stats = dead_code_elimination(&mut stmts);
        assert_eq!(stats.removed, 0);
        assert_eq!(stmts.len(), 2);
    }

    #[test]
    fn test_cse_basic() {
        let mut stmts = vec![
            Stmt::BinOp {
                dest: "t1".into(),
                op: BinOp::Add,
                left: Value::Var("a".into()),
                right: Value::Var("b".into()),
            },
            Stmt::BinOp {
                dest: "t2".into(),
                op: BinOp::Add,
                left: Value::Var("a".into()),
                right: Value::Var("b".into()),
            },
        ];
        let stats = common_subexpression_elimination(&mut stmts);
        assert_eq!(stats.modified, 1);
        // t2 should become a copy of t1
        assert_eq!(stmts[1], Stmt::Copy { dest: "t2".into(), source: "t1".into() });
    }

    #[test]
    fn test_cse_invalidated_by_redefinition() {
        let mut stmts = vec![
            Stmt::BinOp {
                dest: "t1".into(),
                op: BinOp::Add,
                left: Value::Var("a".into()),
                right: Value::Var("b".into()),
            },
            Stmt::Assign { dest: "t1".into(), value: Value::Int(0) }, // kills t1
            Stmt::BinOp {
                dest: "t2".into(),
                op: BinOp::Add,
                left: Value::Var("a".into()),
                right: Value::Var("b".into()),
            },
        ];
        let stats = common_subexpression_elimination(&mut stmts);
        assert_eq!(stats.modified, 0); // t1 was killed, so no CSE
    }

    #[test]
    fn test_loop_detection() {
        let stmts = vec![
            Stmt::Label("loop".into()),
            Stmt::Assign { dest: "x".into(), value: Value::Int(1) },
            Stmt::Jump("loop".into()),
        ];
        let loops = find_loops(&stmts);
        assert_eq!(loops.len(), 1);
        assert_eq!(loops[0], (0, 2));
    }

    #[test]
    fn test_pass_manager_o0() {
        let mut stmts = vec![
            Stmt::Assign { dest: "x".into(), value: Value::Int(10) },
        ];
        let mut pm = PassManager::for_level(OptLevel::O0);
        pm.run(&mut stmts);
        assert!(pm.results().is_empty());
    }

    #[test]
    fn test_pass_manager_o1() {
        let mut stmts = vec![
            Stmt::Assign { dest: "x".into(), value: Value::Int(10) },
            Stmt::Assign { dest: "y".into(), value: Value::Int(20) },
            Stmt::BinOp {
                dest: "z".into(),
                op: BinOp::Add,
                left: Value::Var("x".into()),
                right: Value::Int(5),
            },
            Stmt::Return(Value::Var("z".into())),
        ];
        let mut pm = PassManager::for_level(OptLevel::O1);
        pm.run(&mut stmts);
        assert!(!pm.results().is_empty());
    }

    #[test]
    fn test_pass_manager_fixed_point() {
        let mut stmts = vec![
            Stmt::Assign { dest: "x".into(), value: Value::Int(10) },
            Stmt::BinOp {
                dest: "y".into(),
                op: BinOp::Add,
                left: Value::Var("x".into()),
                right: Value::Int(5),
            },
            Stmt::Return(Value::Var("y".into())),
        ];
        let mut pm = PassManager::for_level(OptLevel::O1);
        let iters = pm.run_to_fixed_point(&mut stmts);
        assert!(iters >= 1);
    }

    #[test]
    fn test_stmt_defined_var() {
        let stmt = Stmt::BinOp {
            dest: "x".into(),
            op: BinOp::Add,
            left: Value::Int(1),
            right: Value::Int(2),
        };
        assert_eq!(stmt.defined_var(), Some("x"));
    }

    #[test]
    fn test_stmt_used_vars() {
        let stmt = Stmt::BinOp {
            dest: "z".into(),
            op: BinOp::Add,
            left: Value::Var("x".into()),
            right: Value::Var("y".into()),
        };
        let used = stmt.used_vars();
        assert_eq!(used.len(), 2);
        assert!(used.contains(&"x".to_string()));
        assert!(used.contains(&"y".to_string()));
    }

    #[test]
    fn test_opt_level_display() {
        assert_eq!(format!("{}", OptLevel::O0), "O0");
        assert_eq!(format!("{}", OptLevel::O1), "O1");
        assert_eq!(format!("{}", OptLevel::O2), "O2");
    }

    #[test]
    fn test_pass_kind_display() {
        assert_eq!(format!("{}", PassKind::ConstantPropagation), "ConstProp");
        assert_eq!(format!("{}", PassKind::DeadCodeElimination), "DCE");
    }

    #[test]
    fn test_boolean_constant_fold() {
        let stmt = Stmt::BinOp {
            dest: "r".into(),
            op: BinOp::And,
            left: Value::Bool(true),
            right: Value::Bool(false),
        };
        let folded = constant_fold(&stmt);
        assert_eq!(folded, Stmt::Assign { dest: "r".into(), value: Value::Bool(false) });
    }
}
