//! JIT compilation concepts — trace recording, hot loop detection, trace
//! optimization (constant folding, dead store elimination), guard insertion,
//! deoptimization triggers, compilation tiers (interpreter/baseline/optimized).

use std::collections::HashMap;
use std::fmt;

// ── Compilation Tiers ──────────────────────────────────────────────────────

/// Compilation tier for a function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    Interpreter,
    Baseline,
    Optimized,
}

impl fmt::Display for Tier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Interpreter => write!(f, "interpreter"),
            Self::Baseline => write!(f, "baseline"),
            Self::Optimized => write!(f, "optimized"),
        }
    }
}

// ── Trace Operations ───────────────────────────────────────────────────────

/// A single operation in a trace.
#[derive(Debug, Clone, PartialEq)]
pub enum TraceOp {
    /// Load a constant i64 value.
    Const(i64),
    /// Load from a virtual register.
    LoadReg(u32),
    /// Store to a virtual register.
    StoreReg(u32),
    /// Add top two values.
    Add,
    /// Subtract top two values.
    Sub,
    /// Multiply top two values.
    Mul,
    /// Signed divide.
    Div,
    /// Negate.
    Neg,
    /// Compare equal.
    Eq,
    /// Compare less-than.
    Lt,
    /// A guard that checks a condition. If it fails, deoptimize.
    Guard { expected: bool, deopt_id: u32 },
    /// Jump to a trace offset.
    Jump(usize),
    /// Loop back to start of trace.
    LoopBack,
    /// Return a value.
    Return,
    /// A no-op (used after dead store elimination).
    Nop,
    /// Call a function by id.
    Call(u32),
    /// Side effect (e.g., memory write). Cannot be eliminated.
    SideEffect(String),
}

impl fmt::Display for TraceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Const(v) => write!(f, "const {v}"),
            Self::LoadReg(r) => write!(f, "load r{r}"),
            Self::StoreReg(r) => write!(f, "store r{r}"),
            Self::Add => write!(f, "add"),
            Self::Sub => write!(f, "sub"),
            Self::Mul => write!(f, "mul"),
            Self::Div => write!(f, "div"),
            Self::Neg => write!(f, "neg"),
            Self::Eq => write!(f, "eq"),
            Self::Lt => write!(f, "lt"),
            Self::Guard { expected, deopt_id } => {
                write!(f, "guard({expected}) deopt={deopt_id}")
            }
            Self::Jump(off) => write!(f, "jmp {off}"),
            Self::LoopBack => write!(f, "loopback"),
            Self::Return => write!(f, "ret"),
            Self::Nop => write!(f, "nop"),
            Self::Call(id) => write!(f, "call {id}"),
            Self::SideEffect(desc) => write!(f, "effect({desc})"),
        }
    }
}

// ── Trace ──────────────────────────────────────────────────────────────────

/// A recorded execution trace.
#[derive(Debug, Clone)]
pub struct Trace {
    pub id: u32,
    pub ops: Vec<TraceOp>,
    pub loop_header: Option<usize>,
    pub guard_count: u32,
    pub origin_function: u32,
}

impl Trace {
    pub fn new(id: u32, origin_function: u32) -> Self {
        Self {
            id,
            ops: Vec::new(),
            loop_header: None,
            guard_count: 0,
            origin_function,
        }
    }

    /// Append an operation.
    pub fn push(&mut self, op: TraceOp) {
        if matches!(op, TraceOp::Guard { .. }) {
            self.guard_count += 1;
        }
        self.ops.push(op);
    }

    /// Number of operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// True if trace has no operations.
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Mark this trace as a loop trace.
    pub fn set_loop_header(&mut self, offset: usize) {
        self.loop_header = Some(offset);
    }

    /// True if this is a loop trace.
    pub fn is_loop_trace(&self) -> bool {
        self.loop_header.is_some()
    }

    /// Count non-Nop operations (the "live" ops).
    pub fn live_op_count(&self) -> usize {
        self.ops.iter().filter(|op| !matches!(op, TraceOp::Nop)).count()
    }
}

// ── Deoptimization Info ────────────────────────────────────────────────────

/// Information needed to deoptimize (bail out) from compiled code.
#[derive(Debug, Clone)]
pub struct DeoptInfo {
    pub deopt_id: u32,
    pub reason: String,
    /// The bytecode PC to resume at in the interpreter.
    pub resume_pc: usize,
    /// Register snapshot to restore.
    pub register_snapshot: Vec<(u32, i64)>,
}

// ── Hot Loop Detection ─────────────────────────────────────────────────────

/// Tracks loop execution counts to detect hot loops.
pub struct HotLoopDetector {
    /// Map from loop header PC → execution count.
    counts: HashMap<usize, u64>,
    /// Threshold to consider a loop "hot."
    threshold: u64,
}

impl HotLoopDetector {
    pub fn new(threshold: u64) -> Self {
        Self {
            counts: HashMap::new(),
            threshold,
        }
    }

    /// Record one iteration of a loop and return true if it became hot.
    pub fn record(&mut self, loop_pc: usize) -> bool {
        let count = self.counts.entry(loop_pc).or_insert(0);
        *count += 1;
        *count == self.threshold
    }

    /// Check if a loop is hot.
    pub fn is_hot(&self, loop_pc: usize) -> bool {
        self.counts.get(&loop_pc).copied().unwrap_or(0) >= self.threshold
    }

    /// Current count for a loop.
    pub fn count(&self, loop_pc: usize) -> u64 {
        self.counts.get(&loop_pc).copied().unwrap_or(0)
    }

    /// Reset a loop's counter.
    pub fn reset(&mut self, loop_pc: usize) {
        self.counts.remove(&loop_pc);
    }

    /// Reset all counters.
    pub fn reset_all(&mut self) {
        self.counts.clear();
    }

    /// Number of tracked loops.
    pub fn tracked_count(&self) -> usize {
        self.counts.len()
    }
}

// ── Trace Optimizer ────────────────────────────────────────────────────────

/// Optimizes a trace in place.
pub struct TraceOptimizer;

impl TraceOptimizer {
    /// Constant folding: evaluate sequences of `Const, Const, Add/Sub/Mul/Div`
    /// into a single `Const`. Removes Nops between passes to enable chained folds.
    pub fn constant_fold(trace: &mut Trace) {
        loop {
            trace.ops.retain(|op| !matches!(op, TraceOp::Nop));
            let mut changed = false;
            let mut i = 0;
            while i + 2 < trace.ops.len() {
                let a_val = match &trace.ops[i] {
                    TraceOp::Const(v) => Some(*v),
                    _ => None,
                };
                let b_val = match &trace.ops[i + 1] {
                    TraceOp::Const(v) => Some(*v),
                    _ => None,
                };
                if let (Some(a), Some(b)) = (a_val, b_val) {
                    let result = match &trace.ops[i + 2] {
                        TraceOp::Add => Some(a.wrapping_add(b)),
                        TraceOp::Sub => Some(a.wrapping_sub(b)),
                        TraceOp::Mul => Some(a.wrapping_mul(b)),
                        TraceOp::Div if b != 0 => Some(a.wrapping_div(b)),
                        _ => None,
                    };
                    if let Some(val) = result {
                        trace.ops[i] = TraceOp::Const(val);
                        trace.ops[i + 1] = TraceOp::Nop;
                        trace.ops[i + 2] = TraceOp::Nop;
                        changed = true;
                        // Don't advance i — the new Const might participate in another fold.
                        continue;
                    }
                }
                i += 1;
            }
            if !changed {
                break;
            }
        }
    }

    /// Dead store elimination: remove `StoreReg(r)` if the register is
    /// overwritten before it is read.
    pub fn dead_store_elimination(trace: &mut Trace) {
        let len = trace.ops.len();
        for i in 0..len {
            let reg = match &trace.ops[i] {
                TraceOp::StoreReg(r) => *r,
                _ => continue,
            };
            // Scan forward for the next use of this register.
            let mut is_dead = false;
            for j in (i + 1)..len {
                match &trace.ops[j] {
                    TraceOp::LoadReg(r) if *r == reg => break,                 // read — not dead
                    TraceOp::StoreReg(r) if *r == reg => { is_dead = true; break; } // overwritten
                    TraceOp::SideEffect(_) | TraceOp::Call(_) => break,        // might read
                    TraceOp::Return | TraceOp::LoopBack | TraceOp::Jump(_) => break,
                    _ => {}
                }
            }
            if is_dead {
                trace.ops[i] = TraceOp::Nop;
            }
        }
    }

    /// Remove all Nop instructions (compact the trace).
    pub fn remove_nops(trace: &mut Trace) {
        trace.ops.retain(|op| !matches!(op, TraceOp::Nop));
    }

    /// Run all optimization passes.
    pub fn optimize(trace: &mut Trace) {
        Self::constant_fold(trace);
        Self::dead_store_elimination(trace);
        Self::remove_nops(trace);
    }
}

// ── Compilation Pipeline ───────────────────────────────────────────────────

/// The state of a function in the JIT pipeline.
#[derive(Debug, Clone)]
pub struct FunctionState {
    pub func_id: u32,
    pub name: String,
    pub tier: Tier,
    pub execution_count: u64,
    pub trace: Option<Trace>,
    pub deopt_count: u32,
    pub deopt_infos: Vec<DeoptInfo>,
}

impl FunctionState {
    pub fn new(func_id: u32, name: impl Into<String>) -> Self {
        Self {
            func_id,
            name: name.into(),
            tier: Tier::Interpreter,
            execution_count: 0,
            trace: None,
            deopt_count: 0,
            deopt_infos: Vec::new(),
        }
    }
}

/// JIT compilation statistics.
#[derive(Debug, Clone, Default)]
pub struct JitStats {
    pub traces_recorded: u64,
    pub traces_compiled: u64,
    pub baseline_compilations: u64,
    pub optimized_compilations: u64,
    pub deoptimizations: u64,
    pub guards_inserted: u64,
    pub constant_folds: u64,
    pub dead_stores_eliminated: u64,
}

/// The JIT compilation pipeline.
pub struct JitPipeline {
    functions: HashMap<u32, FunctionState>,
    hot_detector: HotLoopDetector,
    /// Execution count threshold for baseline tier.
    baseline_threshold: u64,
    /// Execution count threshold for optimized tier.
    optimized_threshold: u64,
    /// Maximum deoptimizations before demotion.
    max_deopts: u32,
    stats: JitStats,
    next_trace_id: u32,
    next_deopt_id: u32,
}

impl JitPipeline {
    pub fn new(
        baseline_threshold: u64,
        optimized_threshold: u64,
        hot_loop_threshold: u64,
    ) -> Self {
        Self {
            functions: HashMap::new(),
            hot_detector: HotLoopDetector::new(hot_loop_threshold),
            baseline_threshold,
            optimized_threshold,
            max_deopts: 10,
            stats: JitStats::default(),
            next_trace_id: 0,
            next_deopt_id: 0,
        }
    }

    pub fn stats(&self) -> &JitStats {
        &self.stats
    }

    /// Register a function.
    pub fn register_function(&mut self, func_id: u32, name: impl Into<String>) {
        self.functions
            .entry(func_id)
            .or_insert_with(|| FunctionState::new(func_id, name));
    }

    /// Get a function's current tier.
    pub fn tier(&self, func_id: u32) -> Option<Tier> {
        self.functions.get(&func_id).map(|f| f.tier)
    }

    /// Record a function execution and potentially promote its tier.
    pub fn record_execution(&mut self, func_id: u32) -> Option<Tier> {
        let fs = self.functions.get_mut(&func_id)?;
        fs.execution_count += 1;
        let count = fs.execution_count;
        let current_tier = fs.tier;

        if current_tier == Tier::Interpreter && count >= self.baseline_threshold {
            fs.tier = Tier::Baseline;
            self.stats.baseline_compilations += 1;
            return Some(Tier::Baseline);
        }

        if current_tier == Tier::Baseline && count >= self.optimized_threshold {
            fs.tier = Tier::Optimized;
            self.stats.optimized_compilations += 1;
            return Some(Tier::Optimized);
        }

        None
    }

    /// Record a loop iteration.
    pub fn record_loop(&mut self, loop_pc: usize) -> bool {
        self.hot_detector.record(loop_pc)
    }

    /// Start recording a trace for a function.
    pub fn start_trace(&mut self, func_id: u32) -> Option<u32> {
        if !self.functions.contains_key(&func_id) {
            return None;
        }
        let trace_id = self.next_trace_id;
        self.next_trace_id += 1;
        let trace = Trace::new(trace_id, func_id);
        let fs = self.functions.get_mut(&func_id).unwrap();
        fs.trace = Some(trace);
        self.stats.traces_recorded += 1;
        Some(trace_id)
    }

    /// Append an operation to the active trace.
    pub fn trace_op(&mut self, func_id: u32, op: TraceOp) -> bool {
        if let Some(fs) = self.functions.get_mut(&func_id) {
            if let Some(trace) = &mut fs.trace {
                if matches!(op, TraceOp::Guard { .. }) {
                    self.stats.guards_inserted += 1;
                }
                trace.push(op);
                return true;
            }
        }
        false
    }

    /// Finalize a trace, apply optimizations, and "compile" it.
    pub fn compile_trace(&mut self, func_id: u32) -> Option<usize> {
        let fs = self.functions.get_mut(&func_id)?;
        let trace = fs.trace.as_mut()?;

        let before = trace.len();
        TraceOptimizer::optimize(trace);
        let after = trace.len();

        self.stats.constant_folds += (before - after) as u64;
        self.stats.traces_compiled += 1;

        Some(after)
    }

    /// Trigger a deoptimization: demote the function if too many deopts.
    pub fn deoptimize(
        &mut self,
        func_id: u32,
        reason: impl Into<String>,
        resume_pc: usize,
    ) -> Option<Tier> {
        let deopt_id = self.next_deopt_id;
        self.next_deopt_id += 1;
        self.stats.deoptimizations += 1;

        let fs = self.functions.get_mut(&func_id)?;
        fs.deopt_count += 1;
        fs.deopt_infos.push(DeoptInfo {
            deopt_id,
            reason: reason.into(),
            resume_pc,
            register_snapshot: Vec::new(),
        });

        if fs.deopt_count >= self.max_deopts {
            // Demote back one tier.
            let new_tier = match fs.tier {
                Tier::Optimized => Tier::Baseline,
                Tier::Baseline => Tier::Interpreter,
                Tier::Interpreter => Tier::Interpreter,
            };
            fs.tier = new_tier;
            fs.deopt_count = 0;
            fs.trace = None;
            Some(new_tier)
        } else {
            None
        }
    }

    /// Get function state.
    pub fn function_state(&self, func_id: u32) -> Option<&FunctionState> {
        self.functions.get(&func_id)
    }

    /// Number of registered functions.
    pub fn function_count(&self) -> usize {
        self.functions.len()
    }

    /// Get the hot loop detector.
    pub fn hot_detector(&self) -> &HotLoopDetector {
        &self.hot_detector
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ordering() {
        assert!(Tier::Interpreter < Tier::Baseline);
        assert!(Tier::Baseline < Tier::Optimized);
    }

    #[test]
    fn hot_loop_detection() {
        let mut det = HotLoopDetector::new(5);
        for _ in 0..4 {
            assert!(!det.record(100));
        }
        assert!(det.record(100));
        assert!(det.is_hot(100));
        assert_eq!(det.count(100), 5);
    }

    #[test]
    fn hot_loop_reset() {
        let mut det = HotLoopDetector::new(3);
        for _ in 0..3 {
            det.record(10);
        }
        assert!(det.is_hot(10));
        det.reset(10);
        assert!(!det.is_hot(10));
    }

    #[test]
    fn constant_folding() {
        let mut trace = Trace::new(0, 0);
        trace.push(TraceOp::Const(3));
        trace.push(TraceOp::Const(4));
        trace.push(TraceOp::Add);
        trace.push(TraceOp::Return);

        TraceOptimizer::constant_fold(&mut trace);
        TraceOptimizer::remove_nops(&mut trace);

        assert_eq!(trace.ops.len(), 2);
        assert_eq!(trace.ops[0], TraceOp::Const(7));
        assert_eq!(trace.ops[1], TraceOp::Return);
    }

    #[test]
    fn constant_folding_chain() {
        // (2 + 3) * 4 = 20
        let mut trace = Trace::new(0, 0);
        trace.push(TraceOp::Const(2));
        trace.push(TraceOp::Const(3));
        trace.push(TraceOp::Add);
        trace.push(TraceOp::Const(4));
        trace.push(TraceOp::Mul);
        trace.push(TraceOp::Return);

        TraceOptimizer::constant_fold(&mut trace);
        TraceOptimizer::remove_nops(&mut trace);

        // After first fold: Const(5), Const(4), Mul, Return
        // After second fold: Const(20), Return
        assert_eq!(trace.ops[0], TraceOp::Const(20));
    }

    #[test]
    fn dead_store_elimination() {
        let mut trace = Trace::new(0, 0);
        trace.push(TraceOp::Const(1));
        trace.push(TraceOp::StoreReg(0)); // dead: overwritten before read
        trace.push(TraceOp::Const(2));
        trace.push(TraceOp::StoreReg(0)); // live
        trace.push(TraceOp::LoadReg(0));
        trace.push(TraceOp::Return);

        TraceOptimizer::dead_store_elimination(&mut trace);
        TraceOptimizer::remove_nops(&mut trace);

        // The first StoreReg(0) should be removed.
        let store_count = trace
            .ops
            .iter()
            .filter(|op| matches!(op, TraceOp::StoreReg(0)))
            .count();
        assert_eq!(store_count, 1);
    }

    #[test]
    fn optimize_combined() {
        let mut trace = Trace::new(0, 0);
        trace.push(TraceOp::Const(10));
        trace.push(TraceOp::Const(20));
        trace.push(TraceOp::Add);
        trace.push(TraceOp::StoreReg(0));
        trace.push(TraceOp::Const(99));
        trace.push(TraceOp::StoreReg(0)); // overwrite
        trace.push(TraceOp::LoadReg(0));
        trace.push(TraceOp::Return);

        TraceOptimizer::optimize(&mut trace);

        // Const fold should reduce 10+20 to 30.
        // DSE should eliminate the first StoreReg(0).
        assert!(trace.ops.contains(&TraceOp::Const(30)));
        let store_count = trace
            .ops
            .iter()
            .filter(|op| matches!(op, TraceOp::StoreReg(0)))
            .count();
        assert_eq!(store_count, 1);
    }

    #[test]
    fn pipeline_tier_promotion() {
        let mut jit = JitPipeline::new(5, 20, 100);
        jit.register_function(0, "f");

        for _ in 0..4 {
            assert!(jit.record_execution(0).is_none());
        }
        assert_eq!(jit.record_execution(0), Some(Tier::Baseline));
        assert_eq!(jit.tier(0), Some(Tier::Baseline));

        for _ in 0..14 {
            jit.record_execution(0);
        }
        assert_eq!(jit.record_execution(0), Some(Tier::Optimized));
    }

    #[test]
    fn trace_recording_and_compile() {
        let mut jit = JitPipeline::new(5, 20, 100);
        jit.register_function(0, "f");
        let tid = jit.start_trace(0).unwrap();
        assert_eq!(tid, 0);

        jit.trace_op(0, TraceOp::Const(1));
        jit.trace_op(0, TraceOp::Const(2));
        jit.trace_op(0, TraceOp::Add);
        jit.trace_op(0, TraceOp::Return);

        let compiled_len = jit.compile_trace(0).unwrap();
        // After constant folding: Const(3) + Return = 2 ops.
        assert_eq!(compiled_len, 2);
        assert_eq!(jit.stats().traces_compiled, 1);
    }

    #[test]
    fn deoptimization() {
        let mut jit = JitPipeline::new(2, 10, 100);
        jit.register_function(0, "f");
        // Promote to baseline.
        jit.record_execution(0);
        jit.record_execution(0);
        assert_eq!(jit.tier(0), Some(Tier::Baseline));

        // Deoptimize many times.
        for i in 0..10 {
            jit.deoptimize(0, format!("guard {i}"), 0);
        }
        // Should have been demoted.
        assert_eq!(jit.tier(0), Some(Tier::Interpreter));
    }

    #[test]
    fn guard_tracking() {
        let mut jit = JitPipeline::new(5, 20, 100);
        jit.register_function(0, "f");
        jit.start_trace(0);
        jit.trace_op(0, TraceOp::Guard {
            expected: true,
            deopt_id: 0,
        });
        jit.trace_op(0, TraceOp::Guard {
            expected: false,
            deopt_id: 1,
        });
        assert_eq!(jit.stats().guards_inserted, 2);
        let fs = jit.function_state(0).unwrap();
        assert_eq!(fs.trace.as_ref().unwrap().guard_count, 2);
    }

    #[test]
    fn trace_is_loop() {
        let mut trace = Trace::new(0, 0);
        assert!(!trace.is_loop_trace());
        trace.set_loop_header(10);
        assert!(trace.is_loop_trace());
    }

    #[test]
    fn trace_live_ops() {
        let mut trace = Trace::new(0, 0);
        trace.push(TraceOp::Const(1));
        trace.push(TraceOp::Nop);
        trace.push(TraceOp::Return);
        assert_eq!(trace.live_op_count(), 2);
    }

    #[test]
    fn jit_stats_accumulate() {
        let mut jit = JitPipeline::new(1, 5, 100);
        jit.register_function(0, "a");
        jit.register_function(1, "b");
        jit.record_execution(0);
        jit.record_execution(1);
        assert_eq!(jit.stats().baseline_compilations, 2);
    }

    #[test]
    fn trace_op_display() {
        assert_eq!(TraceOp::Const(42).to_string(), "const 42");
        assert_eq!(TraceOp::Add.to_string(), "add");
        assert_eq!(TraceOp::LoadReg(3).to_string(), "load r3");
    }

    #[test]
    fn div_by_zero_not_folded() {
        let mut trace = Trace::new(0, 0);
        trace.push(TraceOp::Const(10));
        trace.push(TraceOp::Const(0));
        trace.push(TraceOp::Div);

        TraceOptimizer::constant_fold(&mut trace);
        // Should not fold — div by zero.
        assert_eq!(trace.ops[2], TraceOp::Div);
    }

    #[test]
    fn hot_detector_tracked_count() {
        let mut det = HotLoopDetector::new(5);
        det.record(1);
        det.record(2);
        det.record(3);
        assert_eq!(det.tracked_count(), 3);
        det.reset_all();
        assert_eq!(det.tracked_count(), 0);
    }

    #[test]
    fn function_not_registered() {
        let mut jit = JitPipeline::new(5, 20, 100);
        assert!(jit.start_trace(99).is_none());
        assert!(jit.record_execution(99).is_none());
    }

    #[test]
    fn side_effect_blocks_dse() {
        let mut trace = Trace::new(0, 0);
        trace.push(TraceOp::Const(1));
        trace.push(TraceOp::StoreReg(0));
        trace.push(TraceOp::SideEffect("write".to_string()));
        trace.push(TraceOp::Const(2));
        trace.push(TraceOp::StoreReg(0));
        trace.push(TraceOp::LoadReg(0));
        trace.push(TraceOp::Return);

        TraceOptimizer::dead_store_elimination(&mut trace);
        // The side effect should block DSE.
        let store_count = trace
            .ops
            .iter()
            .filter(|op| matches!(op, TraceOp::StoreReg(0)))
            .count();
        assert_eq!(store_count, 2);
    }
}
