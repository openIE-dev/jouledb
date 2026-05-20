//! WASM interpreter (simplified) — stack machine execution, basic instructions
//! (i32 add/sub/mul/div, local.get/set, call, br, block, loop, if/else,
//! return), value stack, call stack, trap handling, instruction dispatch.

use std::fmt;

// ── Values ─────────────────────────────────────────────────────────────────

/// A WASM runtime value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WasmValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl WasmValue {
    /// Extract an i32, or trap.
    pub fn as_i32(&self) -> Result<i32, Trap> {
        match self {
            Self::I32(v) => Ok(*v),
            _ => Err(Trap::TypeMismatch {
                expected: "i32".to_string(),
                got: self.type_name().to_string(),
            }),
        }
    }

    /// Extract an i64, or trap.
    pub fn as_i64(&self) -> Result<i64, Trap> {
        match self {
            Self::I64(v) => Ok(*v),
            _ => Err(Trap::TypeMismatch {
                expected: "i64".to_string(),
                got: self.type_name().to_string(),
            }),
        }
    }

    /// Extract an f32, or trap.
    pub fn as_f32(&self) -> Result<f32, Trap> {
        match self {
            Self::F32(v) => Ok(*v),
            _ => Err(Trap::TypeMismatch {
                expected: "f32".to_string(),
                got: self.type_name().to_string(),
            }),
        }
    }

    /// Extract an f64, or trap.
    pub fn as_f64(&self) -> Result<f64, Trap> {
        match self {
            Self::F64(v) => Ok(*v),
            _ => Err(Trap::TypeMismatch {
                expected: "f64".to_string(),
                got: self.type_name().to_string(),
            }),
        }
    }

    /// Runtime type name.
    pub fn type_name(&self) -> &'static str {
        match self {
            Self::I32(_) => "i32",
            Self::I64(_) => "i64",
            Self::F32(_) => "f32",
            Self::F64(_) => "f64",
        }
    }
}

impl fmt::Display for WasmValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::I32(v) => write!(f, "{v}:i32"),
            Self::I64(v) => write!(f, "{v}:i64"),
            Self::F32(v) => write!(f, "{v}:f32"),
            Self::F64(v) => write!(f, "{v}:f64"),
        }
    }
}

// ── Instructions ───────────────────────────────────────────────────────────

/// A simplified WASM instruction set.
#[derive(Debug, Clone, PartialEq)]
pub enum Instruction {
    // Constants
    I32Const(i32),
    I64Const(i64),
    F32Const(f32),
    F64Const(f64),

    // i32 arithmetic
    I32Add,
    I32Sub,
    I32Mul,
    I32DivS,
    I32RemS,
    I32And,
    I32Or,
    I32Xor,
    I32Eqz,
    I32Eq,
    I32LtS,
    I32GtS,

    // Locals
    LocalGet(u32),
    LocalSet(u32),
    LocalTee(u32),

    // Control flow
    /// Block with label arity (number of result values).
    Block(u32),
    /// Loop with label arity.
    Loop(u32),
    /// If (with label arity). Branches to else/end on false.
    If(u32),
    Else,
    End,
    /// Branch to label depth.
    Br(u32),
    /// Branch if top of stack is nonzero.
    BrIf(u32),
    /// Return from current function.
    Return,

    // Calls
    /// Call function by index.
    Call(u32),

    // Stack
    Drop,
    Select,

    // No-op
    Nop,
    Unreachable,
}

// ── Traps ──────────────────────────────────────────────────────────────────

/// Runtime errors (traps).
#[derive(Debug, Clone, PartialEq)]
pub enum Trap {
    StackUnderflow,
    StackOverflow { limit: usize },
    TypeMismatch { expected: String, got: String },
    DivisionByZero,
    IntegerOverflow,
    UnreachableExecuted,
    UndefinedLocal(u32),
    UndefinedFunction(u32),
    CallStackOverflow { limit: usize },
    LabelStackUnderflow,
    InstructionLimitExceeded { limit: u64 },
    InvalidBranchDepth { depth: u32, max: u32 },
}

impl fmt::Display for Trap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StackUnderflow => write!(f, "stack underflow"),
            Self::StackOverflow { limit } => write!(f, "stack overflow (limit {limit})"),
            Self::TypeMismatch { expected, got } => {
                write!(f, "type mismatch: expected {expected}, got {got}")
            }
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::IntegerOverflow => write!(f, "integer overflow"),
            Self::UnreachableExecuted => write!(f, "unreachable executed"),
            Self::UndefinedLocal(idx) => write!(f, "undefined local {idx}"),
            Self::UndefinedFunction(idx) => write!(f, "undefined function {idx}"),
            Self::CallStackOverflow { limit } => {
                write!(f, "call stack overflow (limit {limit})")
            }
            Self::LabelStackUnderflow => write!(f, "label stack underflow"),
            Self::InstructionLimitExceeded { limit } => {
                write!(f, "instruction limit exceeded ({limit})")
            }
            Self::InvalidBranchDepth { depth, max } => {
                write!(f, "invalid branch depth {depth} (max {max})")
            }
        }
    }
}

// ── Label ──────────────────────────────────────────────────────────────────

/// A control flow label on the label stack.
#[derive(Debug, Clone)]
struct Label {
    /// Program counter to branch to (loop goes back, block goes forward).
    branch_target: usize,
    /// Value stack height when label was entered.
    stack_height: usize,
    /// Number of result values this label produces.
    arity: u32,
    /// Is this a loop label? (affects branch target semantics.)
    is_loop: bool,
}

// ── Call Frame ──────────────────────────────────────────────────────────────

/// A call frame on the call stack.
#[derive(Debug, Clone)]
struct CallFrame {
    /// Return address (instruction index in the *caller's* function).
    return_pc: usize,
    /// Function index of the caller.
    return_func: u32,
    /// Locals for this frame (params + locals).
    locals: Vec<WasmValue>,
    /// Label stack for this frame.
    labels: Vec<Label>,
    /// Value stack height on entry.
    stack_height: usize,
    /// Number of result values.
    result_arity: u32,
}

// ── Function Definition ────────────────────────────────────────────────────

/// A function definition in the interpreter.
#[derive(Debug, Clone)]
pub struct FuncDef {
    /// Number of parameters.
    pub param_count: u32,
    /// Number of locals (excluding parameters).
    pub local_count: u32,
    /// Number of result values.
    pub result_count: u32,
    /// Instructions.
    pub body: Vec<Instruction>,
}

impl FuncDef {
    pub fn new(
        param_count: u32,
        local_count: u32,
        result_count: u32,
        body: Vec<Instruction>,
    ) -> Self {
        Self {
            param_count,
            local_count,
            result_count,
            body,
        }
    }

    /// Total number of local slots (params + locals).
    pub fn total_locals(&self) -> u32 {
        self.param_count + self.local_count
    }
}

// ── Interpreter ────────────────────────────────────────────────────────────

/// Configuration for the interpreter.
#[derive(Debug, Clone)]
pub struct InterpreterConfig {
    /// Maximum value stack depth.
    pub max_stack_depth: usize,
    /// Maximum call stack depth.
    pub max_call_depth: usize,
    /// Maximum number of instructions to execute (0 = unlimited).
    pub max_instructions: u64,
}

impl Default for InterpreterConfig {
    fn default() -> Self {
        Self {
            max_stack_depth: 1024,
            max_call_depth: 256,
            max_instructions: 1_000_000,
        }
    }
}

/// Execution statistics.
#[derive(Debug, Clone, Default)]
pub struct ExecStats {
    pub instructions_executed: u64,
    pub max_stack_depth_reached: usize,
    pub max_call_depth_reached: usize,
    pub calls_made: u64,
}

/// The WASM interpreter.
pub struct Interpreter {
    /// Functions available to call.
    functions: Vec<FuncDef>,
    /// Value stack.
    value_stack: Vec<WasmValue>,
    /// Call stack.
    call_stack: Vec<CallFrame>,
    /// Currently executing function index.
    current_func: u32,
    /// Program counter within current function.
    pc: usize,
    /// Execution config.
    config: InterpreterConfig,
    /// Statistics.
    stats: ExecStats,
}

impl Interpreter {
    /// Create an interpreter with the given functions and config.
    pub fn new(functions: Vec<FuncDef>, config: InterpreterConfig) -> Self {
        Self {
            functions,
            value_stack: Vec::new(),
            call_stack: Vec::new(),
            current_func: 0,
            pc: 0,
            config,
            stats: ExecStats::default(),
        }
    }

    /// Get execution statistics.
    pub fn stats(&self) -> &ExecStats {
        &self.stats
    }

    /// Current value stack depth.
    pub fn stack_depth(&self) -> usize {
        self.value_stack.len()
    }

    /// Push a value onto the stack.
    fn push(&mut self, val: WasmValue) -> Result<(), Trap> {
        if self.value_stack.len() >= self.config.max_stack_depth {
            return Err(Trap::StackOverflow {
                limit: self.config.max_stack_depth,
            });
        }
        self.value_stack.push(val);
        if self.value_stack.len() > self.stats.max_stack_depth_reached {
            self.stats.max_stack_depth_reached = self.value_stack.len();
        }
        Ok(())
    }

    /// Pop a value from the stack.
    fn pop(&mut self) -> Result<WasmValue, Trap> {
        self.value_stack.pop().ok_or(Trap::StackUnderflow)
    }

    /// Peek at the top of the stack.
    fn peek(&self) -> Result<&WasmValue, Trap> {
        self.value_stack.last().ok_or(Trap::StackUnderflow)
    }

    /// Get the current labels stack (from the top call frame, or an empty one).
    fn current_labels(&self) -> &[Label] {
        if let Some(frame) = self.call_stack.last() {
            &frame.labels
        } else {
            &[]
        }
    }

    /// Invoke a function by index with the given arguments.
    /// Returns the result values.
    pub fn invoke(&mut self, func_idx: u32, args: &[WasmValue]) -> Result<Vec<WasmValue>, Trap> {
        self.value_stack.clear();
        self.call_stack.clear();
        self.stats = ExecStats::default();

        let func = self
            .functions
            .get(func_idx as usize)
            .ok_or(Trap::UndefinedFunction(func_idx))?;

        if args.len() != func.param_count as usize {
            return Err(Trap::TypeMismatch {
                expected: format!("{} args", func.param_count),
                got: format!("{} args", args.len()),
            });
        }

        // Build locals: params first, then zero-initialized locals.
        let mut locals = Vec::with_capacity(func.total_locals() as usize);
        locals.extend_from_slice(args);
        for _ in 0..func.local_count {
            locals.push(WasmValue::I32(0));
        }

        let result_arity = func.result_count;

        // Push the initial frame.
        self.call_stack.push(CallFrame {
            return_pc: 0,
            return_func: func_idx,
            locals,
            labels: Vec::new(),
            stack_height: 0,
            result_arity,
        });
        self.current_func = func_idx;
        self.pc = 0;

        if self.call_stack.len() > self.stats.max_call_depth_reached {
            self.stats.max_call_depth_reached = self.call_stack.len();
        }

        self.run()?;

        // Collect results.
        let mut results = Vec::new();
        for _ in 0..result_arity {
            results.push(self.pop()?);
        }
        results.reverse();
        Ok(results)
    }

    /// Main execution loop.
    fn run(&mut self) -> Result<(), Trap> {
        loop {
            // Check instruction limit.
            if self.config.max_instructions > 0
                && self.stats.instructions_executed >= self.config.max_instructions
            {
                return Err(Trap::InstructionLimitExceeded {
                    limit: self.config.max_instructions,
                });
            }

            let func_idx = self.current_func as usize;
            let func = &self.functions[func_idx];
            if self.pc >= func.body.len() {
                // Implicit return at end of function.
                if self.call_stack.len() <= 1 {
                    // Top-level function done.
                    self.call_stack.pop();
                    return Ok(());
                }
                return self.do_return();
            }

            let instr = func.body[self.pc].clone();
            self.pc += 1;
            self.stats.instructions_executed += 1;

            match instr {
                Instruction::Nop => {}
                Instruction::Unreachable => return Err(Trap::UnreachableExecuted),

                // Constants
                Instruction::I32Const(v) => self.push(WasmValue::I32(v))?,
                Instruction::I64Const(v) => self.push(WasmValue::I64(v))?,
                Instruction::F32Const(v) => self.push(WasmValue::F32(v))?,
                Instruction::F64Const(v) => self.push(WasmValue::F64(v))?,

                // i32 arithmetic
                Instruction::I32Add => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(a.wrapping_add(b)))?;
                }
                Instruction::I32Sub => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(a.wrapping_sub(b)))?;
                }
                Instruction::I32Mul => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(a.wrapping_mul(b)))?;
                }
                Instruction::I32DivS => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    if b == 0 {
                        return Err(Trap::DivisionByZero);
                    }
                    if a == i32::MIN && b == -1 {
                        return Err(Trap::IntegerOverflow);
                    }
                    self.push(WasmValue::I32(a / b))?;
                }
                Instruction::I32RemS => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    if b == 0 {
                        return Err(Trap::DivisionByZero);
                    }
                    self.push(WasmValue::I32(a.wrapping_rem(b)))?;
                }
                Instruction::I32And => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(a & b))?;
                }
                Instruction::I32Or => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(a | b))?;
                }
                Instruction::I32Xor => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(a ^ b))?;
                }
                Instruction::I32Eqz => {
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(if a == 0 { 1 } else { 0 }))?;
                }
                Instruction::I32Eq => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(if a == b { 1 } else { 0 }))?;
                }
                Instruction::I32LtS => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(if a < b { 1 } else { 0 }))?;
                }
                Instruction::I32GtS => {
                    let b = self.pop()?.as_i32()?;
                    let a = self.pop()?.as_i32()?;
                    self.push(WasmValue::I32(if a > b { 1 } else { 0 }))?;
                }

                // Locals
                Instruction::LocalGet(idx) => {
                    let val = self.get_local(idx)?;
                    self.push(val)?;
                }
                Instruction::LocalSet(idx) => {
                    let val = self.pop()?;
                    self.set_local(idx, val)?;
                }
                Instruction::LocalTee(idx) => {
                    let val = *self.peek()?;
                    self.set_local(idx, val)?;
                }

                // Control flow
                Instruction::Block(arity) => {
                    // Push a label; branch target is the matching End.
                    let target = self.find_end(func_idx, self.pc);
                    let frame = self.call_stack.last_mut().unwrap();
                    frame.labels.push(Label {
                        branch_target: target,
                        stack_height: self.value_stack.len(),
                        arity,
                        is_loop: false,
                    });
                }
                Instruction::Loop(arity) => {
                    // Loop label: branch goes back to start of the loop.
                    let target = self.pc - 1; // the Loop instruction itself
                    let frame = self.call_stack.last_mut().unwrap();
                    frame.labels.push(Label {
                        branch_target: target,
                        stack_height: self.value_stack.len(),
                        arity,
                        is_loop: true,
                    });
                }
                Instruction::If(arity) => {
                    let cond = self.pop()?.as_i32()?;
                    if cond != 0 {
                        // Enter if-true block.
                        let target = self.find_end(func_idx, self.pc);
                        let frame = self.call_stack.last_mut().unwrap();
                        frame.labels.push(Label {
                            branch_target: target,
                            stack_height: self.value_stack.len(),
                            arity,
                            is_loop: false,
                        });
                    } else {
                        // Skip to Else or End.
                        let else_or_end = self.find_else_or_end(func_idx, self.pc);
                        self.pc = else_or_end;
                        // If we landed on Else, the next iteration will process what's after it.
                        if self.pc < self.functions[func_idx].body.len()
                            && self.functions[func_idx].body[self.pc] == Instruction::Else
                        {
                            self.pc += 1;
                            let target = self.find_end(func_idx, self.pc);
                            let frame = self.call_stack.last_mut().unwrap();
                            frame.labels.push(Label {
                                branch_target: target,
                                stack_height: self.value_stack.len(),
                                arity,
                                is_loop: false,
                            });
                        }
                    }
                }
                Instruction::Else => {
                    // Reached else from the true branch — skip to End.
                    let frame = self.call_stack.last_mut().unwrap();
                    if let Some(label) = frame.labels.pop() {
                        self.pc = label.branch_target;
                    }
                }
                Instruction::End => {
                    let frame = self.call_stack.last_mut().unwrap();
                    frame.labels.pop();
                }
                Instruction::Br(depth) => {
                    self.do_branch(depth)?;
                }
                Instruction::BrIf(depth) => {
                    let cond = self.pop()?.as_i32()?;
                    if cond != 0 {
                        self.do_branch(depth)?;
                    }
                }
                Instruction::Return => {
                    if self.call_stack.len() <= 1 {
                        self.call_stack.pop();
                        return Ok(());
                    }
                    self.do_return()?;
                }

                // Calls
                Instruction::Call(fidx) => {
                    self.do_call(fidx)?;
                }

                // Stack ops
                Instruction::Drop => {
                    self.pop()?;
                }
                Instruction::Select => {
                    let cond = self.pop()?.as_i32()?;
                    let val2 = self.pop()?;
                    let val1 = self.pop()?;
                    self.push(if cond != 0 { val1 } else { val2 })?;
                }
            }
        }
    }

    /// Get a local variable from the current frame.
    fn get_local(&self, idx: u32) -> Result<WasmValue, Trap> {
        let frame = self.call_stack.last().ok_or(Trap::StackUnderflow)?;
        frame
            .locals
            .get(idx as usize)
            .copied()
            .ok_or(Trap::UndefinedLocal(idx))
    }

    /// Set a local variable in the current frame.
    fn set_local(&mut self, idx: u32, val: WasmValue) -> Result<(), Trap> {
        let frame = self.call_stack.last_mut().ok_or(Trap::StackUnderflow)?;
        let slot = frame
            .locals
            .get_mut(idx as usize)
            .ok_or(Trap::UndefinedLocal(idx))?;
        *slot = val;
        Ok(())
    }

    /// Find the matching End for a Block/If starting at `from_pc`.
    fn find_end(&self, func_idx: usize, from_pc: usize) -> usize {
        let body = &self.functions[func_idx].body;
        let mut depth = 1u32;
        let mut pc = from_pc;
        while pc < body.len() {
            match &body[pc] {
                Instruction::Block(_) | Instruction::Loop(_) | Instruction::If(_) => depth += 1,
                Instruction::End => {
                    depth -= 1;
                    if depth == 0 {
                        return pc;
                    }
                }
                _ => {}
            }
            pc += 1;
        }
        body.len()
    }

    /// Find the matching Else or End for an If starting at `from_pc`.
    fn find_else_or_end(&self, func_idx: usize, from_pc: usize) -> usize {
        let body = &self.functions[func_idx].body;
        let mut depth = 1u32;
        let mut pc = from_pc;
        while pc < body.len() {
            match &body[pc] {
                Instruction::Block(_) | Instruction::Loop(_) | Instruction::If(_) => depth += 1,
                Instruction::Else if depth == 1 => return pc,
                Instruction::End => {
                    depth -= 1;
                    if depth == 0 {
                        return pc;
                    }
                }
                _ => {}
            }
            pc += 1;
        }
        body.len()
    }

    /// Perform a branch to the given label depth.
    fn do_branch(&mut self, depth: u32) -> Result<(), Trap> {
        let frame = self.call_stack.last_mut().ok_or(Trap::LabelStackUnderflow)?;
        let label_count = frame.labels.len() as u32;
        if depth >= label_count {
            return Err(Trap::InvalidBranchDepth {
                depth,
                max: label_count.saturating_sub(1),
            });
        }
        let label_idx = frame.labels.len() - 1 - depth as usize;
        let label = frame.labels[label_idx].clone();

        // Collect result values from top of stack.
        let arity = if label.is_loop { 0 } else { label.arity };
        let mut results = Vec::new();
        for _ in 0..arity {
            results.push(self.pop()?);
        }

        // Unwind the value stack.
        self.value_stack.truncate(label.stack_height);

        // Push results back.
        for val in results.into_iter().rev() {
            self.push(val)?;
        }

        // Remove labels up to and including the target.
        let frame = self.call_stack.last_mut().unwrap();
        frame.labels.truncate(label_idx + if label.is_loop { 1 } else { 0 });

        // Jump.
        if label.is_loop {
            // For loops, branch_target points to the Loop instruction.
            // We re-execute it.
            self.pc = label.branch_target;
        } else {
            // For blocks, branch_target points to the End instruction.
            // Skip past it.
            self.pc = label.branch_target + 1;
        }

        Ok(())
    }

    /// Perform a function call.
    fn do_call(&mut self, func_idx: u32) -> Result<(), Trap> {
        if self.call_stack.len() >= self.config.max_call_depth {
            return Err(Trap::CallStackOverflow {
                limit: self.config.max_call_depth,
            });
        }

        let func = self
            .functions
            .get(func_idx as usize)
            .ok_or(Trap::UndefinedFunction(func_idx))?;

        // Pop arguments from the stack.
        let param_count = func.param_count as usize;
        let local_count = func.local_count;
        let result_count = func.result_count;
        let total_locals = func.total_locals() as usize;

        if self.value_stack.len() < param_count {
            return Err(Trap::StackUnderflow);
        }

        let mut locals = Vec::with_capacity(total_locals);
        // Collect parameters in order: stack has them bottom-to-top.
        let stack_len = self.value_stack.len();
        let start = stack_len - param_count;
        for i in start..stack_len {
            locals.push(self.value_stack[i]);
        }
        self.value_stack.truncate(start);

        // Zero-init remaining locals.
        for _ in 0..local_count {
            locals.push(WasmValue::I32(0));
        }

        // Save return context.
        let frame = CallFrame {
            return_pc: self.pc,
            return_func: self.current_func,
            locals,
            labels: Vec::new(),
            stack_height: self.value_stack.len(),
            result_arity: result_count,
        };
        self.call_stack.push(frame);
        self.current_func = func_idx;
        self.pc = 0;
        self.stats.calls_made += 1;

        if self.call_stack.len() > self.stats.max_call_depth_reached {
            self.stats.max_call_depth_reached = self.call_stack.len();
        }

        Ok(())
    }

    /// Return from the current function.
    fn do_return(&mut self) -> Result<(), Trap> {
        let frame = self.call_stack.pop().ok_or(Trap::StackUnderflow)?;

        // Collect results.
        let mut results = Vec::new();
        for _ in 0..frame.result_arity {
            results.push(self.pop()?);
        }

        // Unwind to the saved stack height.
        self.value_stack.truncate(frame.stack_height);

        // Push results back.
        for val in results.into_iter().rev() {
            self.push(val)?;
        }

        // Restore caller context.
        self.current_func = frame.return_func;
        self.pc = frame.return_pc;

        Ok(())
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> InterpreterConfig {
        InterpreterConfig::default()
    }

    fn make_interp(funcs: Vec<FuncDef>) -> Interpreter {
        Interpreter::new(funcs, default_config())
    }

    #[test]
    fn simple_add() {
        // func(a: i32, b: i32) -> i32 { a + b }
        let func = FuncDef::new(
            2, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32Add,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(3), WasmValue::I32(7)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(10)]);
    }

    #[test]
    fn subtraction() {
        let func = FuncDef::new(
            2, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32Sub,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(10), WasmValue::I32(4)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(6)]);
    }

    #[test]
    fn multiplication() {
        let func = FuncDef::new(
            2, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32Mul,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(6), WasmValue::I32(7)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(42)]);
    }

    #[test]
    fn division_by_zero_traps() {
        let func = FuncDef::new(
            2, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32DivS,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(10), WasmValue::I32(0)]);
        assert_eq!(result, Err(Trap::DivisionByZero));
    }

    #[test]
    fn i32_min_div_neg1_traps() {
        let func = FuncDef::new(
            2, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32DivS,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(i32::MIN), WasmValue::I32(-1)]);
        assert_eq!(result, Err(Trap::IntegerOverflow));
    }

    #[test]
    fn local_set_and_get() {
        // func(a: i32) -> i32 { let local0 = a + 1; local0 * 2 }
        let func = FuncDef::new(
            1, 1, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(1),
                Instruction::I32Add,
                Instruction::LocalSet(1),
                Instruction::LocalGet(1),
                Instruction::I32Const(2),
                Instruction::I32Mul,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(5)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(12)]);
    }

    #[test]
    fn unreachable_traps() {
        let func = FuncDef::new(0, 0, 0, vec![Instruction::Unreachable]);
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[]);
        assert_eq!(result, Err(Trap::UnreachableExecuted));
    }

    #[test]
    fn function_call() {
        // func0(a, b) -> i32 { a + b }
        let add = FuncDef::new(
            2, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::LocalGet(1),
                Instruction::I32Add,
            ],
        );
        // func1(x) -> i32 { call func0(x, 10) }
        let caller = FuncDef::new(
            1, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(10),
                Instruction::Call(0),
            ],
        );
        let mut interp = make_interp(vec![add, caller]);
        let result = interp.invoke(1, &[WasmValue::I32(5)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(15)]);
    }

    #[test]
    fn block_and_br() {
        // func() -> i32 {
        //   block {
        //     i32.const 42
        //     br 0
        //     i32.const 99  // dead code
        //   }
        // }
        let func = FuncDef::new(
            0, 0, 1,
            vec![
                Instruction::Block(1),
                Instruction::I32Const(42),
                Instruction::Br(0),
                Instruction::I32Const(99),
                Instruction::End,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(42)]);
    }

    #[test]
    fn loop_with_counter() {
        // Sum 1..=5 using a loop.
        // func() -> i32 {
        //   local0 = 0 (sum), local1 = 0 (counter)
        //   loop {
        //     counter += 1
        //     sum += counter
        //     if counter < 5 then br 0 (loop)
        //   }
        //   sum
        // }
        let func = FuncDef::new(
            0, 2, 1,
            vec![
                Instruction::Loop(0),
                  // counter += 1
                  Instruction::LocalGet(1),
                  Instruction::I32Const(1),
                  Instruction::I32Add,
                  Instruction::LocalSet(1),
                  // sum += counter
                  Instruction::LocalGet(0),
                  Instruction::LocalGet(1),
                  Instruction::I32Add,
                  Instruction::LocalSet(0),
                  // if counter < 5
                  Instruction::LocalGet(1),
                  Instruction::I32Const(5),
                  Instruction::I32LtS,
                  Instruction::BrIf(0),
                Instruction::End,
                Instruction::LocalGet(0),
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(15)]);
    }

    #[test]
    fn if_else_true_branch() {
        // func(x: i32) -> i32 { if x then 1 else 0 }
        let func = FuncDef::new(
            1, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::If(1),
                Instruction::I32Const(1),
                Instruction::Else,
                Instruction::I32Const(0),
                Instruction::End,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(1)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(1)]);
    }

    #[test]
    fn if_else_false_branch() {
        let func = FuncDef::new(
            1, 0, 1,
            vec![
                Instruction::LocalGet(0),
                Instruction::If(1),
                Instruction::I32Const(1),
                Instruction::Else,
                Instruction::I32Const(0),
                Instruction::End,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(0)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(0)]);
    }

    #[test]
    fn select_instruction() {
        let func = FuncDef::new(
            1, 0, 1,
            vec![
                Instruction::I32Const(10),
                Instruction::I32Const(20),
                Instruction::LocalGet(0),
                Instruction::Select,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let r1 = interp.invoke(0, &[WasmValue::I32(1)]).unwrap();
        assert_eq!(r1, vec![WasmValue::I32(10)]);
        let r2 = interp.invoke(0, &[WasmValue::I32(0)]).unwrap();
        assert_eq!(r2, vec![WasmValue::I32(20)]);
    }

    #[test]
    fn drop_instruction() {
        let func = FuncDef::new(
            0, 0, 1,
            vec![
                Instruction::I32Const(42),
                Instruction::I32Const(99),
                Instruction::Drop,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(42)]);
    }

    #[test]
    fn instruction_limit() {
        let func = FuncDef::new(
            0, 0, 0,
            vec![
                Instruction::Loop(0),
                Instruction::Br(0),
                Instruction::End,
            ],
        );
        let config = InterpreterConfig {
            max_instructions: 100,
            ..Default::default()
        };
        let mut interp = Interpreter::new(vec![func], config);
        let result = interp.invoke(0, &[]);
        assert!(matches!(result, Err(Trap::InstructionLimitExceeded { .. })));
    }

    #[test]
    fn call_stack_overflow() {
        // Recursive function that never returns.
        let func = FuncDef::new(0, 0, 0, vec![Instruction::Call(0)]);
        let config = InterpreterConfig {
            max_call_depth: 10,
            ..Default::default()
        };
        let mut interp = Interpreter::new(vec![func], config);
        let result = interp.invoke(0, &[]);
        assert!(matches!(result, Err(Trap::CallStackOverflow { .. })));
    }

    #[test]
    fn stats_tracking() {
        let func = FuncDef::new(
            0, 0, 1,
            vec![
                Instruction::I32Const(1),
                Instruction::I32Const(2),
                Instruction::I32Add,
            ],
        );
        let mut interp = make_interp(vec![func]);
        interp.invoke(0, &[]).unwrap();
        assert_eq!(interp.stats().instructions_executed, 3);
        assert!(interp.stats().max_stack_depth_reached >= 1);
    }

    #[test]
    fn wasm_value_type_checking() {
        let v = WasmValue::I32(42);
        assert!(v.as_i32().is_ok());
        assert!(v.as_i64().is_err());
        assert_eq!(v.type_name(), "i32");
    }

    #[test]
    fn local_tee() {
        // func(a: i32) -> i32 { local_tee stores and keeps value on stack }
        let func = FuncDef::new(
            1, 1, 1,
            vec![
                Instruction::I32Const(99),
                Instruction::LocalTee(1),
                Instruction::Drop,
                Instruction::LocalGet(1),
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[WasmValue::I32(0)]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(99)]);
    }

    #[test]
    fn i32_comparison_ops() {
        let func = FuncDef::new(
            0, 0, 1,
            vec![
                Instruction::I32Const(5),
                Instruction::I32Const(10),
                Instruction::I32LtS,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(1)]);
    }

    #[test]
    fn undefined_function_trap() {
        let func = FuncDef::new(0, 0, 0, vec![Instruction::Nop]);
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(99, &[]);
        assert!(matches!(result, Err(Trap::UndefinedFunction(99))));
    }

    #[test]
    fn trap_display() {
        let t = Trap::DivisionByZero;
        assert_eq!(t.to_string(), "division by zero");
    }

    #[test]
    fn bitwise_ops() {
        let func = FuncDef::new(
            0, 0, 1,
            vec![
                Instruction::I32Const(0b1100),
                Instruction::I32Const(0b1010),
                Instruction::I32And,
            ],
        );
        let mut interp = make_interp(vec![func]);
        let result = interp.invoke(0, &[]).unwrap();
        assert_eq!(result, vec![WasmValue::I32(0b1000)]);
    }
}
