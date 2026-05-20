//! Stack-based bytecode VM — instructions (push/pop/add/sub/mul/div/cmp/jmp/
//! call/ret), value stack, call frames, local variables, basic mark-sweep GC
//! on value references, and disassembler.

use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Values ──────────────────────────────────────────────────────────────────

/// A value that can live on the VM stack or in local slots.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Nil / unit.
    Nil,
    /// Boolean.
    Bool(bool),
    /// Integer (64-bit signed).
    Int(i64),
    /// Floating-point (64-bit).
    Float(f64),
    /// Heap-allocated string.
    Str(String),
    /// Reference to a heap object (by id).
    Ref(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Nil => write!(f, "nil"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(n) => write!(f, "{n}"),
            Self::Float(n) => write!(f, "{n}"),
            Self::Str(s) => write!(f, "{s}"),
            Self::Ref(id) => write!(f, "ref({id})"),
        }
    }
}

impl Value {
    /// Attempt to convert to int.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(n) => Some(*n),
            Self::Float(f) => Some(*f as i64),
            Self::Bool(b) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Attempt to convert to float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            Self::Float(f) => Some(*f),
            Self::Int(n) => Some(*n as f64),
            _ => None,
        }
    }

    /// Truthy check.
    pub fn is_truthy(&self) -> bool {
        match self {
            Self::Nil => false,
            Self::Bool(b) => *b,
            Self::Int(n) => *n != 0,
            Self::Float(f) => *f != 0.0,
            Self::Str(s) => !s.is_empty(),
            Self::Ref(_) => true,
        }
    }
}

// ── Instructions ────────────────────────────────────────────────────────────

/// Bytecode instructions.
#[derive(Debug, Clone, PartialEq)]
pub enum Opcode {
    /// Push a constant onto the stack.
    Push(Value),
    /// Pop and discard the top of the stack.
    Pop,
    /// Duplicate the top value.
    Dup,
    /// Swap the top two values.
    Swap,
    /// Arithmetic: pop two, push result.
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Neg,
    /// Comparison: pop two, push Bool.
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,
    /// Logical.
    Not,
    And,
    Or,
    /// Load local variable by slot index.
    LoadLocal(usize),
    /// Store top of stack into local slot.
    StoreLocal(usize),
    /// Unconditional jump to instruction index.
    Jump(usize),
    /// Jump if top of stack is falsy (consumes top).
    JumpIfFalse(usize),
    /// Jump if top of stack is truthy (consumes top).
    JumpIfTrue(usize),
    /// Call function at instruction index with N arguments.
    Call(usize, usize),
    /// Return from current call frame.
    Return,
    /// Allocate a heap object from top N stack values, push Ref.
    Alloc(usize),
    /// Load field from heap object: Ref on stack, field index.
    LoadField(usize),
    /// Store into heap object field: value, Ref on stack, field index.
    StoreField(usize),
    /// Print the top value (for debugging).
    Print,
    /// Halt execution.
    Halt,
    /// No operation.
    Nop,
    /// String concatenation: pop two strings, push result.
    Concat,
}

impl fmt::Display for Opcode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Push(v) => write!(f, "PUSH {v}"),
            Self::Pop => write!(f, "POP"),
            Self::Dup => write!(f, "DUP"),
            Self::Swap => write!(f, "SWAP"),
            Self::Add => write!(f, "ADD"),
            Self::Sub => write!(f, "SUB"),
            Self::Mul => write!(f, "MUL"),
            Self::Div => write!(f, "DIV"),
            Self::Mod => write!(f, "MOD"),
            Self::Neg => write!(f, "NEG"),
            Self::Eq => write!(f, "EQ"),
            Self::Ne => write!(f, "NE"),
            Self::Lt => write!(f, "LT"),
            Self::Gt => write!(f, "GT"),
            Self::Le => write!(f, "LE"),
            Self::Ge => write!(f, "GE"),
            Self::Not => write!(f, "NOT"),
            Self::And => write!(f, "AND"),
            Self::Or => write!(f, "OR"),
            Self::LoadLocal(slot) => write!(f, "LOAD_LOCAL {slot}"),
            Self::StoreLocal(slot) => write!(f, "STORE_LOCAL {slot}"),
            Self::Jump(addr) => write!(f, "JUMP {addr}"),
            Self::JumpIfFalse(addr) => write!(f, "JUMP_IF_FALSE {addr}"),
            Self::JumpIfTrue(addr) => write!(f, "JUMP_IF_TRUE {addr}"),
            Self::Call(addr, argc) => write!(f, "CALL {addr} ({argc} args)"),
            Self::Return => write!(f, "RETURN"),
            Self::Alloc(n) => write!(f, "ALLOC {n}"),
            Self::LoadField(idx) => write!(f, "LOAD_FIELD {idx}"),
            Self::StoreField(idx) => write!(f, "STORE_FIELD {idx}"),
            Self::Print => write!(f, "PRINT"),
            Self::Halt => write!(f, "HALT"),
            Self::Nop => write!(f, "NOP"),
            Self::Concat => write!(f, "CONCAT"),
        }
    }
}

// ── Heap object ─────────────────────────────────────────────────────────────

/// A heap-allocated object.
#[derive(Debug, Clone)]
struct HeapObject {
    fields: Vec<Value>,
    marked: bool,
}

// ── Call frame ──────────────────────────────────────────────────────────────

/// A call frame on the VM call stack.
#[derive(Debug, Clone)]
struct CallFrame {
    /// Instruction pointer to return to.
    return_ip: usize,
    /// Base of local variable slots for this frame.
    local_base: usize,
    /// Number of local slots allocated.
    local_count: usize,
    /// Stack depth at the time of the call.
    stack_base: usize,
}

// ── VM errors ───────────────────────────────────────────────────────────────

/// Runtime errors from the VM.
#[derive(Debug, Clone, PartialEq)]
pub enum VmError {
    /// Stack underflow.
    StackUnderflow,
    /// Stack overflow.
    StackOverflow,
    /// Type error during execution.
    TypeError(String),
    /// Division by zero.
    DivisionByZero,
    /// Invalid instruction pointer.
    InvalidIp(usize),
    /// Invalid local variable slot.
    InvalidLocal(usize),
    /// Invalid heap reference.
    InvalidRef(usize),
    /// Invalid field index.
    InvalidField(usize),
    /// Execution limit exceeded.
    ExecutionLimit,
    /// Halt instruction reached.
    Halted,
}

impl fmt::Display for VmError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StackUnderflow => write!(f, "stack underflow"),
            Self::StackOverflow => write!(f, "stack overflow"),
            Self::TypeError(msg) => write!(f, "type error: {msg}"),
            Self::DivisionByZero => write!(f, "division by zero"),
            Self::InvalidIp(ip) => write!(f, "invalid instruction pointer: {ip}"),
            Self::InvalidLocal(slot) => write!(f, "invalid local slot: {slot}"),
            Self::InvalidRef(id) => write!(f, "invalid heap reference: {id}"),
            Self::InvalidField(idx) => write!(f, "invalid field index: {idx}"),
            Self::ExecutionLimit => write!(f, "execution limit exceeded"),
            Self::Halted => write!(f, "execution halted"),
        }
    }
}

// ── VM ──────────────────────────────────────────────────────────────────────

/// A stack-based bytecode virtual machine.
pub struct Vm {
    /// The bytecode program.
    code: Vec<Opcode>,
    /// Instruction pointer.
    ip: usize,
    /// Value stack.
    stack: Vec<Value>,
    /// Local variable slots.
    locals: Vec<Value>,
    /// Call frame stack.
    frames: Vec<CallFrame>,
    /// Heap.
    heap: HashMap<usize, HeapObject>,
    next_heap_id: usize,
    /// Maximum stack depth.
    max_stack: usize,
    /// Maximum steps before ExecutionLimit.
    max_steps: u64,
    /// Output from Print instructions.
    output: Vec<String>,
    /// Whether execution has halted.
    halted: bool,
}

impl Vm {
    /// Create a new VM with the given bytecode.
    pub fn new(code: Vec<Opcode>) -> Self {
        Self {
            code,
            ip: 0,
            stack: Vec::with_capacity(256),
            locals: Vec::new(),
            frames: Vec::new(),
            heap: HashMap::new(),
            next_heap_id: 0,
            max_stack: 4096,
            max_steps: 1_000_000,
            output: Vec::new(),
            halted: false,
        }
    }

    /// Set the maximum number of execution steps.
    pub fn set_max_steps(&mut self, limit: u64) {
        self.max_steps = limit;
    }

    /// Set the maximum stack depth.
    pub fn set_max_stack(&mut self, limit: usize) {
        self.max_stack = limit;
    }

    /// Get collected output from Print instructions.
    pub fn output(&self) -> &[String] {
        &self.output
    }

    /// Get the current value stack.
    pub fn stack(&self) -> &[Value] {
        &self.stack
    }

    /// Get the current IP.
    pub fn ip(&self) -> usize {
        self.ip
    }

    /// Pre-allocate local variable slots for the top-level.
    pub fn allocate_locals(&mut self, count: usize) {
        self.locals.resize(count, Value::Nil);
    }

    fn push(&mut self, v: Value) -> Result<(), VmError> {
        if self.stack.len() >= self.max_stack {
            return Err(VmError::StackOverflow);
        }
        self.stack.push(v);
        Ok(())
    }

    fn pop(&mut self) -> Result<Value, VmError> {
        self.stack.pop().ok_or(VmError::StackUnderflow)
    }

    fn peek(&self) -> Result<&Value, VmError> {
        self.stack.last().ok_or(VmError::StackUnderflow)
    }

    fn local_slot(&self, slot: usize) -> Result<usize, VmError> {
        let base = self.frames.last().map_or(0, |f| f.local_base);
        let idx = base + slot;
        if idx >= self.locals.len() {
            return Err(VmError::InvalidLocal(slot));
        }
        Ok(idx)
    }

    /// Run the VM to completion, returning the top of stack.
    pub fn run(&mut self) -> Result<Value, VmError> {
        let mut steps: u64 = 0;

        loop {
            if steps >= self.max_steps {
                return Err(VmError::ExecutionLimit);
            }
            steps += 1;

            if self.ip >= self.code.len() {
                break;
            }

            let instr = self.code[self.ip].clone();
            self.ip += 1;

            match instr {
                Opcode::Push(v) => self.push(v)?,
                Opcode::Pop => { self.pop()?; }
                Opcode::Dup => {
                    let v = self.peek()?.clone();
                    self.push(v)?;
                }
                Opcode::Swap => {
                    let a = self.pop()?;
                    let b = self.pop()?;
                    self.push(a)?;
                    self.push(b)?;
                }
                Opcode::Add => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.arith_op(&a, &b, "+")?;
                    self.push(result)?;
                }
                Opcode::Sub => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.arith_op(&a, &b, "-")?;
                    self.push(result)?;
                }
                Opcode::Mul => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    let result = self.arith_op(&a, &b, "*")?;
                    self.push(result)?;
                }
                Opcode::Div => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    // Check division by zero
                    match (&a, &b) {
                        (Value::Int(_), Value::Int(0)) => return Err(VmError::DivisionByZero),
                        (Value::Float(_), Value::Float(d)) if *d == 0.0 => {
                            return Err(VmError::DivisionByZero)
                        }
                        (Value::Int(_), Value::Float(d)) if *d == 0.0 => {
                            return Err(VmError::DivisionByZero)
                        }
                        (Value::Float(_), Value::Int(0)) => {
                            return Err(VmError::DivisionByZero)
                        }
                        _ => {}
                    }
                    let result = self.arith_op(&a, &b, "/")?;
                    self.push(result)?;
                }
                Opcode::Mod => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    match (&a, &b) {
                        (Value::Int(x), Value::Int(y)) => {
                            if *y == 0 {
                                return Err(VmError::DivisionByZero);
                            }
                            self.push(Value::Int(x % y))?;
                        }
                        _ => return Err(VmError::TypeError("mod requires ints".into())),
                    }
                }
                Opcode::Neg => {
                    let v = self.pop()?;
                    match v {
                        Value::Int(n) => self.push(Value::Int(-n))?,
                        Value::Float(f) => self.push(Value::Float(-f))?,
                        _ => return Err(VmError::TypeError("neg requires number".into())),
                    }
                }
                Opcode::Eq => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(a == b))?;
                }
                Opcode::Ne => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(a != b))?;
                }
                Opcode::Lt => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(self.compare(&a, &b)? < 0))?;
                }
                Opcode::Gt => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(self.compare(&a, &b)? > 0))?;
                }
                Opcode::Le => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(self.compare(&a, &b)? <= 0))?;
                }
                Opcode::Ge => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(self.compare(&a, &b)? >= 0))?;
                }
                Opcode::Not => {
                    let v = self.pop()?;
                    self.push(Value::Bool(!v.is_truthy()))?;
                }
                Opcode::And => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(a.is_truthy() && b.is_truthy()))?;
                }
                Opcode::Or => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.push(Value::Bool(a.is_truthy() || b.is_truthy()))?;
                }
                Opcode::LoadLocal(slot) => {
                    let idx = self.local_slot(slot)?;
                    let v = self.locals[idx].clone();
                    self.push(v)?;
                }
                Opcode::StoreLocal(slot) => {
                    let idx = self.local_slot(slot)?;
                    let v = self.pop()?;
                    self.locals[idx] = v;
                }
                Opcode::Jump(addr) => {
                    self.ip = addr;
                }
                Opcode::JumpIfFalse(addr) => {
                    let v = self.pop()?;
                    if !v.is_truthy() {
                        self.ip = addr;
                    }
                }
                Opcode::JumpIfTrue(addr) => {
                    let v = self.pop()?;
                    if v.is_truthy() {
                        self.ip = addr;
                    }
                }
                Opcode::Call(addr, argc) => {
                    let local_base = self.locals.len();
                    // Push args into new local slots
                    let stack_len = self.stack.len();
                    if stack_len < argc {
                        return Err(VmError::StackUnderflow);
                    }
                    // Pop args in reverse order
                    let mut args = Vec::with_capacity(argc);
                    for _ in 0..argc {
                        args.push(self.pop()?);
                    }
                    args.reverse();
                    for arg in args {
                        self.locals.push(arg);
                    }
                    // Reserve some extra local slots
                    let local_count = argc + 8;
                    while self.locals.len() < local_base + local_count {
                        self.locals.push(Value::Nil);
                    }
                    self.frames.push(CallFrame {
                        return_ip: self.ip,
                        local_base,
                        local_count,
                        stack_base: self.stack.len(),
                    });
                    self.ip = addr;
                }
                Opcode::Return => {
                    let retval = if !self.stack.is_empty() {
                        self.pop()?
                    } else {
                        Value::Nil
                    };
                    if let Some(frame) = self.frames.pop() {
                        self.ip = frame.return_ip;
                        self.locals.truncate(frame.local_base);
                        self.stack.truncate(frame.stack_base);
                        self.push(retval)?;
                    } else {
                        self.push(retval)?;
                        break;
                    }
                }
                Opcode::Alloc(n) => {
                    let mut fields = Vec::with_capacity(n);
                    for _ in 0..n {
                        fields.push(self.pop()?);
                    }
                    fields.reverse();
                    let id = self.next_heap_id;
                    self.next_heap_id += 1;
                    self.heap.insert(id, HeapObject { fields, marked: false });
                    self.push(Value::Ref(id))?;
                }
                Opcode::LoadField(idx) => {
                    let r = self.pop()?;
                    if let Value::Ref(id) = r {
                        let obj = self.heap.get(&id).ok_or(VmError::InvalidRef(id))?;
                        let val = obj.fields.get(idx).ok_or(VmError::InvalidField(idx))?.clone();
                        self.push(val)?;
                    } else {
                        return Err(VmError::TypeError("load_field requires ref".into()));
                    }
                }
                Opcode::StoreField(idx) => {
                    let r = self.pop()?;
                    let val = self.pop()?;
                    if let Value::Ref(id) = r {
                        let obj = self.heap.get_mut(&id).ok_or(VmError::InvalidRef(id))?;
                        if idx >= obj.fields.len() {
                            return Err(VmError::InvalidField(idx));
                        }
                        obj.fields[idx] = val;
                    } else {
                        return Err(VmError::TypeError("store_field requires ref".into()));
                    }
                }
                Opcode::Print => {
                    let v = self.peek()?.clone();
                    self.output.push(format!("{v}"));
                }
                Opcode::Halt => {
                    self.halted = true;
                    return Err(VmError::Halted);
                }
                Opcode::Nop => {}
                Opcode::Concat => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    match (a, b) {
                        (Value::Str(sa), Value::Str(sb)) => {
                            self.push(Value::Str(format!("{sa}{sb}")))?;
                        }
                        (a, b) => {
                            self.push(Value::Str(format!("{a}{b}")))?;
                        }
                    }
                }
            }
        }

        if let Some(v) = self.stack.last() {
            Ok(v.clone())
        } else {
            Ok(Value::Nil)
        }
    }

    fn arith_op(&self, a: &Value, b: &Value, op: &str) -> Result<Value, VmError> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => match op {
                "+" => Ok(Value::Int(x.wrapping_add(*y))),
                "-" => Ok(Value::Int(x.wrapping_sub(*y))),
                "*" => Ok(Value::Int(x.wrapping_mul(*y))),
                "/" => Ok(Value::Int(x.wrapping_div(*y))),
                _ => Err(VmError::TypeError(format!("unknown op {op}"))),
            },
            (Value::Float(x), Value::Float(y)) => match op {
                "+" => Ok(Value::Float(x + y)),
                "-" => Ok(Value::Float(x - y)),
                "*" => Ok(Value::Float(x * y)),
                "/" => Ok(Value::Float(x / y)),
                _ => Err(VmError::TypeError(format!("unknown op {op}"))),
            },
            (Value::Int(x), Value::Float(y)) => {
                let xf = *x as f64;
                match op {
                    "+" => Ok(Value::Float(xf + y)),
                    "-" => Ok(Value::Float(xf - y)),
                    "*" => Ok(Value::Float(xf * y)),
                    "/" => Ok(Value::Float(xf / y)),
                    _ => Err(VmError::TypeError(format!("unknown op {op}"))),
                }
            }
            (Value::Float(x), Value::Int(y)) => {
                let yf = *y as f64;
                match op {
                    "+" => Ok(Value::Float(x + yf)),
                    "-" => Ok(Value::Float(x - yf)),
                    "*" => Ok(Value::Float(x * yf)),
                    "/" => Ok(Value::Float(x / yf)),
                    _ => Err(VmError::TypeError(format!("unknown op {op}"))),
                }
            }
            _ => Err(VmError::TypeError(format!("cannot {op} {a} and {b}"))),
        }
    }

    fn compare(&self, a: &Value, b: &Value) -> Result<i32, VmError> {
        match (a, b) {
            (Value::Int(x), Value::Int(y)) => Ok(x.cmp(y) as i32),
            (Value::Float(x), Value::Float(y)) => {
                Ok(x.partial_cmp(y).map_or(0, |o| o as i32))
            }
            (Value::Int(x), Value::Float(y)) => {
                let xf = *x as f64;
                Ok(xf.partial_cmp(y).map_or(0, |o| o as i32))
            }
            (Value::Float(x), Value::Int(y)) => {
                let yf = *y as f64;
                Ok(x.partial_cmp(&yf).map_or(0, |o| o as i32))
            }
            (Value::Str(a), Value::Str(b)) => Ok(a.cmp(b) as i32),
            _ => Err(VmError::TypeError(format!("cannot compare {a} and {b}"))),
        }
    }

    // ── GC ──────────────────────────────────────────────────────────────

    /// Run a mark-sweep garbage collection pass.
    pub fn collect_garbage(&mut self) {
        // Unmark all objects
        for obj in self.heap.values_mut() {
            obj.marked = false;
        }

        // Mark from roots: stack + locals
        let mut worklist: Vec<usize> = Vec::new();
        for v in &self.stack {
            if let Value::Ref(id) = v {
                worklist.push(*id);
            }
        }
        for v in &self.locals {
            if let Value::Ref(id) = v {
                worklist.push(*id);
            }
        }

        let mut visited = HashSet::new();
        while let Some(id) = worklist.pop() {
            if visited.contains(&id) {
                continue;
            }
            visited.insert(id);
            if let Some(obj) = self.heap.get_mut(&id) {
                obj.marked = true;
                let field_refs: Vec<usize> = obj
                    .fields
                    .iter()
                    .filter_map(|v| if let Value::Ref(r) = v { Some(*r) } else { None })
                    .collect();
                worklist.extend(field_refs);
            }
        }

        // Sweep: remove unmarked objects
        self.heap.retain(|_, obj| obj.marked);
    }

    /// Number of live heap objects.
    pub fn heap_size(&self) -> usize {
        self.heap.len()
    }
}

// ── Disassembler ────────────────────────────────────────────────────────────

/// Disassemble a bytecode program into a human-readable string.
pub fn disassemble(code: &[Opcode]) -> String {
    let mut out = String::new();
    for (i, op) in code.iter().enumerate() {
        out.push_str(&format!("{i:04}: {op}\n"));
    }
    out
}

/// Disassemble a single instruction.
pub fn disassemble_one(op: &Opcode) -> String {
    format!("{op}")
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn run(code: Vec<Opcode>) -> Value {
        Vm::new(code).run().unwrap()
    }

    #[test]
    fn test_push_int() {
        let v = run(vec![Opcode::Push(Value::Int(42))]);
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn test_push_float() {
        let v = run(vec![Opcode::Push(Value::Float(3.14))]);
        assert_eq!(v, Value::Float(3.14));
    }

    #[test]
    fn test_add_ints() {
        let v = run(vec![
            Opcode::Push(Value::Int(10)),
            Opcode::Push(Value::Int(20)),
            Opcode::Add,
        ]);
        assert_eq!(v, Value::Int(30));
    }

    #[test]
    fn test_sub() {
        let v = run(vec![
            Opcode::Push(Value::Int(50)),
            Opcode::Push(Value::Int(30)),
            Opcode::Sub,
        ]);
        assert_eq!(v, Value::Int(20));
    }

    #[test]
    fn test_mul() {
        let v = run(vec![
            Opcode::Push(Value::Int(6)),
            Opcode::Push(Value::Int(7)),
            Opcode::Mul,
        ]);
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn test_div() {
        let v = run(vec![
            Opcode::Push(Value::Int(100)),
            Opcode::Push(Value::Int(4)),
            Opcode::Div,
        ]);
        assert_eq!(v, Value::Int(25));
    }

    #[test]
    fn test_div_by_zero() {
        let mut vm = Vm::new(vec![
            Opcode::Push(Value::Int(1)),
            Opcode::Push(Value::Int(0)),
            Opcode::Div,
        ]);
        let err = vm.run().unwrap_err();
        assert_eq!(err, VmError::DivisionByZero);
    }

    #[test]
    fn test_mod() {
        let v = run(vec![
            Opcode::Push(Value::Int(10)),
            Opcode::Push(Value::Int(3)),
            Opcode::Mod,
        ]);
        assert_eq!(v, Value::Int(1));
    }

    #[test]
    fn test_neg() {
        let v = run(vec![Opcode::Push(Value::Int(5)), Opcode::Neg]);
        assert_eq!(v, Value::Int(-5));
    }

    #[test]
    fn test_comparison_eq() {
        let v = run(vec![
            Opcode::Push(Value::Int(5)),
            Opcode::Push(Value::Int(5)),
            Opcode::Eq,
        ]);
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn test_comparison_lt() {
        let v = run(vec![
            Opcode::Push(Value::Int(3)),
            Opcode::Push(Value::Int(5)),
            Opcode::Lt,
        ]);
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn test_logical_not() {
        let v = run(vec![Opcode::Push(Value::Bool(true)), Opcode::Not]);
        assert_eq!(v, Value::Bool(false));
    }

    #[test]
    fn test_logical_and() {
        let v = run(vec![
            Opcode::Push(Value::Bool(true)),
            Opcode::Push(Value::Bool(false)),
            Opcode::And,
        ]);
        assert_eq!(v, Value::Bool(false));
    }

    #[test]
    fn test_jump() {
        // Push 1, jump over push 2, push 3
        let v = run(vec![
            Opcode::Push(Value::Int(1)),
            Opcode::Jump(3),
            Opcode::Push(Value::Int(2)),
            Opcode::Push(Value::Int(3)),
            Opcode::Add,
        ]);
        assert_eq!(v, Value::Int(4)); // 1 + 3
    }

    #[test]
    fn test_jump_if_false() {
        let v = run(vec![
            Opcode::Push(Value::Bool(false)),
            Opcode::JumpIfFalse(3),
            Opcode::Push(Value::Int(1)),
            Opcode::Push(Value::Int(42)),
        ]);
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn test_locals() {
        let mut vm = Vm::new(vec![
            Opcode::Push(Value::Int(99)),
            Opcode::StoreLocal(0),
            Opcode::LoadLocal(0),
        ]);
        vm.allocate_locals(4);
        let v = vm.run().unwrap();
        assert_eq!(v, Value::Int(99));
    }

    #[test]
    fn test_call_return() {
        // main: push 10, push 20, call func at 5 with 2 args
        // func: load local 0, load local 1, add, return
        let mut vm = Vm::new(vec![
            Opcode::Push(Value::Int(10)),  // 0
            Opcode::Push(Value::Int(20)),  // 1
            Opcode::Call(4, 2),            // 2
            Opcode::Halt,                  // 3
            Opcode::LoadLocal(0),          // 4 (func start)
            Opcode::LoadLocal(1),          // 5
            Opcode::Add,                   // 6
            Opcode::Return,                // 7
        ]);
        vm.allocate_locals(4);
        let err = vm.run().unwrap_err();
        assert_eq!(err, VmError::Halted);
        // Check stack has result
        assert_eq!(vm.stack().last(), Some(&Value::Int(30)));
    }

    #[test]
    fn test_dup() {
        let v = run(vec![Opcode::Push(Value::Int(5)), Opcode::Dup, Opcode::Add]);
        assert_eq!(v, Value::Int(10));
    }

    #[test]
    fn test_swap() {
        let v = run(vec![
            Opcode::Push(Value::Int(1)),
            Opcode::Push(Value::Int(2)),
            Opcode::Swap,
        ]);
        // After swap: stack is [2, 1], top is 1
        assert_eq!(v, Value::Int(1));
    }

    #[test]
    fn test_heap_alloc_and_load_field() {
        let v = run(vec![
            Opcode::Push(Value::Int(10)),
            Opcode::Push(Value::Int(20)),
            Opcode::Alloc(2),         // create object with 2 fields
            Opcode::LoadField(1),     // load field 1 (20)
        ]);
        assert_eq!(v, Value::Int(20));
    }

    #[test]
    fn test_heap_store_field() {
        let v = run(vec![
            Opcode::Push(Value::Int(10)),
            Opcode::Alloc(1),              // obj with [10]
            Opcode::Dup,                   // dup ref
            Opcode::Push(Value::Int(99)),  // new value
            Opcode::Swap,                  // put ref on top for store
            Opcode::StoreField(0),         // store 99 at field 0
            Opcode::LoadField(0),          // load back
        ]);
        assert_eq!(v, Value::Int(99));
    }

    #[test]
    fn test_gc_collects_unreachable() {
        let mut vm = Vm::new(vec![
            Opcode::Push(Value::Int(1)),
            Opcode::Alloc(1),      // creates object 0
            Opcode::Pop,           // drop the reference
            Opcode::Push(Value::Int(2)),
            Opcode::Alloc(1),      // creates object 1
        ]);
        vm.run().unwrap();
        assert_eq!(vm.heap_size(), 2);
        vm.collect_garbage();
        assert_eq!(vm.heap_size(), 1); // only the reachable one
    }

    #[test]
    fn test_print() {
        let mut vm = Vm::new(vec![
            Opcode::Push(Value::Str("hello".into())),
            Opcode::Print,
        ]);
        vm.run().unwrap();
        assert_eq!(vm.output(), &["hello"]);
    }

    #[test]
    fn test_concat() {
        let v = run(vec![
            Opcode::Push(Value::Str("foo".into())),
            Opcode::Push(Value::Str("bar".into())),
            Opcode::Concat,
        ]);
        assert_eq!(v, Value::Str("foobar".into()));
    }

    #[test]
    fn test_execution_limit() {
        let mut vm = Vm::new(vec![Opcode::Jump(0)]); // infinite loop
        vm.set_max_steps(100);
        let err = vm.run().unwrap_err();
        assert_eq!(err, VmError::ExecutionLimit);
    }

    #[test]
    fn test_stack_underflow() {
        let mut vm = Vm::new(vec![Opcode::Pop]);
        let err = vm.run().unwrap_err();
        assert_eq!(err, VmError::StackUnderflow);
    }

    #[test]
    fn test_float_arithmetic() {
        let v = run(vec![
            Opcode::Push(Value::Float(1.5)),
            Opcode::Push(Value::Float(2.5)),
            Opcode::Add,
        ]);
        assert_eq!(v, Value::Float(4.0));
    }

    #[test]
    fn test_mixed_int_float() {
        let v = run(vec![
            Opcode::Push(Value::Int(2)),
            Opcode::Push(Value::Float(3.5)),
            Opcode::Mul,
        ]);
        assert_eq!(v, Value::Float(7.0));
    }

    #[test]
    fn test_disassemble() {
        let code = vec![
            Opcode::Push(Value::Int(1)),
            Opcode::Push(Value::Int(2)),
            Opcode::Add,
            Opcode::Halt,
        ];
        let text = disassemble(&code);
        assert!(text.contains("PUSH 1"));
        assert!(text.contains("ADD"));
        assert!(text.contains("HALT"));
    }

    #[test]
    fn test_disassemble_one() {
        let s = disassemble_one(&Opcode::Jump(42));
        assert_eq!(s, "JUMP 42");
    }

    #[test]
    fn test_value_display() {
        assert_eq!(format!("{}", Value::Nil), "nil");
        assert_eq!(format!("{}", Value::Int(42)), "42");
        assert_eq!(format!("{}", Value::Bool(true)), "true");
    }

    #[test]
    fn test_value_truthy() {
        assert!(!Value::Nil.is_truthy());
        assert!(Value::Int(1).is_truthy());
        assert!(!Value::Int(0).is_truthy());
        assert!(Value::Str("x".into()).is_truthy());
        assert!(!Value::Str(String::new()).is_truthy());
    }

    #[test]
    fn test_nop() {
        let v = run(vec![Opcode::Push(Value::Int(7)), Opcode::Nop]);
        assert_eq!(v, Value::Int(7));
    }
}
