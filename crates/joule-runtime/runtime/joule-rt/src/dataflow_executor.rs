//! Dataflow Graph Executor
//!
//! This module executes dataflow graphs with energy-aware scheduling.
//! Operators fire when all their inputs are ready, enabling implicit parallelism.
//!
//! # Key Features
//!
//! - Respects dataflow dependencies (operators fire when inputs ready)
//! - Energy-aware parallelism control based on thermal state
//! - Bounded channels between operators to prevent memory blowup
//! - Integration with the async runtime for non-blocking execution

use std::collections::{HashMap, VecDeque};

/// Unique identifier for operators in the executor
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OperatorId(pub u32);

/// Unique identifier for channels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChannelId(pub u32);

// ============================================================================
// Token Types
// ============================================================================

/// Value carried by tokens through the dataflow graph
#[derive(Debug, Clone)]
pub enum TokenValue {
    /// Unit/void value
    Unit,
    /// Boolean
    Bool(bool),
    /// Signed integer
    Int(i64),
    /// Unsigned integer
    Uint(u64),
    /// 64-bit float
    Float(f64),
    /// Pointer/address
    Ptr(u64),
    /// Array of values
    Array(Vec<TokenValue>),
    /// Tuple of values
    Tuple(Vec<TokenValue>),
}

impl TokenValue {
    pub fn as_bool(&self) -> bool {
        match self {
            TokenValue::Bool(b) => *b,
            TokenValue::Int(i) => *i != 0,
            TokenValue::Uint(u) => *u != 0,
            _ => false,
        }
    }

    pub fn as_i64(&self) -> i64 {
        match self {
            TokenValue::Int(i) => *i,
            TokenValue::Uint(u) => *u as i64,
            TokenValue::Bool(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    pub fn as_u64(&self) -> u64 {
        match self {
            TokenValue::Uint(u) => *u,
            TokenValue::Int(i) => *i as u64,
            TokenValue::Bool(b) => {
                if *b {
                    1
                } else {
                    0
                }
            }
            _ => 0,
        }
    }

    pub fn as_f64(&self) -> f64 {
        match self {
            TokenValue::Float(f) => *f,
            TokenValue::Int(i) => *i as f64,
            TokenValue::Uint(u) => *u as f64,
            _ => 0.0,
        }
    }
}

// ============================================================================
// Channel Implementation
// ============================================================================

/// A bounded FIFO channel between dataflow operators
pub struct DfChannel {
    id: ChannelId,
    /// Bounded buffer of tokens
    buffer: VecDeque<TokenValue>,
    /// Maximum capacity
    capacity: usize,
    /// Whether the channel is closed
    closed: bool,
}

impl DfChannel {
    pub fn new(id: ChannelId, capacity: usize) -> Self {
        Self {
            id,
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            closed: false,
        }
    }

    /// Try to send a token (non-blocking)
    pub fn try_send(&mut self, value: TokenValue) -> Result<(), TokenValue> {
        if self.closed {
            return Err(value);
        }
        if self.buffer.len() >= self.capacity {
            return Err(value);
        }
        self.buffer.push_back(value);
        Ok(())
    }

    /// Try to receive a token (non-blocking)
    pub fn try_recv(&mut self) -> Option<TokenValue> {
        self.buffer.pop_front()
    }

    /// Check if the channel has data available
    pub fn has_data(&self) -> bool {
        !self.buffer.is_empty()
    }

    /// Check if the channel is full
    pub fn is_full(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Check if the channel is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Close the channel
    pub fn close(&mut self) {
        self.closed = true;
    }

    /// Check if channel is closed
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Get the channel ID
    pub fn id(&self) -> ChannelId {
        self.id
    }
}

// ============================================================================
// Compute Operations
// ============================================================================

/// Compute operation kinds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Neg,

    // Bitwise
    BitAnd,
    BitOr,
    BitXor,
    BitNot,
    Shl,
    Shr,

    // Comparison
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,

    // Logical
    And,
    Or,
    Not,

    // Type conversion
    IntToFloat,
    FloatToInt,

    // Special
    Min,
    Max,
    Abs,
    Sqrt,
    Fma,
}

impl ComputeOp {
    /// Execute the operation on the given inputs
    pub fn execute(&self, inputs: &[TokenValue]) -> TokenValue {
        match self {
            // Binary arithmetic
            ComputeOp::Add => {
                if inputs.len() >= 2 {
                    match (&inputs[0], &inputs[1]) {
                        (TokenValue::Int(a), TokenValue::Int(b)) => TokenValue::Int(a + b),
                        (TokenValue::Uint(a), TokenValue::Uint(b)) => TokenValue::Uint(a + b),
                        (TokenValue::Float(a), TokenValue::Float(b)) => TokenValue::Float(a + b),
                        _ => TokenValue::Int(inputs[0].as_i64() + inputs[1].as_i64()),
                    }
                } else if inputs.len() == 1 {
                    inputs[0].clone() // Identity
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Sub => {
                if inputs.len() >= 2 {
                    match (&inputs[0], &inputs[1]) {
                        (TokenValue::Int(a), TokenValue::Int(b)) => TokenValue::Int(a - b),
                        (TokenValue::Float(a), TokenValue::Float(b)) => TokenValue::Float(a - b),
                        _ => TokenValue::Int(inputs[0].as_i64() - inputs[1].as_i64()),
                    }
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Mul => {
                if inputs.len() >= 2 {
                    match (&inputs[0], &inputs[1]) {
                        (TokenValue::Int(a), TokenValue::Int(b)) => TokenValue::Int(a * b),
                        (TokenValue::Float(a), TokenValue::Float(b)) => TokenValue::Float(a * b),
                        _ => TokenValue::Int(inputs[0].as_i64() * inputs[1].as_i64()),
                    }
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Div => {
                if inputs.len() >= 2 {
                    match (&inputs[0], &inputs[1]) {
                        (TokenValue::Int(a), TokenValue::Int(b)) if *b != 0 => {
                            TokenValue::Int(a / b)
                        }
                        (TokenValue::Float(a), TokenValue::Float(b)) => TokenValue::Float(a / b),
                        _ => {
                            let b = inputs[1].as_i64();
                            if b != 0 {
                                TokenValue::Int(inputs[0].as_i64() / b)
                            } else {
                                TokenValue::Int(0) // Avoid division by zero
                            }
                        }
                    }
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Rem => {
                if inputs.len() >= 2 {
                    let b = inputs[1].as_i64();
                    if b != 0 {
                        TokenValue::Int(inputs[0].as_i64() % b)
                    } else {
                        TokenValue::Int(0)
                    }
                } else {
                    TokenValue::Int(0)
                }
            }

            // Unary
            ComputeOp::Neg => {
                if !inputs.is_empty() {
                    match &inputs[0] {
                        TokenValue::Int(a) => TokenValue::Int(-a),
                        TokenValue::Float(a) => TokenValue::Float(-a),
                        _ => TokenValue::Int(-inputs[0].as_i64()),
                    }
                } else {
                    TokenValue::Int(0)
                }
            }

            // Bitwise
            ComputeOp::BitAnd => {
                if inputs.len() >= 2 {
                    TokenValue::Int(inputs[0].as_i64() & inputs[1].as_i64())
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::BitOr => {
                if inputs.len() >= 2 {
                    TokenValue::Int(inputs[0].as_i64() | inputs[1].as_i64())
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::BitXor => {
                if inputs.len() >= 2 {
                    TokenValue::Int(inputs[0].as_i64() ^ inputs[1].as_i64())
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::BitNot => {
                if !inputs.is_empty() {
                    TokenValue::Int(!inputs[0].as_i64())
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Shl => {
                if inputs.len() >= 2 {
                    TokenValue::Int(inputs[0].as_i64() << (inputs[1].as_u64() & 63))
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Shr => {
                if inputs.len() >= 2 {
                    TokenValue::Int(inputs[0].as_i64() >> (inputs[1].as_u64() & 63))
                } else {
                    TokenValue::Int(0)
                }
            }

            // Comparison
            ComputeOp::Eq => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_i64() == inputs[1].as_i64())
                } else {
                    TokenValue::Bool(false)
                }
            }
            ComputeOp::Ne => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_i64() != inputs[1].as_i64())
                } else {
                    TokenValue::Bool(true)
                }
            }
            ComputeOp::Lt => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_i64() < inputs[1].as_i64())
                } else {
                    TokenValue::Bool(false)
                }
            }
            ComputeOp::Le => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_i64() <= inputs[1].as_i64())
                } else {
                    TokenValue::Bool(false)
                }
            }
            ComputeOp::Gt => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_i64() > inputs[1].as_i64())
                } else {
                    TokenValue::Bool(false)
                }
            }
            ComputeOp::Ge => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_i64() >= inputs[1].as_i64())
                } else {
                    TokenValue::Bool(false)
                }
            }

            // Logical
            ComputeOp::And => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_bool() && inputs[1].as_bool())
                } else {
                    TokenValue::Bool(false)
                }
            }
            ComputeOp::Or => {
                if inputs.len() >= 2 {
                    TokenValue::Bool(inputs[0].as_bool() || inputs[1].as_bool())
                } else {
                    TokenValue::Bool(false)
                }
            }
            ComputeOp::Not => {
                if !inputs.is_empty() {
                    TokenValue::Bool(!inputs[0].as_bool())
                } else {
                    TokenValue::Bool(true)
                }
            }

            // Type conversion
            ComputeOp::IntToFloat => {
                if !inputs.is_empty() {
                    TokenValue::Float(inputs[0].as_i64() as f64)
                } else {
                    TokenValue::Float(0.0)
                }
            }
            ComputeOp::FloatToInt => {
                if !inputs.is_empty() {
                    TokenValue::Int(inputs[0].as_f64() as i64)
                } else {
                    TokenValue::Int(0)
                }
            }

            // Special
            ComputeOp::Min => {
                if inputs.len() >= 2 {
                    let a = inputs[0].as_i64();
                    let b = inputs[1].as_i64();
                    TokenValue::Int(a.min(b))
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Max => {
                if inputs.len() >= 2 {
                    let a = inputs[0].as_i64();
                    let b = inputs[1].as_i64();
                    TokenValue::Int(a.max(b))
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Abs => {
                if !inputs.is_empty() {
                    match &inputs[0] {
                        TokenValue::Int(a) => TokenValue::Int(a.abs()),
                        TokenValue::Float(a) => TokenValue::Float(a.abs()),
                        _ => TokenValue::Int(inputs[0].as_i64().abs()),
                    }
                } else {
                    TokenValue::Int(0)
                }
            }
            ComputeOp::Sqrt => {
                if !inputs.is_empty() {
                    TokenValue::Float(inputs[0].as_f64().sqrt())
                } else {
                    TokenValue::Float(0.0)
                }
            }
            ComputeOp::Fma => {
                // Fused multiply-add: a * b + c
                if inputs.len() >= 3 {
                    let a = inputs[0].as_f64();
                    let b = inputs[1].as_f64();
                    let c = inputs[2].as_f64();
                    TokenValue::Float(a.mul_add(b, c))
                } else {
                    TokenValue::Float(0.0)
                }
            }
        }
    }
}

// ============================================================================
// Operator Definitions
// ============================================================================

/// A dataflow operator
pub enum DfOperator {
    /// Pure computation
    Compute {
        op: ComputeOp,
        inputs: Vec<ChannelId>,
        output: ChannelId,
    },

    /// Steer: conditional routing
    Steer {
        decider: ChannelId,
        data: ChannelId,
        true_out: ChannelId,
        false_out: ChannelId,
    },

    /// Stream: fused loop induction variable
    Stream {
        start: ChannelId,
        step: ChannelId,
        bound: ChannelId,
        output: ChannelId,
        done: ChannelId,
        /// Internal state
        current: Option<i64>,
        step_value: Option<i64>,
        bound_value: Option<i64>,
    },

    /// Merge: join multiple inputs
    Merge {
        inputs: Vec<ChannelId>,
        output: ChannelId,
    },

    /// Split: fan-out to multiple outputs
    Split {
        input: ChannelId,
        outputs: Vec<ChannelId>,
    },

    /// Constant: emit a constant value
    Constant {
        value: TokenValue,
        output: ChannelId,
        remaining: Option<u64>,
    },

    /// Source: external input
    Source { external_id: u32, output: ChannelId },

    /// Sink: external output
    Sink { input: ChannelId, external_id: u32 },

    /// Reduce: accumulate values
    Reduce {
        op: ComputeOp,
        initial: ChannelId,
        values: ChannelId,
        count_or_done: ChannelId,
        output: ChannelId,
        /// Internal accumulator
        accumulator: Option<TokenValue>,
        remaining: Option<u64>,
    },
}

impl DfOperator {
    /// Get input channel IDs
    pub fn inputs(&self) -> Vec<ChannelId> {
        match self {
            DfOperator::Compute { inputs, .. } => inputs.clone(),
            DfOperator::Steer { decider, data, .. } => vec![*decider, *data],
            DfOperator::Stream {
                start, step, bound, ..
            } => vec![*start, *step, *bound],
            DfOperator::Merge { inputs, .. } => inputs.clone(),
            DfOperator::Split { input, .. } => vec![*input],
            DfOperator::Constant { .. } => vec![],
            DfOperator::Source { .. } => vec![],
            DfOperator::Sink { input, .. } => vec![*input],
            DfOperator::Reduce {
                initial,
                values,
                count_or_done,
                ..
            } => {
                vec![*initial, *values, *count_or_done]
            }
        }
    }

    /// Get output channel IDs
    pub fn outputs(&self) -> Vec<ChannelId> {
        match self {
            DfOperator::Compute { output, .. } => vec![*output],
            DfOperator::Steer {
                true_out,
                false_out,
                ..
            } => vec![*true_out, *false_out],
            DfOperator::Stream { output, done, .. } => vec![*output, *done],
            DfOperator::Merge { output, .. } => vec![*output],
            DfOperator::Split { outputs, .. } => outputs.clone(),
            DfOperator::Constant { output, .. } => vec![*output],
            DfOperator::Source { output, .. } => vec![*output],
            DfOperator::Sink { .. } => vec![],
            DfOperator::Reduce { output, .. } => vec![*output],
        }
    }
}

// ============================================================================
// Thermal State
// ============================================================================

/// Thermal state for controlling parallelism
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThermalState {
    Cool,
    Nominal,
    Elevated,
    Hot,
    Critical,
}

impl ThermalState {
    /// Get the parallelism factor (0.0 to 1.0)
    pub fn parallelism_factor(&self) -> f32 {
        match self {
            ThermalState::Cool => 1.0,
            ThermalState::Nominal => 0.8,
            ThermalState::Elevated => 0.5,
            ThermalState::Hot => 0.25,
            ThermalState::Critical => 0.1,
        }
    }

    /// Maximum parallel operators for a given hardware parallelism
    pub fn max_parallel(&self, hw_parallelism: usize) -> usize {
        let factor = self.parallelism_factor();
        ((hw_parallelism as f32) * factor).ceil() as usize
    }
}

// ============================================================================
// Execution Statistics
// ============================================================================

/// Statistics from dataflow execution
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total operators executed
    pub operators_executed: u64,
    /// Total tokens processed
    pub tokens_processed: u64,
    /// Number of execution steps
    pub steps: u64,
    /// Maximum parallelism achieved
    pub max_parallelism: usize,
    /// Energy estimate (relative units)
    pub estimated_energy: f64,
}

// ============================================================================
// Dataflow Executor
// ============================================================================

/// Executor for dataflow graphs
pub struct DataflowExecutor {
    /// All operators
    operators: Vec<DfOperator>,
    /// All channels
    channels: Vec<DfChannel>,
    /// External inputs (values to inject)
    external_inputs: HashMap<u32, VecDeque<TokenValue>>,
    /// External outputs (collected results)
    external_outputs: HashMap<u32, Vec<TokenValue>>,
    /// Current thermal state
    thermal_state: ThermalState,
    /// Hardware parallelism (number of cores)
    hw_parallelism: usize,
    /// Execution statistics
    stats: ExecutionStats,
    /// Whether execution is complete
    done: bool,
}

impl DataflowExecutor {
    /// Create a new executor
    pub fn new() -> Self {
        Self {
            operators: Vec::new(),
            channels: Vec::new(),
            external_inputs: HashMap::new(),
            external_outputs: HashMap::new(),
            thermal_state: ThermalState::Nominal,
            hw_parallelism: num_cpus(),
            stats: ExecutionStats::default(),
            done: false,
        }
    }

    /// Set the thermal state
    pub fn set_thermal_state(&mut self, state: ThermalState) {
        self.thermal_state = state;
    }

    /// Add a channel
    pub fn add_channel(&mut self, capacity: usize) -> ChannelId {
        let id = ChannelId(self.channels.len() as u32);
        self.channels.push(DfChannel::new(id, capacity));
        id
    }

    /// Add an operator
    pub fn add_operator(&mut self, op: DfOperator) -> OperatorId {
        let id = OperatorId(self.operators.len() as u32);
        self.operators.push(op);
        id
    }

    /// Provide external input
    pub fn provide_input(&mut self, external_id: u32, value: TokenValue) {
        self.external_inputs
            .entry(external_id)
            .or_insert_with(VecDeque::new)
            .push_back(value);
    }

    /// Get external outputs
    pub fn get_outputs(&self, external_id: u32) -> Option<&Vec<TokenValue>> {
        self.external_outputs.get(&external_id)
    }

    /// Get execution statistics
    pub fn stats(&self) -> &ExecutionStats {
        &self.stats
    }

    /// Check if any operator can fire
    fn find_ready_operators(&self) -> Vec<usize> {
        let max_parallel = self.thermal_state.max_parallel(self.hw_parallelism);
        let mut ready = Vec::new();

        for (idx, op) in self.operators.iter().enumerate() {
            if ready.len() >= max_parallel {
                break;
            }

            if self.can_fire(op) {
                ready.push(idx);
            }
        }

        ready
    }

    /// Check if an operator can fire (all inputs have data)
    fn can_fire(&self, op: &DfOperator) -> bool {
        match op {
            DfOperator::Constant { remaining, .. } => remaining.map(|r| r > 0).unwrap_or(true),
            DfOperator::Source { external_id, .. } => self
                .external_inputs
                .get(external_id)
                .map(|q| !q.is_empty())
                .unwrap_or(false),
            DfOperator::Stream {
                start,
                step,
                bound,
                current,
                ..
            } => {
                if current.is_some() {
                    // Already initialized - can fire
                    true
                } else {
                    // Need to initialize - check if all inputs have data
                    let has_start = self
                        .channels
                        .get(start.0 as usize)
                        .map(|ch| ch.has_data())
                        .unwrap_or(false);
                    let has_step = self
                        .channels
                        .get(step.0 as usize)
                        .map(|ch| ch.has_data())
                        .unwrap_or(false);
                    let has_bound = self
                        .channels
                        .get(bound.0 as usize)
                        .map(|ch| ch.has_data())
                        .unwrap_or(false);
                    has_start && has_step && has_bound
                }
            }
            DfOperator::Reduce {
                accumulator,
                remaining,
                ..
            } => {
                // Can fire if accumulating or has result ready
                accumulator.is_some() && remaining.map(|r| r > 0).unwrap_or(true)
            }
            _ => {
                // For most operators, all inputs must have data
                for ch_id in op.inputs() {
                    if let Some(ch) = self.channels.get(ch_id.0 as usize) {
                        if ch.is_empty() {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Execute one operator
    fn execute_operator(&mut self, idx: usize) {
        // Take the operator temporarily to work around borrow checker
        let mut op = std::mem::replace(
            &mut self.operators[idx],
            DfOperator::Constant {
                value: TokenValue::Unit,
                output: ChannelId(0),
                remaining: Some(0),
            },
        );

        match &mut op {
            DfOperator::Compute {
                op: compute_op,
                inputs,
                output,
            } => {
                // Gather inputs
                let mut values = Vec::new();
                for ch_id in inputs.iter() {
                    if let Some(ch) = self.channels.get_mut(ch_id.0 as usize) {
                        if let Some(val) = ch.try_recv() {
                            values.push(val);
                        }
                    }
                }

                // Execute operation
                let result = compute_op.execute(&values);
                self.stats.tokens_processed += 1;

                // Send result
                if let Some(ch) = self.channels.get_mut(output.0 as usize) {
                    let _ = ch.try_send(result);
                }
            }

            DfOperator::Steer {
                decider,
                data,
                true_out,
                false_out,
            } => {
                let cond = self
                    .channels
                    .get_mut(decider.0 as usize)
                    .and_then(|ch| ch.try_recv());
                let val = self
                    .channels
                    .get_mut(data.0 as usize)
                    .and_then(|ch| ch.try_recv());

                if let (Some(cond), Some(val)) = (cond, val) {
                    let target = if cond.as_bool() {
                        *true_out
                    } else {
                        *false_out
                    };
                    if let Some(ch) = self.channels.get_mut(target.0 as usize) {
                        let _ = ch.try_send(val);
                    }
                    self.stats.tokens_processed += 1;
                }
            }

            DfOperator::Stream {
                start,
                step,
                bound,
                output,
                done,
                current,
                step_value,
                bound_value,
            } => {
                // Initialize if needed
                if current.is_none() {
                    let s = self
                        .channels
                        .get_mut(start.0 as usize)
                        .and_then(|ch| ch.try_recv());
                    let st = self
                        .channels
                        .get_mut(step.0 as usize)
                        .and_then(|ch| ch.try_recv());
                    let b = self
                        .channels
                        .get_mut(bound.0 as usize)
                        .and_then(|ch| ch.try_recv());

                    if let (Some(s), Some(st), Some(b)) = (s, st, b) {
                        *current = Some(s.as_i64());
                        *step_value = Some(st.as_i64());
                        *bound_value = Some(b.as_i64());
                    }
                }

                // Emit current value if not done
                if let (Some(cur), Some(stp), Some(bnd)) = (*current, *step_value, *bound_value) {
                    if cur < bnd {
                        // Emit current value
                        if let Some(ch) = self.channels.get_mut(output.0 as usize) {
                            let _ = ch.try_send(TokenValue::Int(cur));
                        }
                        *current = Some(cur + stp);
                        self.stats.tokens_processed += 1;
                    } else {
                        // Done - emit done signal
                        if let Some(ch) = self.channels.get_mut(done.0 as usize) {
                            let _ = ch.try_send(TokenValue::Bool(true));
                        }
                        *current = None; // Reset for next iteration
                        self.stats.tokens_processed += 1;
                    }
                }
            }

            DfOperator::Merge { inputs, output } => {
                // Take first available input
                for ch_id in inputs.iter() {
                    if let Some(ch) = self.channels.get_mut(ch_id.0 as usize) {
                        if let Some(val) = ch.try_recv() {
                            if let Some(out_ch) = self.channels.get_mut(output.0 as usize) {
                                let _ = out_ch.try_send(val);
                            }
                            self.stats.tokens_processed += 1;
                            break;
                        }
                    }
                }
            }

            DfOperator::Split { input, outputs } => {
                if let Some(ch) = self.channels.get_mut(input.0 as usize) {
                    if let Some(val) = ch.try_recv() {
                        for out_id in outputs.iter() {
                            if let Some(out_ch) = self.channels.get_mut(out_id.0 as usize) {
                                let _ = out_ch.try_send(val.clone());
                            }
                        }
                        self.stats.tokens_processed += 1;
                    }
                }
            }

            DfOperator::Constant {
                value,
                output,
                remaining,
            } => {
                if remaining.map(|r| r > 0).unwrap_or(true) {
                    if let Some(ch) = self.channels.get_mut(output.0 as usize) {
                        let _ = ch.try_send(value.clone());
                    }
                    if let Some(r) = remaining {
                        *r -= 1;
                    }
                    self.stats.tokens_processed += 1;
                }
            }

            DfOperator::Source {
                external_id,
                output,
            } => {
                if let Some(queue) = self.external_inputs.get_mut(external_id) {
                    if let Some(val) = queue.pop_front() {
                        if let Some(ch) = self.channels.get_mut(output.0 as usize) {
                            let _ = ch.try_send(val);
                        }
                        self.stats.tokens_processed += 1;
                    }
                }
            }

            DfOperator::Sink { input, external_id } => {
                if let Some(ch) = self.channels.get_mut(input.0 as usize) {
                    if let Some(val) = ch.try_recv() {
                        self.external_outputs
                            .entry(*external_id)
                            .or_insert_with(Vec::new)
                            .push(val);
                        self.stats.tokens_processed += 1;
                    }
                }
            }

            DfOperator::Reduce {
                op: reduce_op,
                initial,
                values,
                count_or_done,
                output,
                accumulator,
                remaining,
            } => {
                // Initialize if needed
                if accumulator.is_none() {
                    let init = self
                        .channels
                        .get_mut(initial.0 as usize)
                        .and_then(|ch| ch.try_recv());
                    let count = self
                        .channels
                        .get_mut(count_or_done.0 as usize)
                        .and_then(|ch| ch.try_recv());

                    if let (Some(init), Some(count)) = (init, count) {
                        *accumulator = Some(init);
                        *remaining = Some(count.as_u64());
                    }
                }

                // Reduce values
                if let (Some(acc), Some(rem)) = (accumulator.as_mut(), remaining.as_mut()) {
                    if *rem > 0 {
                        if let Some(ch) = self.channels.get_mut(values.0 as usize) {
                            if let Some(val) = ch.try_recv() {
                                let result = reduce_op.execute(&[acc.clone(), val]);
                                *acc = result;
                                *rem -= 1;
                                self.stats.tokens_processed += 1;
                            }
                        }
                    }

                    // Emit result when done
                    if *rem == 0 {
                        if let Some(ch) = self.channels.get_mut(output.0 as usize) {
                            let _ = ch.try_send(acc.clone());
                        }
                        *accumulator = None;
                        self.stats.tokens_processed += 1;
                    }
                }
            }
        }

        // Put operator back
        self.operators[idx] = op;
        self.stats.operators_executed += 1;
    }

    /// Run until no more operators can fire
    pub fn run(&mut self) {
        loop {
            let ready = self.find_ready_operators();

            if ready.is_empty() {
                self.done = true;
                break;
            }

            let ready_count = ready.len();

            // Update max parallelism stat
            if ready_count > self.stats.max_parallelism {
                self.stats.max_parallelism = ready_count;
            }

            // Execute ready operators
            for idx in ready {
                self.execute_operator(idx);
            }

            self.stats.steps += 1;

            // Estimate energy for this step
            self.stats.estimated_energy += ready_count as f64 * 0.001; // Placeholder
        }
    }

    /// Run a fixed number of steps
    pub fn run_steps(&mut self, max_steps: u64) {
        for _ in 0..max_steps {
            let ready = self.find_ready_operators();

            if ready.is_empty() {
                self.done = true;
                break;
            }

            // Update max parallelism stat
            if ready.len() > self.stats.max_parallelism {
                self.stats.max_parallelism = ready.len();
            }

            // Execute ready operators
            for idx in ready {
                self.execute_operator(idx);
            }

            self.stats.steps += 1;
        }
    }

    /// Check if execution is complete
    pub fn is_done(&self) -> bool {
        self.done
    }
}

impl Default for DataflowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_add() {
        let mut exec = DataflowExecutor::new();

        // Create channels
        let ch_a = exec.add_channel(4);
        let ch_b = exec.add_channel(4);
        let ch_out = exec.add_channel(4);

        // Add operators
        exec.add_operator(DfOperator::Source {
            external_id: 0,
            output: ch_a,
        });
        exec.add_operator(DfOperator::Source {
            external_id: 1,
            output: ch_b,
        });
        exec.add_operator(DfOperator::Compute {
            op: ComputeOp::Add,
            inputs: vec![ch_a, ch_b],
            output: ch_out,
        });
        exec.add_operator(DfOperator::Sink {
            input: ch_out,
            external_id: 0,
        });

        // Provide inputs
        exec.provide_input(0, TokenValue::Int(5));
        exec.provide_input(1, TokenValue::Int(3));

        // Run
        exec.run();

        // Check output
        let outputs = exec.get_outputs(0).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_i64(), 8);
    }

    #[test]
    fn test_steer() {
        let mut exec = DataflowExecutor::new();

        // Create channels
        let ch_cond = exec.add_channel(4);
        let ch_data = exec.add_channel(4);
        let ch_true = exec.add_channel(4);
        let ch_false = exec.add_channel(4);

        // Add operators
        exec.add_operator(DfOperator::Source {
            external_id: 0,
            output: ch_cond,
        });
        exec.add_operator(DfOperator::Source {
            external_id: 1,
            output: ch_data,
        });
        exec.add_operator(DfOperator::Steer {
            decider: ch_cond,
            data: ch_data,
            true_out: ch_true,
            false_out: ch_false,
        });
        exec.add_operator(DfOperator::Sink {
            input: ch_true,
            external_id: 0,
        });
        exec.add_operator(DfOperator::Sink {
            input: ch_false,
            external_id: 1,
        });

        // Test true path
        exec.provide_input(0, TokenValue::Bool(true));
        exec.provide_input(1, TokenValue::Int(42));
        exec.run();

        let true_out = exec.get_outputs(0).unwrap();
        assert_eq!(true_out.len(), 1);
        assert_eq!(true_out[0].as_i64(), 42);
    }

    #[test]
    fn test_stream_loop() {
        let mut exec = DataflowExecutor::new();

        // Create channels
        let ch_start = exec.add_channel(4);
        let ch_step = exec.add_channel(4);
        let ch_bound = exec.add_channel(4);
        let ch_output = exec.add_channel(16);
        let ch_done = exec.add_channel(4);

        // Add stream operator
        exec.add_operator(DfOperator::Source {
            external_id: 0,
            output: ch_start,
        });
        exec.add_operator(DfOperator::Source {
            external_id: 1,
            output: ch_step,
        });
        exec.add_operator(DfOperator::Source {
            external_id: 2,
            output: ch_bound,
        });
        exec.add_operator(DfOperator::Stream {
            start: ch_start,
            step: ch_step,
            bound: ch_bound,
            output: ch_output,
            done: ch_done,
            current: None,
            step_value: None,
            bound_value: None,
        });
        exec.add_operator(DfOperator::Sink {
            input: ch_output,
            external_id: 0,
        });
        exec.add_operator(DfOperator::Sink {
            input: ch_done,
            external_id: 1,
        });

        // Initialize stream: start=0, step=1, bound=5
        exec.provide_input(0, TokenValue::Int(0));
        exec.provide_input(1, TokenValue::Int(1));
        exec.provide_input(2, TokenValue::Int(5));

        // Run
        exec.run();

        // Should output 0, 1, 2, 3, 4
        let outputs = exec.get_outputs(0).unwrap();
        assert_eq!(outputs.len(), 5);
        for (i, val) in outputs.iter().enumerate() {
            assert_eq!(val.as_i64(), i as i64);
        }
    }

    #[test]
    fn test_thermal_parallelism() {
        assert_eq!(ThermalState::Cool.max_parallel(8), 8);
        assert_eq!(ThermalState::Nominal.max_parallel(8), 7); // 8 * 0.8 = 6.4, ceil = 7
        assert_eq!(ThermalState::Hot.max_parallel(8), 2); // 8 * 0.25 = 2
        assert_eq!(ThermalState::Critical.max_parallel(8), 1); // 8 * 0.1 = 0.8, ceil = 1
    }

    #[test]
    fn test_compute_operations() {
        assert_eq!(
            ComputeOp::Add
                .execute(&[TokenValue::Int(2), TokenValue::Int(3)])
                .as_i64(),
            5
        );
        assert_eq!(
            ComputeOp::Mul
                .execute(&[TokenValue::Int(4), TokenValue::Int(5)])
                .as_i64(),
            20
        );
        assert_eq!(
            ComputeOp::Lt
                .execute(&[TokenValue::Int(3), TokenValue::Int(5)])
                .as_bool(),
            true
        );
        assert_eq!(
            ComputeOp::Sqrt.execute(&[TokenValue::Float(9.0)]).as_f64(),
            3.0
        );
    }

    #[test]
    fn test_split_merge() {
        let mut exec = DataflowExecutor::new();

        // Create channels
        let ch_in = exec.add_channel(4);
        let ch_a = exec.add_channel(4);
        let ch_b = exec.add_channel(4);
        let ch_out = exec.add_channel(4);

        // Split then merge
        exec.add_operator(DfOperator::Source {
            external_id: 0,
            output: ch_in,
        });
        exec.add_operator(DfOperator::Split {
            input: ch_in,
            outputs: vec![ch_a, ch_b],
        });
        exec.add_operator(DfOperator::Merge {
            inputs: vec![ch_a, ch_b],
            output: ch_out,
        });
        exec.add_operator(DfOperator::Sink {
            input: ch_out,
            external_id: 0,
        });

        exec.provide_input(0, TokenValue::Int(42));
        exec.run();

        // Should get the value through merge
        let outputs = exec.get_outputs(0).unwrap();
        assert!(!outputs.is_empty());
    }
}
