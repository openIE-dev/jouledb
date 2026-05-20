//! ZK Circuit — arithmetic circuit representation for zero-knowledge proofs.
//!
//! Provides a gate-level arithmetic circuit abstraction (add, mul, const),
//! wire values, an R1CS constraint system, a fluent circuit builder API,
//! witness generation, and satisfiability checking.  All field arithmetic
//! uses a configurable prime modulus over `u64`.

use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CircuitError {
    InvalidWire(String),
    UnsatisfiedConstraint(String),
    WitnessGenerationFailed(String),
    InvalidConfig(String),
    DuplicateLabel(String),
    ModulusZero,
}

impl fmt::Display for CircuitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWire(s) => write!(f, "invalid wire: {s}"),
            Self::UnsatisfiedConstraint(s) => write!(f, "unsatisfied constraint: {s}"),
            Self::WitnessGenerationFailed(s) => write!(f, "witness generation failed: {s}"),
            Self::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            Self::DuplicateLabel(s) => write!(f, "duplicate label: {s}"),
            Self::ModulusZero => write!(f, "modulus must be non-zero"),
        }
    }
}

impl std::error::Error for CircuitError {}

// ── Wire ────────────────────────────────────────────────────────

/// A wire in the arithmetic circuit, identified by index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Wire(pub usize);

impl fmt::Display for Wire {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "w{}", self.0)
    }
}

// ── Gate types ──────────────────────────────────────────────────

/// Supported arithmetic gate operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GateOp {
    /// out = left + right (mod p)
    Add { left: Wire, right: Wire, out: Wire },
    /// out = left * right (mod p)
    Mul { left: Wire, right: Wire, out: Wire },
    /// out = constant value
    Const { out: Wire, value: u64 },
    /// out = left + constant (mod p)
    AddConst { left: Wire, out: Wire, value: u64 },
    /// out = left * constant (mod p)
    MulConst { left: Wire, out: Wire, value: u64 },
}

impl fmt::Display for GateOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Add { left, right, out } => write!(f, "{out} = {left} + {right}"),
            Self::Mul { left, right, out } => write!(f, "{out} = {left} * {right}"),
            Self::Const { out, value } => write!(f, "{out} = {value}"),
            Self::AddConst { left, out, value } => write!(f, "{out} = {left} + {value}"),
            Self::MulConst { left, out, value } => write!(f, "{out} = {left} * {value}"),
        }
    }
}

// ── R1CS Constraint ─────────────────────────────────────────────

/// A single R1CS constraint: <a, w> * <b, w> = <c, w>
/// where w is the full witness vector (1, inputs, intermediates, outputs).
#[derive(Debug, Clone)]
pub struct R1csConstraint {
    /// Sparse vector a: (wire_index, coefficient)
    pub a: Vec<(usize, u64)>,
    /// Sparse vector b
    pub b: Vec<(usize, u64)>,
    /// Sparse vector c
    pub c: Vec<(usize, u64)>,
}

impl fmt::Display for R1csConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn fmt_lc(lc: &[(usize, u64)]) -> String {
            if lc.is_empty() {
                return "0".to_string();
            }
            lc.iter()
                .map(|(i, c)| format!("{c}*w{i}"))
                .collect::<Vec<_>>()
                .join(" + ")
        }
        write!(f, "({}) * ({}) = ({})", fmt_lc(&self.a), fmt_lc(&self.b), fmt_lc(&self.c))
    }
}

// ── CircuitConfig ───────────────────────────────────────────────

/// Configuration for the arithmetic circuit.
#[derive(Debug, Clone)]
pub struct CircuitConfig {
    pub modulus: u64,
    pub max_gates: usize,
    pub max_wires: usize,
    pub label: String,
}

impl Default for CircuitConfig {
    fn default() -> Self {
        Self {
            modulus: 0xFFFF_FFFF_FFFF_FFC5, // large 64-bit prime
            max_gates: 1_000_000,
            max_wires: 2_000_000,
            label: String::from("circuit"),
        }
    }
}

impl CircuitConfig {
    pub fn new() -> Self { Self::default() }

    pub fn with_modulus(mut self, m: u64) -> Self { self.modulus = m; self }
    pub fn with_max_gates(mut self, n: usize) -> Self { self.max_gates = n; self }
    pub fn with_max_wires(mut self, n: usize) -> Self { self.max_wires = n; self }
    pub fn with_label(mut self, l: impl Into<String>) -> Self { self.label = l.into(); self }

    pub fn validate(&self) -> Result<(), CircuitError> {
        if self.modulus == 0 {
            return Err(CircuitError::ModulusZero);
        }
        if self.max_gates == 0 || self.max_wires == 0 {
            return Err(CircuitError::InvalidConfig("max_gates and max_wires must be > 0".into()));
        }
        Ok(())
    }
}

impl fmt::Display for CircuitConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CircuitConfig(label={}, mod={:#x}, max_gates={}, max_wires={})",
            self.label, self.modulus, self.max_gates, self.max_wires)
    }
}

// ── Witness ─────────────────────────────────────────────────────

/// Holds wire assignments for a circuit execution.
#[derive(Debug, Clone)]
pub struct Witness {
    pub values: HashMap<usize, u64>,
    pub modulus: u64,
}

impl Witness {
    pub fn new(modulus: u64) -> Self {
        Self { values: HashMap::new(), modulus }
    }

    pub fn set(&mut self, wire: Wire, val: u64) {
        self.values.insert(wire.0, val % self.modulus);
    }

    pub fn get(&self, wire: Wire) -> Option<u64> {
        self.values.get(&wire.0).copied()
    }

    pub fn len(&self) -> usize { self.values.len() }
    pub fn is_empty(&self) -> bool { self.values.is_empty() }
}

impl fmt::Display for Witness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Witness({} wires, mod={})", self.values.len(), self.modulus)
    }
}

// ── ArithmeticCircuit ───────────────────────────────────────────

/// Complete arithmetic circuit with gates, wires, and constraint system.
#[derive(Debug, Clone)]
pub struct ArithmeticCircuit {
    config: CircuitConfig,
    gates: Vec<GateOp>,
    next_wire: usize,
    input_wires: Vec<Wire>,
    output_wires: Vec<Wire>,
    wire_labels: HashMap<String, Wire>,
}

impl ArithmeticCircuit {
    pub fn new(config: CircuitConfig) -> Result<Self, CircuitError> {
        config.validate()?;
        Ok(Self {
            config,
            gates: Vec::new(),
            next_wire: 1,  // Reserve wire 0 for constant one (R1CS)
            input_wires: Vec::new(),
            output_wires: Vec::new(),
            wire_labels: HashMap::new(),
        })
    }

    /// Allocate a fresh wire.
    pub fn alloc_wire(&mut self) -> Result<Wire, CircuitError> {
        if self.next_wire >= self.config.max_wires {
            return Err(CircuitError::InvalidWire("max wires reached".into()));
        }
        let w = Wire(self.next_wire);
        self.next_wire += 1;
        Ok(w)
    }

    /// Allocate a named input wire.
    pub fn alloc_input(&mut self, label: &str) -> Result<Wire, CircuitError> {
        if self.wire_labels.contains_key(label) {
            return Err(CircuitError::DuplicateLabel(label.into()));
        }
        let w = self.alloc_wire()?;
        self.input_wires.push(w);
        self.wire_labels.insert(label.to_string(), w);
        Ok(w)
    }

    /// Mark a wire as an output.
    pub fn mark_output(&mut self, w: Wire) {
        if !self.output_wires.contains(&w) {
            self.output_wires.push(w);
        }
    }

    /// Add an addition gate.
    pub fn add_gate(&mut self, left: Wire, right: Wire) -> Result<Wire, CircuitError> {
        let out = self.alloc_wire()?;
        self.gates.push(GateOp::Add { left, right, out });
        Ok(out)
    }

    /// Add a multiplication gate.
    pub fn mul_gate(&mut self, left: Wire, right: Wire) -> Result<Wire, CircuitError> {
        let out = self.alloc_wire()?;
        self.gates.push(GateOp::Mul { left, right, out });
        Ok(out)
    }

    /// Add a constant gate.
    pub fn const_gate(&mut self, value: u64) -> Result<Wire, CircuitError> {
        let out = self.alloc_wire()?;
        let v = value % self.config.modulus;
        self.gates.push(GateOp::Const { out, value: v });
        Ok(out)
    }

    /// Add a gate: out = wire + constant.
    pub fn add_const_gate(&mut self, left: Wire, value: u64) -> Result<Wire, CircuitError> {
        let out = self.alloc_wire()?;
        let v = value % self.config.modulus;
        self.gates.push(GateOp::AddConst { left, out, value: v });
        Ok(out)
    }

    /// Add a gate: out = wire * constant.
    pub fn mul_const_gate(&mut self, left: Wire, value: u64) -> Result<Wire, CircuitError> {
        let out = self.alloc_wire()?;
        let v = value % self.config.modulus;
        self.gates.push(GateOp::MulConst { left, out, value: v });
        Ok(out)
    }

    pub fn gate_count(&self) -> usize { self.gates.len() }
    pub fn wire_count(&self) -> usize { self.next_wire }
    pub fn inputs(&self) -> &[Wire] { &self.input_wires }
    pub fn outputs(&self) -> &[Wire] { &self.output_wires }
    pub fn gates(&self) -> &[GateOp] { &self.gates }
    pub fn modulus(&self) -> u64 { self.config.modulus }
    pub fn label(&self) -> &str { &self.config.label }

    /// Generate a witness by evaluating all gates given input assignments.
    pub fn generate_witness(&self, inputs: &HashMap<Wire, u64>) -> Result<Witness, CircuitError> {
        let m = self.config.modulus;
        let mut witness = Witness::new(m);

        witness.set(Wire(0), 1);  // Wire 0 is the constant one wire
        for (&w, &v) in inputs {
            witness.set(w, v);
        }

        for gate in &self.gates {
            match gate {
                GateOp::Add { left, right, out } => {
                    let a = witness.get(*left).ok_or_else(|| {
                        CircuitError::WitnessGenerationFailed(format!("missing {left}"))
                    })?;
                    let b = witness.get(*right).ok_or_else(|| {
                        CircuitError::WitnessGenerationFailed(format!("missing {right}"))
                    })?;
                    witness.set(*out, (a as u128 + b as u128).wrapping_rem(m as u128) as u64);
                }
                GateOp::Mul { left, right, out } => {
                    let a = witness.get(*left).ok_or_else(|| {
                        CircuitError::WitnessGenerationFailed(format!("missing {left}"))
                    })?;
                    let b = witness.get(*right).ok_or_else(|| {
                        CircuitError::WitnessGenerationFailed(format!("missing {right}"))
                    })?;
                    witness.set(*out, (a as u128 * b as u128).wrapping_rem(m as u128) as u64);
                }
                GateOp::Const { out, value } => {
                    witness.set(*out, *value);
                }
                GateOp::AddConst { left, out, value } => {
                    let a = witness.get(*left).ok_or_else(|| {
                        CircuitError::WitnessGenerationFailed(format!("missing {left}"))
                    })?;
                    witness.set(*out, (a as u128 + *value as u128).wrapping_rem(m as u128) as u64);
                }
                GateOp::MulConst { left, out, value } => {
                    let a = witness.get(*left).ok_or_else(|| {
                        CircuitError::WitnessGenerationFailed(format!("missing {left}"))
                    })?;
                    witness.set(*out, (a as u128 * *value as u128).wrapping_rem(m as u128) as u64);
                }
            }
        }

        Ok(witness)
    }

    /// Convert the circuit to R1CS constraints.
    pub fn to_r1cs(&self) -> Vec<R1csConstraint> {
        let mut constraints = Vec::new();

        for gate in &self.gates {
            match gate {
                GateOp::Mul { left, right, out } => {
                    // a * b = c  →  a=[left], b=[right], c=[out]
                    constraints.push(R1csConstraint {
                        a: vec![(left.0, 1)],
                        b: vec![(right.0, 1)],
                        c: vec![(out.0, 1)],
                    });
                }
                GateOp::Add { left, right, out } => {
                    // (left + right) * 1 = out
                    constraints.push(R1csConstraint {
                        a: vec![(left.0, 1), (right.0, 1)],
                        b: vec![(0, 1)], // wire-0 as constant 1 convention
                        c: vec![(out.0, 1)],
                    });
                }
                GateOp::Const { out, value } => {
                    // 1 * value = out  →  a=[1], b=[value*w0], c=[out]
                    constraints.push(R1csConstraint {
                        a: vec![(0, 1)],
                        b: vec![(0, *value)],
                        c: vec![(out.0, 1)],
                    });
                }
                GateOp::AddConst { left, out, value } => {
                    // (left + value) * 1 = out
                    constraints.push(R1csConstraint {
                        a: vec![(left.0, 1), (0, *value)],
                        b: vec![(0, 1)],
                        c: vec![(out.0, 1)],
                    });
                }
                GateOp::MulConst { left, out, value } => {
                    // left * value = out
                    constraints.push(R1csConstraint {
                        a: vec![(left.0, 1)],
                        b: vec![(0, *value)],
                        c: vec![(out.0, 1)],
                    });
                }
            }
        }

        constraints
    }

    /// Check satisfiability of the R1CS system against a witness.
    pub fn check_satisfiability(&self, witness: &Witness) -> Result<bool, CircuitError> {
        let constraints = self.to_r1cs();
        let m = self.config.modulus as u128;

        for (i, c) in constraints.iter().enumerate() {
            let eval_lc = |lc: &[(usize, u64)]| -> u128 {
                let mut sum: u128 = 0;
                for &(idx, coeff) in lc {
                    let w_val = if idx == 0 { 1u64 } else { witness.get(Wire(idx)).unwrap_or(0) };
                    sum = (sum + (coeff as u128 * w_val as u128) % m) % m;
                }
                sum
            };

            let a_val = eval_lc(&c.a);
            let b_val = eval_lc(&c.b);
            let c_val = eval_lc(&c.c);

            if (a_val * b_val) % m != c_val % m {
                return Err(CircuitError::UnsatisfiedConstraint(
                    format!("constraint {i}: {a_val}*{b_val} != {c_val} (mod {m})")
                ));
            }
        }

        Ok(true)
    }
}

impl fmt::Display for ArithmeticCircuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ArithmeticCircuit(label={}, gates={}, wires={}, inputs={}, outputs={})",
            self.config.label, self.gates.len(), self.next_wire,
            self.input_wires.len(), self.output_wires.len())
    }
}

// ── Circuit Builder (convenience wrapper) ───────────────────────

/// Fluent builder for common circuit patterns.
pub struct CircuitBuilder {
    circuit: ArithmeticCircuit,
}

impl CircuitBuilder {
    pub fn new(config: CircuitConfig) -> Result<Self, CircuitError> {
        Ok(Self { circuit: ArithmeticCircuit::new(config)? })
    }

    pub fn input(mut self, label: &str) -> Result<(Self, Wire), CircuitError> {
        let w = self.circuit.alloc_input(label)?;
        Ok((self, w))
    }

    pub fn constant(mut self, value: u64) -> Result<(Self, Wire), CircuitError> {
        let w = self.circuit.const_gate(value)?;
        Ok((self, w))
    }

    pub fn add(mut self, a: Wire, b: Wire) -> Result<(Self, Wire), CircuitError> {
        let w = self.circuit.add_gate(a, b)?;
        Ok((self, w))
    }

    pub fn mul(mut self, a: Wire, b: Wire) -> Result<(Self, Wire), CircuitError> {
        let w = self.circuit.mul_gate(a, b)?;
        Ok((self, w))
    }

    pub fn output(mut self, w: Wire) -> Self {
        self.circuit.mark_output(w);
        self
    }

    pub fn build(self) -> ArithmeticCircuit {
        self.circuit
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> CircuitConfig {
        CircuitConfig::new().with_modulus(101).with_label("test")
    }

    #[test]
    fn test_config_default() {
        let c = CircuitConfig::default();
        assert!(c.modulus > 0);
        assert_eq!(c.label, "circuit");
    }

    #[test]
    fn test_config_builder() {
        let c = CircuitConfig::new()
            .with_modulus(97)
            .with_max_gates(500)
            .with_max_wires(1000)
            .with_label("mycirc");
        assert_eq!(c.modulus, 97);
        assert_eq!(c.max_gates, 500);
        assert_eq!(c.label, "mycirc");
    }

    #[test]
    fn test_config_validate_zero_modulus() {
        let c = CircuitConfig::new().with_modulus(0);
        assert!(matches!(c.validate(), Err(CircuitError::ModulusZero)));
    }

    #[test]
    fn test_wire_display() {
        assert_eq!(format!("{}", Wire(42)), "w42");
    }

    #[test]
    fn test_alloc_input() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let w = circ.alloc_input("x").unwrap();
        assert_eq!(circ.inputs().len(), 1);
        assert_eq!(circ.inputs()[0], w);
    }

    #[test]
    fn test_duplicate_label() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        circ.alloc_input("x").unwrap();
        assert!(matches!(circ.alloc_input("x"), Err(CircuitError::DuplicateLabel(_))));
    }

    #[test]
    fn test_add_gate() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let _c = circ.add_gate(a, b).unwrap();
        assert_eq!(circ.gate_count(), 1);
    }

    #[test]
    fn test_mul_gate() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let _c = circ.mul_gate(a, b).unwrap();
        assert_eq!(circ.gate_count(), 1);
    }

    #[test]
    fn test_witness_generation_add() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let out = circ.add_gate(a, b).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, 30);
        inputs.insert(b, 50);
        let w = circ.generate_witness(&inputs).unwrap();
        assert_eq!(w.get(out), Some(80));
    }

    #[test]
    fn test_witness_generation_mul() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let out = circ.mul_gate(a, b).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, 7);
        inputs.insert(b, 13);
        let w = circ.generate_witness(&inputs).unwrap();
        assert_eq!(w.get(out), Some(91)); // 7*13=91 mod 101
    }

    #[test]
    fn test_witness_modular_arithmetic() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let out = circ.mul_gate(a, b).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, 50);
        inputs.insert(b, 50);
        let w = circ.generate_witness(&inputs).unwrap();
        assert_eq!(w.get(out), Some((50u64 * 50) % 101)); // 2500 % 101 = 76
    }

    #[test]
    fn test_const_gate() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let c = circ.const_gate(42).unwrap();
        let mut inputs = HashMap::new();
        let w = circ.generate_witness(&inputs).unwrap();
        assert_eq!(w.get(c), Some(42));
        let _ = inputs; // suppress unused
    }

    #[test]
    fn test_r1cs_conversion() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let _c = circ.mul_gate(a, b).unwrap();
        let constraints = circ.to_r1cs();
        assert_eq!(constraints.len(), 1);
    }

    #[test]
    fn test_satisfiability_pass() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let out = circ.mul_gate(a, b).unwrap();
        circ.mark_output(out);

        let mut inputs = HashMap::new();
        inputs.insert(a, 7);
        inputs.insert(b, 13);
        let w = circ.generate_witness(&inputs).unwrap();
        assert!(circ.check_satisfiability(&w).is_ok());
    }

    #[test]
    fn test_satisfiability_fail() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let b = circ.alloc_input("b").unwrap();
        let out = circ.mul_gate(a, b).unwrap();

        let mut w = Witness::new(101);
        w.set(a, 7);
        w.set(b, 13);
        w.set(out, 99); // wrong: should be 91
        assert!(circ.check_satisfiability(&w).is_err());
    }

    #[test]
    fn test_circuit_builder() {
        let builder = CircuitBuilder::new(small_config()).unwrap();
        let (builder, a) = builder.input("a").unwrap();
        let (builder, b) = builder.input("b").unwrap();
        let (builder, prod) = builder.mul(a, b).unwrap();
        let circ = builder.output(prod).build();
        assert_eq!(circ.gate_count(), 1);
        assert_eq!(circ.outputs().len(), 1);
    }

    #[test]
    fn test_add_const_gate() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let out = circ.add_const_gate(a, 10).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, 25);
        let w = circ.generate_witness(&inputs).unwrap();
        assert_eq!(w.get(out), Some(35));
    }

    #[test]
    fn test_mul_const_gate() {
        let mut circ = ArithmeticCircuit::new(small_config()).unwrap();
        let a = circ.alloc_input("a").unwrap();
        let out = circ.mul_const_gate(a, 3).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert(a, 20);
        let w = circ.generate_witness(&inputs).unwrap();
        assert_eq!(w.get(out), Some(60));
    }

    #[test]
    fn test_circuit_display() {
        let circ = ArithmeticCircuit::new(small_config()).unwrap();
        let s = format!("{circ}");
        assert!(s.contains("test"));
    }

    #[test]
    fn test_witness_display() {
        let w = Witness::new(101);
        assert!(format!("{w}").contains("0 wires"));
    }

    #[test]
    fn test_r1cs_constraint_display() {
        let c = R1csConstraint {
            a: vec![(1, 1)],
            b: vec![(2, 1)],
            c: vec![(3, 1)],
        };
        let s = format!("{c}");
        assert!(s.contains("1*w1"));
    }
}
