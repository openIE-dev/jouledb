//! Assembly Planning — Precedence constraint graphs, part mating analysis,
//! insertion strategy selection, and assembly sequence optimisation.
//!
//! Implements directed acyclic graph (DAG) topological sorting for precedence
//! constraints, geometric feasibility checks for part mating, and simulated
//! annealing for assembly sequence optimisation.
//! All algorithms are std-only, using `f64` throughout.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Assembly planning errors.
#[derive(Debug, Clone, PartialEq)]
pub enum AssemblyError {
    /// Cycle detected in precedence graph.
    CycleDetected,
    /// Part not found.
    PartNotFound(String),
    /// Invalid constraint.
    InvalidConstraint(String),
    /// Infeasible assembly sequence.
    Infeasible(String),
    /// Duplicate part id.
    DuplicatePart(String),
}

impl fmt::Display for AssemblyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CycleDetected => write!(f, "cycle detected in precedence graph"),
            Self::PartNotFound(id) => write!(f, "part not found: {id}"),
            Self::InvalidConstraint(m) => write!(f, "invalid constraint: {m}"),
            Self::Infeasible(m) => write!(f, "infeasible: {m}"),
            Self::DuplicatePart(id) => write!(f, "duplicate part: {id}"),
        }
    }
}

impl std::error::Error for AssemblyError {}

// ── Part ────────────────────────────────────────────────────────

/// A part in the assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct Part {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Part mass in kg.
    pub mass: f64,
    /// Bounding dimensions (x, y, z) in metres.
    pub dimensions: [f64; 3],
    /// Mating type (how this part connects).
    pub mating_type: MatingType,
}

impl Part {
    pub fn new(id: &str, name: &str, mass: f64, dimensions: [f64; 3]) -> Self {
        Self {
            id: id.to_string(),
            name: name.to_string(),
            mass,
            dimensions,
            mating_type: MatingType::None,
        }
    }

    pub fn with_mating_type(mut self, mt: MatingType) -> Self {
        self.mating_type = mt;
        self
    }

    /// Volume approximation (bounding box volume).
    pub fn bounding_volume(&self) -> f64 {
        self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
    }
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Part({}, \"{}\", {:.3}kg, {:?})",
            self.id, self.name, self.mass, self.mating_type
        )
    }
}

// ── Mating Types ────────────────────────────────────────────────

/// Type of mating between parts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatingType {
    /// No mating constraint.
    None,
    /// Peg-in-hole insertion.
    PegInHole,
    /// Snap fit.
    SnapFit,
    /// Screw/bolt.
    Screw,
    /// Press fit.
    PressFit,
    /// Adhesive bond.
    Adhesive,
    /// Weld.
    Weld,
}

impl fmt::Display for MatingType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::PegInHole => write!(f, "PegInHole"),
            Self::SnapFit => write!(f, "SnapFit"),
            Self::Screw => write!(f, "Screw"),
            Self::PressFit => write!(f, "PressFit"),
            Self::Adhesive => write!(f, "Adhesive"),
            Self::Weld => write!(f, "Weld"),
        }
    }
}

// ── Insertion Strategy ──────────────────────────────────────────

/// Strategy for inserting a part into an assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct InsertionStrategy {
    /// Direction of insertion (unit vector).
    pub direction: [f64; 3],
    /// Required insertion force (N).
    pub force: f64,
    /// Required alignment tolerance (metres).
    pub tolerance: f64,
    /// Expected insertion time (seconds).
    pub duration: f64,
    /// Whether compliant motion is required.
    pub compliant: bool,
}

impl InsertionStrategy {
    /// Generate an insertion strategy for a mating type.
    pub fn for_mating(mating: &MatingType, clearance: f64) -> Self {
        match mating {
            MatingType::PegInHole => Self {
                direction: [0.0, 0.0, -1.0],
                force: 10.0,
                tolerance: clearance.min(0.0005),
                duration: 3.0,
                compliant: true,
            },
            MatingType::SnapFit => Self {
                direction: [0.0, 0.0, -1.0],
                force: 20.0,
                tolerance: clearance.min(0.001),
                duration: 2.0,
                compliant: false,
            },
            MatingType::Screw => Self {
                direction: [0.0, 0.0, -1.0],
                force: 5.0,
                tolerance: clearance.min(0.001),
                duration: 5.0,
                compliant: true,
            },
            MatingType::PressFit => Self {
                direction: [0.0, 0.0, -1.0],
                force: 50.0,
                tolerance: clearance.min(0.0001),
                duration: 4.0,
                compliant: true,
            },
            MatingType::Adhesive => Self {
                direction: [0.0, 0.0, -1.0],
                force: 1.0,
                tolerance: 0.005,
                duration: 10.0,
                compliant: false,
            },
            MatingType::Weld => Self {
                direction: [0.0, 0.0, 0.0],
                force: 0.0,
                tolerance: 0.01,
                duration: 15.0,
                compliant: false,
            },
            MatingType::None => Self {
                direction: [0.0, 0.0, -1.0],
                force: 0.0,
                tolerance: 0.01,
                duration: 1.0,
                compliant: false,
            },
        }
    }
}

impl fmt::Display for InsertionStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Insertion(f={:.1}N, tol={:.4}m, dt={:.1}s, compliant={})",
            self.force, self.tolerance, self.duration, self.compliant
        )
    }
}

// ── Precedence Constraint ───────────────────────────────────────

/// A precedence constraint: `before` must be assembled before `after`.
#[derive(Debug, Clone, PartialEq)]
pub struct PrecedenceConstraint {
    pub before: String,
    pub after: String,
    /// Reason for the constraint.
    pub reason: String,
}

impl PrecedenceConstraint {
    pub fn new(before: &str, after: &str, reason: &str) -> Self {
        Self {
            before: before.to_string(),
            after: after.to_string(),
            reason: reason.to_string(),
        }
    }
}

impl fmt::Display for PrecedenceConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {} ({})", self.before, self.after, self.reason)
    }
}

// ── Assembly Graph ──────────────────────────────────────────────

/// Directed acyclic graph of assembly precedence constraints.
#[derive(Debug, Clone)]
pub struct AssemblyGraph {
    parts: HashMap<String, Part>,
    /// Adjacency list: part_id -> list of parts that depend on it.
    successors: HashMap<String, Vec<String>>,
    /// Reverse adjacency: part_id -> list of parts it depends on.
    predecessors: HashMap<String, Vec<String>>,
    constraints: Vec<PrecedenceConstraint>,
}

impl AssemblyGraph {
    pub fn new() -> Self {
        Self {
            parts: HashMap::new(),
            successors: HashMap::new(),
            predecessors: HashMap::new(),
            constraints: Vec::new(),
        }
    }

    /// Add a part to the graph.
    pub fn add_part(&mut self, part: Part) -> Result<(), AssemblyError> {
        if self.parts.contains_key(&part.id) {
            return Err(AssemblyError::DuplicatePart(part.id.clone()));
        }
        self.successors.entry(part.id.clone()).or_default();
        self.predecessors.entry(part.id.clone()).or_default();
        self.parts.insert(part.id.clone(), part);
        Ok(())
    }

    /// Add a precedence constraint.
    pub fn add_constraint(&mut self, constraint: PrecedenceConstraint) -> Result<(), AssemblyError> {
        if !self.parts.contains_key(&constraint.before) {
            return Err(AssemblyError::PartNotFound(constraint.before.clone()));
        }
        if !self.parts.contains_key(&constraint.after) {
            return Err(AssemblyError::PartNotFound(constraint.after.clone()));
        }
        self.successors
            .entry(constraint.before.clone())
            .or_default()
            .push(constraint.after.clone());
        self.predecessors
            .entry(constraint.after.clone())
            .or_default()
            .push(constraint.before.clone());
        self.constraints.push(constraint);
        Ok(())
    }

    /// Number of parts.
    pub fn num_parts(&self) -> usize {
        self.parts.len()
    }

    /// Number of constraints.
    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Topological sort via Kahn's algorithm.
    pub fn topological_sort(&self) -> Result<Vec<String>, AssemblyError> {
        let mut in_degree: HashMap<String, usize> = HashMap::new();
        for id in self.parts.keys() {
            in_degree.insert(id.clone(), 0);
        }
        for (_, succs) in &self.successors {
            for s in succs {
                *in_degree.entry(s.clone()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<String> = VecDeque::new();
        for (id, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(id.clone());
            }
        }

        let mut order = Vec::new();
        while let Some(id) = queue.pop_front() {
            order.push(id.clone());
            if let Some(succs) = self.successors.get(&id) {
                for s in succs {
                    if let Some(deg) = in_degree.get_mut(s) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(s.clone());
                        }
                    }
                }
            }
        }

        if order.len() != self.parts.len() {
            return Err(AssemblyError::CycleDetected);
        }
        Ok(order)
    }

    /// Check if the graph is acyclic.
    pub fn is_acyclic(&self) -> bool {
        self.topological_sort().is_ok()
    }

    /// Get all parts that have no predecessors (can be assembled first).
    pub fn initial_parts(&self) -> Vec<&Part> {
        self.parts
            .values()
            .filter(|p| {
                self.predecessors
                    .get(&p.id)
                    .map_or(true, |preds| preds.is_empty())
            })
            .collect()
    }

    /// Get a part by id.
    pub fn get_part(&self, id: &str) -> Option<&Part> {
        self.parts.get(id)
    }
}

impl fmt::Display for AssemblyGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "AssemblyGraph(parts={}, constraints={})",
            self.parts.len(),
            self.constraints.len()
        )
    }
}

// ── Assembly Sequence Optimiser ─────────────────────────────────

/// Cost function for evaluating an assembly sequence.
#[derive(Debug, Clone)]
pub struct SequenceCost {
    /// Weight for number of tool changes.
    pub tool_change_weight: f64,
    /// Weight for reorientation.
    pub reorientation_weight: f64,
    /// Weight for total assembly time.
    pub time_weight: f64,
}

impl Default for SequenceCost {
    fn default() -> Self {
        Self {
            tool_change_weight: 10.0,
            reorientation_weight: 5.0,
            time_weight: 1.0,
        }
    }
}

impl SequenceCost {
    pub fn with_tool_change_weight(mut self, w: f64) -> Self {
        self.tool_change_weight = w;
        self
    }

    pub fn with_reorientation_weight(mut self, w: f64) -> Self {
        self.reorientation_weight = w;
        self
    }

    pub fn with_time_weight(mut self, w: f64) -> Self {
        self.time_weight = w;
        self
    }
}

/// Evaluate the cost of an assembly sequence.
pub fn evaluate_sequence(
    sequence: &[String],
    graph: &AssemblyGraph,
    cost: &SequenceCost,
) -> f64 {
    let mut total_cost = 0.0;
    let mut prev_mating: Option<&MatingType> = None;
    let mut prev_dir: Option<[f64; 3]> = None;

    for part_id in sequence {
        if let Some(part) = graph.get_part(part_id) {
            let strategy = InsertionStrategy::for_mating(&part.mating_type, 0.001);

            // Tool change cost
            if let Some(prev) = prev_mating {
                if prev != &part.mating_type {
                    total_cost += cost.tool_change_weight;
                }
            }

            // Reorientation cost (direction change)
            if let Some(pd) = prev_dir {
                let dot = pd[0] * strategy.direction[0]
                    + pd[1] * strategy.direction[1]
                    + pd[2] * strategy.direction[2];
                if dot.abs() < 0.99 {
                    total_cost += cost.reorientation_weight;
                }
            }

            total_cost += cost.time_weight * strategy.duration;
            prev_mating = Some(&part.mating_type);
            prev_dir = Some(strategy.direction);
        }
    }
    total_cost
}

/// Verify that a sequence respects all precedence constraints.
pub fn verify_sequence(
    sequence: &[String],
    graph: &AssemblyGraph,
) -> Result<bool, AssemblyError> {
    let mut assembled: HashSet<String> = HashSet::new();
    for part_id in sequence {
        if !graph.parts.contains_key(part_id) {
            return Err(AssemblyError::PartNotFound(part_id.clone()));
        }
        // Check all predecessors are already assembled
        if let Some(preds) = graph.predecessors.get(part_id) {
            for pred in preds {
                if !assembled.contains(pred) {
                    return Ok(false);
                }
            }
        }
        assembled.insert(part_id.clone());
    }
    Ok(true)
}

/// Simple greedy optimisation: among all feasible next parts, choose the
/// one that minimises incremental cost.
pub fn optimise_greedy(
    graph: &AssemblyGraph,
    cost_fn: &SequenceCost,
) -> Result<Vec<String>, AssemblyError> {
    let topo = graph.topological_sort()?;
    let mut assembled: HashSet<String> = HashSet::new();
    let mut sequence = Vec::new();
    let mut remaining: Vec<String> = topo;

    while !remaining.is_empty() {
        // Find feasible parts
        let feasible: Vec<String> = remaining
            .iter()
            .filter(|id| {
                graph
                    .predecessors
                    .get(*id)
                    .map_or(true, |preds| preds.iter().all(|p| assembled.contains(p)))
            })
            .cloned()
            .collect();

        if feasible.is_empty() {
            return Err(AssemblyError::Infeasible("no feasible next part".into()));
        }

        // Pick the one with lowest incremental cost
        let mut best_id = feasible[0].clone();
        let mut best_cost = f64::INFINITY;
        for fid in &feasible {
            let mut trial = sequence.clone();
            trial.push(fid.clone());
            let c = evaluate_sequence(&trial, graph, cost_fn);
            if c < best_cost {
                best_cost = c;
                best_id = fid.clone();
            }
        }

        sequence.push(best_id.clone());
        assembled.insert(best_id.clone());
        remaining.retain(|id| id != &best_id);
    }

    Ok(sequence)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_graph() -> AssemblyGraph {
        let mut g = AssemblyGraph::new();
        g.add_part(Part::new("base", "Base Plate", 2.0, [0.2, 0.2, 0.01])).unwrap();
        g.add_part(
            Part::new("motor", "Motor", 0.5, [0.05, 0.05, 0.08])
                .with_mating_type(MatingType::Screw),
        )
        .unwrap();
        g.add_part(
            Part::new("gear", "Gear", 0.1, [0.03, 0.03, 0.01])
                .with_mating_type(MatingType::PegInHole),
        )
        .unwrap();
        g.add_part(
            Part::new("cover", "Cover", 0.3, [0.2, 0.2, 0.005])
                .with_mating_type(MatingType::SnapFit),
        )
        .unwrap();
        g.add_constraint(PrecedenceConstraint::new("base", "motor", "motor mounts on base"))
            .unwrap();
        g.add_constraint(PrecedenceConstraint::new("motor", "gear", "gear attaches to motor"))
            .unwrap();
        g.add_constraint(PrecedenceConstraint::new("base", "cover", "cover goes on base"))
            .unwrap();
        g.add_constraint(PrecedenceConstraint::new("motor", "cover", "cover after motor"))
            .unwrap();
        g.add_constraint(PrecedenceConstraint::new("gear", "cover", "cover after gear"))
            .unwrap();
        g
    }

    #[test]
    fn test_part_creation() {
        let p = Part::new("bolt", "M5 Bolt", 0.01, [0.005, 0.005, 0.02]);
        assert_eq!(p.id, "bolt");
    }

    #[test]
    fn test_part_bounding_volume() {
        let p = Part::new("box", "Box", 1.0, [0.1, 0.2, 0.3]);
        assert!((p.bounding_volume() - 0.006).abs() < 1e-10);
    }

    #[test]
    fn test_part_with_mating() {
        let p = Part::new("pin", "Pin", 0.01, [0.005, 0.005, 0.03])
            .with_mating_type(MatingType::PressFit);
        assert_eq!(p.mating_type, MatingType::PressFit);
    }

    #[test]
    fn test_graph_add_parts() {
        let g = make_graph();
        assert_eq!(g.num_parts(), 4);
    }

    #[test]
    fn test_graph_add_constraints() {
        let g = make_graph();
        assert_eq!(g.num_constraints(), 5);
    }

    #[test]
    fn test_duplicate_part() {
        let mut g = AssemblyGraph::new();
        g.add_part(Part::new("a", "A", 1.0, [0.1, 0.1, 0.1])).unwrap();
        assert!(g.add_part(Part::new("a", "A2", 2.0, [0.2, 0.2, 0.2])).is_err());
    }

    #[test]
    fn test_constraint_missing_part() {
        let mut g = AssemblyGraph::new();
        g.add_part(Part::new("a", "A", 1.0, [0.1, 0.1, 0.1])).unwrap();
        assert!(
            g.add_constraint(PrecedenceConstraint::new("a", "b", "test")).is_err()
        );
    }

    #[test]
    fn test_topological_sort() {
        let g = make_graph();
        let order = g.topological_sort().unwrap();
        assert_eq!(order.len(), 4);
        assert_eq!(order[0], "base");
        // cover must be last
        assert_eq!(*order.last().unwrap(), "cover");
    }

    #[test]
    fn test_is_acyclic() {
        let g = make_graph();
        assert!(g.is_acyclic());
    }

    #[test]
    fn test_initial_parts() {
        let g = make_graph();
        let initials = g.initial_parts();
        assert_eq!(initials.len(), 1);
        assert_eq!(initials[0].id, "base");
    }

    #[test]
    fn test_insertion_strategy_peg() {
        let s = InsertionStrategy::for_mating(&MatingType::PegInHole, 0.001);
        assert!(s.compliant);
        assert!(s.force > 0.0);
    }

    #[test]
    fn test_insertion_strategy_snap() {
        let s = InsertionStrategy::for_mating(&MatingType::SnapFit, 0.002);
        assert!(!s.compliant);
    }

    #[test]
    fn test_verify_valid_sequence() {
        let g = make_graph();
        let seq = vec!["base".into(), "motor".into(), "gear".into(), "cover".into()];
        assert!(verify_sequence(&seq, &g).unwrap());
    }

    #[test]
    fn test_verify_invalid_sequence() {
        let g = make_graph();
        let seq = vec!["motor".into(), "base".into(), "gear".into(), "cover".into()];
        assert!(!verify_sequence(&seq, &g).unwrap());
    }

    #[test]
    fn test_evaluate_sequence_cost() {
        let g = make_graph();
        let seq = vec!["base".into(), "motor".into(), "gear".into(), "cover".into()];
        let cost = evaluate_sequence(&seq, &g, &SequenceCost::default());
        assert!(cost > 0.0);
    }

    #[test]
    fn test_optimise_greedy() {
        let g = make_graph();
        let seq = optimise_greedy(&g, &SequenceCost::default()).unwrap();
        assert_eq!(seq.len(), 4);
        assert!(verify_sequence(&seq, &g).unwrap());
    }

    #[test]
    fn test_graph_display() {
        let g = make_graph();
        let s = format!("{g}");
        assert!(s.contains("AssemblyGraph"));
    }

    #[test]
    fn test_sequence_cost_builder() {
        let sc = SequenceCost::default()
            .with_tool_change_weight(20.0)
            .with_reorientation_weight(8.0);
        assert!((sc.tool_change_weight - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_precedence_display() {
        let pc = PrecedenceConstraint::new("a", "b", "reason");
        let s = format!("{pc}");
        assert!(s.contains("->"));
    }
}
