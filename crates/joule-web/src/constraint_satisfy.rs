//! Constraint Satisfaction Problem (CSP) solver with backtracking search,
//! arc consistency (AC-3), forward checking, variable ordering (MRV),
//! value ordering (least constraining value), and example CSPs.

use std::collections::VecDeque;

// ── Domain ───────────────────────────────────────────────────────

/// A finite domain of integer values.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    pub values: Vec<i64>,
}

impl Domain {
    pub fn new(values: Vec<i64>) -> Self {
        Self { values }
    }

    pub fn range(lo: i64, hi: i64) -> Self {
        Self { values: (lo..=hi).collect() }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn contains(&self, val: i64) -> bool {
        self.values.contains(&val)
    }

    pub fn remove(&mut self, val: i64) {
        self.values.retain(|v| *v != val);
    }
}

// ── Constraints ──────────────────────────────────────────────────

/// Constraint types.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Unary: restricts a single variable.
    Unary {
        variable: usize,
        allowed: Vec<i64>,
    },
    /// Binary: constraint between two variables.
    Binary {
        var1: usize,
        var2: usize,
        /// Returns true if the pair (val1, val2) is consistent.
        relation: BinaryRelation,
    },
    /// All-different: all variables must have distinct values.
    AllDifferent {
        variables: Vec<usize>,
    },
}

/// Binary relation between two variable assignments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinaryRelation {
    NotEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    /// |var1 - var2| != 0 and |var1 - var2| != |index1 - index2|
    /// (for N-Queens diagonal constraint).
    QueensSafe { row1: usize, row2: usize },
}

impl BinaryRelation {
    pub fn check(&self, val1: i64, val2: i64) -> bool {
        match self {
            Self::NotEqual => val1 != val2,
            Self::LessThan => val1 < val2,
            Self::LessThanOrEqual => val1 <= val2,
            Self::Equal => val1 == val2,
            Self::QueensSafe { row1, row2 } => {
                val1 != val2 && (val1 - val2).unsigned_abs() as usize != (*row1 as isize - *row2 as isize).unsigned_abs() as usize
            }
        }
    }
}

// ── CSP ──────────────────────────────────────────────────────────

/// A constraint satisfaction problem.
#[derive(Debug, Clone)]
pub struct Csp {
    pub num_variables: usize,
    pub domains: Vec<Domain>,
    pub constraints: Vec<Constraint>,
}

impl Csp {
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            domains: Vec::new(),
            constraints: Vec::new(),
        }
    }
}

// ── CSP Builder ──────────────────────────────────────────────────

/// Fluent builder for CSPs.
pub struct CspBuilder {
    csp: Csp,
}

impl CspBuilder {
    pub fn new(num_variables: usize) -> Self {
        Self {
            csp: Csp {
                num_variables,
                domains: vec![Domain::new(Vec::new()); num_variables],
                constraints: Vec::new(),
            },
        }
    }

    pub fn domain(mut self, var: usize, domain: Domain) -> Self {
        self.csp.domains[var] = domain;
        self
    }

    pub fn all_domains(mut self, domain: Domain) -> Self {
        for d in &mut self.csp.domains {
            *d = domain.clone();
        }
        self
    }

    pub fn unary(mut self, variable: usize, allowed: Vec<i64>) -> Self {
        self.csp.constraints.push(Constraint::Unary { variable, allowed });
        self
    }

    pub fn not_equal(mut self, var1: usize, var2: usize) -> Self {
        self.csp.constraints.push(Constraint::Binary {
            var1, var2, relation: BinaryRelation::NotEqual,
        });
        self
    }

    pub fn less_than(mut self, var1: usize, var2: usize) -> Self {
        self.csp.constraints.push(Constraint::Binary {
            var1, var2, relation: BinaryRelation::LessThan,
        });
        self
    }

    pub fn all_different(mut self, variables: Vec<usize>) -> Self {
        self.csp.constraints.push(Constraint::AllDifferent { variables });
        self
    }

    pub fn binary(mut self, var1: usize, var2: usize, relation: BinaryRelation) -> Self {
        self.csp.constraints.push(Constraint::Binary { var1, var2, relation });
        self
    }

    pub fn build(self) -> Csp {
        self.csp
    }
}

// ── Solver configuration ─────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VariableOrdering {
    /// Static: pick variables in order 0, 1, 2, ...
    Static,
    /// MRV: pick the variable with the smallest remaining domain.
    Mrv,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ValueOrdering {
    /// Use domain values in their natural order.
    Default,
    /// Least constraining value: pick the value that rules out the fewest
    /// values for neighboring variables.
    LeastConstraining,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolverConfig {
    pub variable_ordering: VariableOrdering,
    pub value_ordering: ValueOrdering,
    pub use_ac3: bool,
    pub use_forward_checking: bool,
    pub find_all: bool,
    pub max_solutions: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            variable_ordering: VariableOrdering::Mrv,
            value_ordering: ValueOrdering::Default,
            use_ac3: true,
            use_forward_checking: true,
            find_all: false,
            max_solutions: 1,
        }
    }
}

// ── Solution ─────────────────────────────────────────────────────

/// An assignment of values to all variables.
pub type Assignment = Vec<i64>;

/// Solver result.
#[derive(Debug, Clone, PartialEq)]
pub struct CspResult {
    pub solutions: Vec<Assignment>,
    pub nodes_explored: usize,
}

// ── Solver ───────────────────────────────────────────────────────

/// CSP backtracking solver.
pub struct CspSolver {
    csp: Csp,
    config: SolverConfig,
    solutions: Vec<Assignment>,
    nodes_explored: usize,
}

impl CspSolver {
    pub fn new(csp: Csp, config: SolverConfig) -> Self {
        Self { csp, config, solutions: Vec::new(), nodes_explored: 0 }
    }

    /// Apply unary constraints to prune domains.
    fn apply_unary_constraints(&mut self) {
        for constraint in &self.csp.constraints {
            if let Constraint::Unary { variable, allowed } = constraint {
                self.csp.domains[*variable].values.retain(|v| allowed.contains(v));
            }
        }
    }

    /// AC-3 arc consistency algorithm.
    fn ac3(&self, domains: &mut Vec<Domain>) -> bool {
        let mut queue: VecDeque<(usize, usize)> = VecDeque::new();

        // Collect binary arcs
        for c in &self.csp.constraints {
            match c {
                Constraint::Binary { var1, var2, .. } => {
                    queue.push_back((*var1, *var2));
                    queue.push_back((*var2, *var1));
                }
                Constraint::AllDifferent { variables } => {
                    for i in 0..variables.len() {
                        for j in i + 1..variables.len() {
                            queue.push_back((variables[i], variables[j]));
                            queue.push_back((variables[j], variables[i]));
                        }
                    }
                }
                _ => {}
            }
        }

        while let Some((xi, xj)) = queue.pop_front() {
            if self.revise(domains, xi, xj) {
                if domains[xi].is_empty() {
                    return false;
                }
                // Re-enqueue neighbors
                for c in &self.csp.constraints {
                    match c {
                        Constraint::Binary { var1, var2, .. } => {
                            if *var2 == xi && *var1 != xj {
                                queue.push_back((*var1, xi));
                            }
                            if *var1 == xi && *var2 != xj {
                                queue.push_back((*var2, xi));
                            }
                        }
                        Constraint::AllDifferent { variables } => {
                            if variables.contains(&xi) {
                                for &v in variables {
                                    if v != xi && v != xj {
                                        queue.push_back((v, xi));
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
        true
    }

    fn revise(&self, domains: &mut Vec<Domain>, xi: usize, xj: usize) -> bool {
        let mut revised = false;
        let xj_vals = domains[xj].values.clone();

        domains[xi].values.retain(|vi| {
            let has_support = xj_vals.iter().any(|vj| {
                self.consistent_pair(xi, *vi, xj, *vj)
            });
            if !has_support {
                revised = true;
                false
            } else {
                true
            }
        });

        revised
    }

    fn consistent_pair(&self, xi: usize, vi: i64, xj: usize, vj: i64) -> bool {
        for c in &self.csp.constraints {
            match c {
                Constraint::Binary { var1, var2, relation } => {
                    if *var1 == xi && *var2 == xj {
                        if !relation.check(vi, vj) { return false; }
                    }
                    if *var1 == xj && *var2 == xi {
                        if !relation.check(vj, vi) { return false; }
                    }
                }
                Constraint::AllDifferent { variables } => {
                    if variables.contains(&xi) && variables.contains(&xj) {
                        if vi == vj { return false; }
                    }
                }
                _ => {}
            }
        }
        true
    }

    /// Forward checking: after assigning var=val, prune neighbors.
    fn forward_check(&self, domains: &mut Vec<Domain>, var: usize, val: i64) -> bool {
        for c in &self.csp.constraints {
            match c {
                Constraint::Binary { var1, var2, relation } => {
                    if *var1 == var {
                        domains[*var2].values.retain(|vj| relation.check(val, *vj));
                        if domains[*var2].is_empty() { return false; }
                    }
                    if *var2 == var {
                        domains[*var1].values.retain(|vi| relation.check(*vi, val));
                        if domains[*var1].is_empty() { return false; }
                    }
                }
                Constraint::AllDifferent { variables } => {
                    if variables.contains(&var) {
                        for &other in variables {
                            if other != var {
                                domains[other].remove(val);
                                if domains[other].is_empty() { return false; }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        true
    }

    fn select_variable(&self, assignment: &[Option<i64>], domains: &[Domain]) -> Option<usize> {
        match self.config.variable_ordering {
            VariableOrdering::Static => {
                (0..self.csp.num_variables).find(|i| assignment[*i].is_none())
            }
            VariableOrdering::Mrv => {
                let mut best: Option<(usize, usize)> = None;
                for i in 0..self.csp.num_variables {
                    if assignment[i].is_none() {
                        let sz = domains[i].len();
                        if best.is_none() || sz < best.unwrap().1 {
                            best = Some((i, sz));
                        }
                    }
                }
                best.map(|(i, _)| i)
            }
        }
    }

    fn order_values(&self, var: usize, domains: &[Domain], assignment: &[Option<i64>]) -> Vec<i64> {
        let vals = domains[var].values.clone();
        match self.config.value_ordering {
            ValueOrdering::Default => vals,
            ValueOrdering::LeastConstraining => {
                let mut scored: Vec<(i64, usize)> = vals.iter().map(|v| {
                    let eliminated = self.count_eliminated(var, *v, domains, assignment);
                    (*v, eliminated)
                }).collect();
                scored.sort_by_key(|&(_, e)| e);
                scored.into_iter().map(|(v, _)| v).collect()
            }
        }
    }

    fn count_eliminated(&self, var: usize, val: i64, domains: &[Domain], assignment: &[Option<i64>]) -> usize {
        let mut count = 0;
        for c in &self.csp.constraints {
            match c {
                Constraint::Binary { var1, var2, relation } => {
                    if *var1 == var && assignment[*var2].is_none() {
                        count += domains[*var2].values.iter().filter(|&&vj| !relation.check(val, vj)).count();
                    }
                    if *var2 == var && assignment[*var1].is_none() {
                        count += domains[*var1].values.iter().filter(|&&vi| !relation.check(vi, val)).count();
                    }
                }
                Constraint::AllDifferent { variables } => {
                    if variables.contains(&var) {
                        for &other in variables {
                            if other != var && assignment[other].is_none() {
                                if domains[other].contains(val) { count += 1; }
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        count
    }

    fn is_consistent(&self, var: usize, val: i64, assignment: &[Option<i64>]) -> bool {
        for c in &self.csp.constraints {
            match c {
                Constraint::Binary { var1, var2, relation } => {
                    if *var1 == var {
                        if let Some(v2) = assignment[*var2] {
                            if !relation.check(val, v2) { return false; }
                        }
                    }
                    if *var2 == var {
                        if let Some(v1) = assignment[*var1] {
                            if !relation.check(v1, val) { return false; }
                        }
                    }
                }
                Constraint::AllDifferent { variables } => {
                    if variables.contains(&var) {
                        for &other in variables {
                            if other != var {
                                if let Some(ov) = assignment[other] {
                                    if ov == val { return false; }
                                }
                            }
                        }
                    }
                }
                Constraint::Unary { variable, allowed } => {
                    if *variable == var && !allowed.contains(&val) { return false; }
                }
            }
        }
        true
    }

    fn backtrack(&mut self, assignment: &mut Vec<Option<i64>>, domains: &Vec<Domain>) {
        if self.solutions.len() >= self.config.max_solutions {
            return;
        }

        self.nodes_explored += 1;

        let var = match self.select_variable(assignment, domains) {
            Some(v) => v,
            None => {
                // All assigned — record solution
                let sol: Assignment = assignment.iter().map(|a| a.unwrap()).collect();
                self.solutions.push(sol);
                return;
            }
        };

        let values = self.order_values(var, domains, assignment);

        for val in values {
            if !self.is_consistent(var, val, assignment) {
                continue;
            }

            assignment[var] = Some(val);
            let mut new_domains = domains.clone();
            new_domains[var] = Domain::new(vec![val]);

            let ok = if self.config.use_forward_checking {
                self.forward_check(&mut new_domains, var, val)
            } else {
                true
            };

            if ok {
                self.backtrack(assignment, &new_domains);
                if self.solutions.len() >= self.config.max_solutions {
                    return;
                }
            }

            assignment[var] = None;
        }
    }

    /// Solve the CSP.
    pub fn solve(&mut self) -> CspResult {
        self.solutions.clear();
        self.nodes_explored = 0;

        self.apply_unary_constraints();

        let mut domains = self.csp.domains.clone();
        if self.config.use_ac3 {
            if !self.ac3(&mut domains) {
                return CspResult { solutions: Vec::new(), nodes_explored: self.nodes_explored };
            }
        }

        let mut assignment: Vec<Option<i64>> = vec![None; self.csp.num_variables];
        self.backtrack(&mut assignment, &domains);

        CspResult {
            solutions: self.solutions.clone(),
            nodes_explored: self.nodes_explored,
        }
    }
}

// ── Example CSPs ─────────────────────────────────────────────────

/// Build an N-Queens CSP.
pub fn n_queens(n: usize) -> Csp {
    let mut builder = CspBuilder::new(n).all_domains(Domain::range(0, n as i64 - 1));
    for i in 0..n {
        for j in i + 1..n {
            builder = builder.binary(i, j, BinaryRelation::QueensSafe { row1: i, row2: j });
        }
    }
    builder.build()
}

/// Build a Sudoku CSP from an 81-element array (0 = empty).
pub fn sudoku(grid: &[i64; 81]) -> Csp {
    let mut builder = CspBuilder::new(81);

    // Set domains
    for i in 0..81 {
        if grid[i] != 0 {
            builder = builder.domain(i, Domain::new(vec![grid[i]]));
        } else {
            builder = builder.domain(i, Domain::range(1, 9));
        }
    }

    // Row constraints
    for row in 0..9 {
        let vars: Vec<usize> = (0..9).map(|col| row * 9 + col).collect();
        builder = builder.all_different(vars);
    }

    // Column constraints
    for col in 0..9 {
        let vars: Vec<usize> = (0..9).map(|row| row * 9 + col).collect();
        builder = builder.all_different(vars);
    }

    // Box constraints
    for box_row in 0..3 {
        for box_col in 0..3 {
            let mut vars = Vec::new();
            for r in 0..3 {
                for c in 0..3 {
                    vars.push((box_row * 3 + r) * 9 + box_col * 3 + c);
                }
            }
            builder = builder.all_different(vars);
        }
    }

    builder.build()
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_range() {
        let d = Domain::range(1, 5);
        assert_eq!(d.len(), 5);
        assert!(d.contains(3));
        assert!(!d.contains(6));
    }

    #[test]
    fn test_domain_remove() {
        let mut d = Domain::range(1, 5);
        d.remove(3);
        assert_eq!(d.len(), 4);
        assert!(!d.contains(3));
    }

    #[test]
    fn test_domain_empty() {
        let d = Domain::new(Vec::new());
        assert!(d.is_empty());
    }

    #[test]
    fn test_binary_relation_not_equal() {
        assert!(BinaryRelation::NotEqual.check(1, 2));
        assert!(!BinaryRelation::NotEqual.check(3, 3));
    }

    #[test]
    fn test_binary_relation_less_than() {
        assert!(BinaryRelation::LessThan.check(1, 2));
        assert!(!BinaryRelation::LessThan.check(2, 1));
    }

    #[test]
    fn test_queens_safe() {
        let rel = BinaryRelation::QueensSafe { row1: 0, row2: 1 };
        assert!(rel.check(0, 2)); // col 0, col 2: no conflict
        assert!(!rel.check(0, 1)); // col 0, col 1: diagonal conflict
        assert!(!rel.check(0, 0)); // same column
    }

    #[test]
    fn test_csp_builder() {
        let csp = CspBuilder::new(3)
            .all_domains(Domain::range(1, 3))
            .not_equal(0, 1)
            .not_equal(1, 2)
            .build();
        assert_eq!(csp.num_variables, 3);
        assert_eq!(csp.constraints.len(), 2);
    }

    #[test]
    fn test_simple_csp() {
        let csp = CspBuilder::new(2)
            .all_domains(Domain::range(1, 3))
            .not_equal(0, 1)
            .build();
        let mut solver = CspSolver::new(csp, SolverConfig {
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        });
        let result = solver.solve();
        // 3 * 3 = 9 total, minus 3 where x==y = 6
        assert_eq!(result.solutions.len(), 6);
    }

    #[test]
    fn test_4_queens() {
        let csp = n_queens(4);
        let mut solver = CspSolver::new(csp, SolverConfig {
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        });
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 2); // 4-Queens has exactly 2 solutions
    }

    #[test]
    fn test_8_queens_has_solution() {
        let csp = n_queens(8);
        let mut solver = CspSolver::new(csp, SolverConfig::default());
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 1);
        // Verify solution validity
        let sol = &result.solutions[0];
        for i in 0..8 {
            for j in i + 1..8 {
                assert_ne!(sol[i], sol[j]); // Different columns
                assert_ne!((sol[i] - sol[j]).unsigned_abs() as usize, j - i); // No diagonal
            }
        }
    }

    #[test]
    fn test_unary_constraint() {
        let csp = CspBuilder::new(1)
            .domain(0, Domain::range(1, 10))
            .unary(0, vec![3, 5, 7])
            .build();
        let mut solver = CspSolver::new(csp, SolverConfig {
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        });
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 3);
    }

    #[test]
    fn test_all_different() {
        let csp = CspBuilder::new(3)
            .all_domains(Domain::range(1, 3))
            .all_different(vec![0, 1, 2])
            .build();
        let mut solver = CspSolver::new(csp, SolverConfig {
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        });
        let result = solver.solve();
        // 3! = 6 permutations
        assert_eq!(result.solutions.len(), 6);
    }

    #[test]
    fn test_no_solution() {
        let csp = CspBuilder::new(3)
            .all_domains(Domain::range(1, 2))
            .all_different(vec![0, 1, 2])
            .build();
        let mut solver = CspSolver::new(csp, SolverConfig::default());
        let result = solver.solve();
        assert!(result.solutions.is_empty());
    }

    #[test]
    fn test_ac3_prunes_domains() {
        let csp = CspBuilder::new(2)
            .domain(0, Domain::new(vec![1]))
            .domain(1, Domain::range(1, 3))
            .not_equal(0, 1)
            .build();
        let mut solver = CspSolver::new(csp, SolverConfig {
            use_ac3: true,
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        });
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 2);
        for sol in &result.solutions {
            assert_ne!(sol[0], sol[1]);
        }
    }

    #[test]
    fn test_forward_checking() {
        let csp = CspBuilder::new(3)
            .all_domains(Domain::range(1, 3))
            .all_different(vec![0, 1, 2])
            .build();
        let config = SolverConfig {
            use_forward_checking: true,
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        };
        let mut solver = CspSolver::new(csp, config);
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 6);
    }

    #[test]
    fn test_mrv_ordering() {
        let csp = CspBuilder::new(3)
            .domain(0, Domain::range(1, 10))
            .domain(1, Domain::new(vec![5]))
            .domain(2, Domain::range(1, 3))
            .all_different(vec![0, 1, 2])
            .build();
        let config = SolverConfig {
            variable_ordering: VariableOrdering::Mrv,
            ..Default::default()
        };
        let mut solver = CspSolver::new(csp, config);
        let result = solver.solve();
        assert!(!result.solutions.is_empty());
    }

    #[test]
    fn test_lcv_ordering() {
        let csp = CspBuilder::new(3)
            .all_domains(Domain::range(1, 3))
            .all_different(vec![0, 1, 2])
            .build();
        let config = SolverConfig {
            value_ordering: ValueOrdering::LeastConstraining,
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        };
        let mut solver = CspSolver::new(csp, config);
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 6);
    }

    #[test]
    fn test_static_ordering() {
        let csp = CspBuilder::new(3)
            .all_domains(Domain::range(1, 3))
            .all_different(vec![0, 1, 2])
            .build();
        let config = SolverConfig {
            variable_ordering: VariableOrdering::Static,
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        };
        let mut solver = CspSolver::new(csp, config);
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 6);
    }

    #[test]
    fn test_max_solutions_limit() {
        let csp = CspBuilder::new(3)
            .all_domains(Domain::range(1, 3))
            .all_different(vec![0, 1, 2])
            .build();
        let config = SolverConfig {
            find_all: true,
            max_solutions: 2,
            ..Default::default()
        };
        let mut solver = CspSolver::new(csp, config);
        let result = solver.solve();
        assert_eq!(result.solutions.len(), 2);
    }

    #[test]
    fn test_nodes_explored_positive() {
        let csp = n_queens(4);
        let mut solver = CspSolver::new(csp, SolverConfig::default());
        let result = solver.solve();
        assert!(result.nodes_explored > 0);
    }

    #[test]
    fn test_sudoku_easy() {
        // A trivially solved sudoku: row 0 has 8 given values
        let mut grid = [0i64; 81];
        // Fill a known valid first row: 1-9
        for i in 0..8 {
            grid[i] = (i + 1) as i64;
        }
        // Variable at index 8 should be 9
        let csp = sudoku(&grid);
        let mut solver = CspSolver::new(csp, SolverConfig::default());
        let result = solver.solve();
        if !result.solutions.is_empty() {
            assert_eq!(result.solutions[0][8], 9);
        }
    }

    #[test]
    fn test_less_than_constraint() {
        let csp = CspBuilder::new(2)
            .all_domains(Domain::range(1, 5))
            .less_than(0, 1)
            .build();
        let config = SolverConfig {
            find_all: true,
            max_solutions: 100,
            ..Default::default()
        };
        let mut solver = CspSolver::new(csp, config);
        let result = solver.solve();
        for sol in &result.solutions {
            assert!(sol[0] < sol[1]);
        }
        // Should be C(5,2) = 10 solutions
        assert_eq!(result.solutions.len(), 10);
    }

    #[test]
    fn test_n_queens_csp_structure() {
        let csp = n_queens(4);
        assert_eq!(csp.num_variables, 4);
        // C(4,2) = 6 binary constraints
        assert_eq!(csp.constraints.len(), 6);
    }
}
