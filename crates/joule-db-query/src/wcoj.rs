//! Worst-Case Optimal Join (WCOJ) engine
//!
//! Implements Leapfrog TrieJoin for multi-way intersection of sorted relations.
//! This is critical for cyclic graph patterns (triangles, cliques) where
//! traditional binary hash-join cascades are exponentially slower.
//!
//! Based on "Leapfrog Triejoin: A Simple, Worst-Case Optimal Join Algorithm"
//! (Veldhuizen, ICDT 2014) and "Worst-Case Optimal Join Algorithms"
//! (Ngo, Porat, Ré, Rudra, PODS 2012).
//!
//! # Key insight
//!
//! Binary joins evaluate one join at a time, producing intermediate results that
//! can be exponentially large. WCOJ evaluates ALL joins simultaneously by
//! intersecting sorted iterators per variable, avoiding intermediate blowup.
//!
//! # Example: Triangle Query
//!
//! ```text
//! SELECT a.src, b.src, c.src
//! FROM edges a, edges b, edges c
//! WHERE a.dst = b.src AND b.dst = c.src AND c.dst = a.src
//! ```
//!
//! Binary join: O(|E|^3/2) intermediate tuples in worst case.
//! WCOJ: O(|E|^3/2) total work — no intermediate blowup.

use std::collections::BTreeSet;

/// A single tuple (row) in a relation, represented as a vector of values.
/// Values are byte-comparable (lexicographic ordering on the serialized form).
pub type Tuple = Vec<Vec<u8>>;

/// A relation is a set of tuples with named columns.
#[derive(Debug, Clone)]
pub struct Relation {
    /// Column names
    pub columns: Vec<String>,
    /// Sorted tuples (sorted lexicographically by the join variable ordering)
    pub tuples: Vec<Tuple>,
}

impl Relation {
    /// Create a new relation from columns and rows.
    pub fn new(columns: Vec<String>, mut tuples: Vec<Tuple>) -> Self {
        tuples.sort();
        tuples.dedup();
        Self { columns, tuples }
    }

    /// Get the column index for a given name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c == name)
    }

    /// Project this relation onto a subset of columns, reordered by the given variable order.
    /// Returns sorted, deduplicated tuples with only the requested columns.
    pub fn project_sorted(&self, var_order: &[String]) -> Vec<Tuple> {
        let indices: Vec<usize> = var_order
            .iter()
            .filter_map(|v| self.column_index(v))
            .collect();

        let mut result: Vec<Tuple> = self
            .tuples
            .iter()
            .map(|row| indices.iter().map(|&i| row[i].clone()).collect())
            .collect();

        result.sort();
        result.dedup();
        result
    }
}

/// An atom in a WCOJ query — a reference to a relation with variable bindings.
///
/// Example: `edges(X, Y)` means column 0 binds to variable X, column 1 to Y.
#[derive(Debug, Clone)]
pub struct Atom {
    /// Name of the source relation
    pub relation_name: String,
    /// Variable names bound to each column position
    pub variables: Vec<String>,
}

/// A WCOJ query: a conjunction of atoms (natural join).
///
/// Example: Triangle query:
///   edges(X, Y), edges(Y, Z), edges(Z, X)
#[derive(Debug, Clone)]
pub struct WcojQuery {
    /// The atoms to join
    pub atoms: Vec<Atom>,
    /// Output variables (columns to return)
    pub output_variables: Vec<String>,
}

impl WcojQuery {
    /// Compute a variable ordering for the leapfrog join.
    ///
    /// This is the frequency heuristic (most-constrained-first). It is
    /// retained as the **baseline** — `crate::wcoj_cost` layers a learned
    /// cost model on top that only deviates from this order when it
    /// predicts a strict improvement (Open Item §10.3). Delegates to
    /// [`WcojQuery::frequency_order`] so the heuristic lives in exactly
    /// one place.
    pub fn compute_variable_order(&self) -> Vec<String> {
        self.frequency_order()
    }
}

/// A sorted iterator over values for a single variable, given a prefix binding.
///
/// This is the core abstraction for Leapfrog TrieJoin. Each "trie iterator"
/// represents one relation's contribution to a variable's domain, filtered
/// by the current prefix of bound variables.
struct TrieIterator {
    /// Sorted distinct values for this variable given the current prefix
    values: Vec<Vec<u8>>,
    /// Current position in values
    pos: usize,
}

impl TrieIterator {
    fn new(values: Vec<Vec<u8>>) -> Self {
        Self { values, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.values.len()
    }

    fn key(&self) -> Option<&[u8]> {
        self.values.get(self.pos).map(|v| v.as_slice())
    }

    /// Advance to the first value >= target.
    fn seek(&mut self, target: &[u8]) {
        // Binary search for efficiency on large domains.
        let start = self.pos;
        match self.values[start..].binary_search_by(|v| v.as_slice().cmp(target)) {
            Ok(offset) => self.pos = start + offset,
            Err(offset) => self.pos = start + offset,
        }
    }

    fn next(&mut self) {
        if !self.at_end() {
            self.pos += 1;
        }
    }
}

/// Leapfrog intersection of multiple sorted iterators.
///
/// Finds the next value present in ALL iterators simultaneously.
/// This is the core of WCOJ — instead of joining two relations at a time,
/// we intersect all relations' contributions to each variable in one pass.
struct LeapfrogJoin {
    iterators: Vec<TrieIterator>,
}

impl LeapfrogJoin {
    fn new(iterators: Vec<TrieIterator>) -> Self {
        Self { iterators }
    }

    /// Find all values in the intersection of all iterators.
    fn intersect(&mut self) -> Vec<Vec<u8>> {
        if self.iterators.is_empty() {
            return Vec::new();
        }
        if self.iterators.iter().any(|it| it.at_end()) {
            return Vec::new();
        }
        if self.iterators.len() == 1 {
            // Single iterator — return all its values.
            return self.iterators[0]
                .values
                .iter()
                .skip(self.iterators[0].pos)
                .cloned()
                .collect();
        }

        let mut result = Vec::new();
        let k = self.iterators.len();

        // Sort iterators by their current key to establish the leapfrog invariant.
        // p = index of the iterator with the largest current key.
        loop {
            // Find max and min keys.
            let mut max_key: Option<Vec<u8>> = None;
            let mut min_idx = 0;
            let mut min_key: Option<Vec<u8>> = None;

            for (i, it) in self.iterators.iter().enumerate() {
                if it.at_end() {
                    return result;
                }
                let key = it.key().unwrap();
                match &max_key {
                    None => max_key = Some(key.to_vec()),
                    Some(mk) => {
                        if key > mk.as_slice() {
                            max_key = Some(key.to_vec());
                        }
                    }
                }
                match &min_key {
                    None => {
                        min_key = Some(key.to_vec());
                        min_idx = i;
                    }
                    Some(mk) => {
                        if key < mk.as_slice() {
                            min_key = Some(key.to_vec());
                            min_idx = i;
                        }
                    }
                }
            }

            let max = max_key.unwrap();
            let min = min_key.unwrap();

            if min == max {
                // All iterators agree — this value is in the intersection.
                result.push(min.clone());
                // Advance all iterators past this value.
                for it in &mut self.iterators {
                    it.next();
                }
            } else {
                // Seek the minimum iterator to the max key.
                self.iterators[min_idx].seek(&max);
            }
        }
    }
}

/// Build trie iterators for a given variable at a given depth in the variable
/// ordering, with the prefix of already-bound variables fixed.
fn build_trie_iterators(
    relations: &[(&Relation, &Atom)],
    variable: &str,
    var_order: &[String],
    depth: usize,
    prefix: &[Vec<u8>],
) -> Vec<TrieIterator> {
    let mut iterators = Vec::new();

    for &(relation, atom) in relations {
        // Check if this atom mentions the current variable.
        let var_col = match atom.variables.iter().position(|v| v == variable) {
            Some(col) => col,
            None => continue,
        };

        // Get the projected tuples for this relation in variable order.
        // Filter by prefix and extract the value at the current depth.
        let mut values: BTreeSet<Vec<u8>> = BTreeSet::new();

        for row in &relation.tuples {
            // Check prefix matches (all previously bound variables).
            let mut prefix_match = true;
            for d in 0..depth {
                let bound_var = &var_order[d];
                if let Some(col) = atom.variables.iter().position(|v| v == bound_var) {
                    if row[col] != prefix[d] {
                        prefix_match = false;
                        break;
                    }
                }
                // If this atom doesn't mention the variable, it doesn't constrain.
            }

            if prefix_match {
                values.insert(row[var_col].clone());
            }
        }

        if !values.is_empty() {
            iterators.push(TrieIterator::new(values.into_iter().collect()));
        }
    }

    iterators
}

/// Execute a WCOJ query using Leapfrog TrieJoin with the frequency
/// heuristic order. Back-compat entry point; discards work telemetry.
///
/// Returns the result as a set of tuples, one per output variable.
pub fn execute_wcoj(query: &WcojQuery, relations: &[(String, Relation)]) -> Vec<Tuple> {
    let var_order = query.compute_variable_order();
    let (results, _work) = execute_wcoj_with_order(query, relations, &var_order);
    results
}

/// Execute a WCOJ query with an explicit variable order, returning both
/// the results and the **leapfrog work** — the total recursive
/// `enumerate` calls plus intermediate bindings explored. Work is the
/// deterministic quantity `crate::wcoj_cost` trains its model to predict
/// and that a good variable order minimises.
pub fn execute_wcoj_with_order(
    query: &WcojQuery,
    relations: &[(String, Relation)],
    var_order: &[String],
) -> (Vec<Tuple>, u64) {
    let atom_relations: Vec<(&Relation, &Atom)> = query
        .atoms
        .iter()
        .filter_map(|atom| {
            relations
                .iter()
                .find(|(name, _)| name == &atom.relation_name)
                .map(|(_, rel)| (rel, atom))
        })
        .collect();

    if atom_relations.is_empty() {
        return (Vec::new(), 0);
    }

    let mut results = Vec::new();
    let mut prefix = Vec::new();
    let mut work: u64 = 0;

    enumerate(
        &atom_relations,
        var_order,
        0,
        &mut prefix,
        &mut results,
        &mut work,
    );

    // Project results to output variables.
    if query.output_variables.is_empty() || query.output_variables == var_order {
        return (results, work);
    }

    let output_indices: Vec<usize> = query
        .output_variables
        .iter()
        .filter_map(|v| var_order.iter().position(|vo| vo == v))
        .collect();

    let projected = results
        .into_iter()
        .map(|row| output_indices.iter().map(|&i| row[i].clone()).collect())
        .collect();
    (projected, work)
}

/// Execute a WCOJ query using the **learned cost model** to choose the
/// variable order, then fold the observed work back into the model so it
/// improves over time. Pass the same `model` across queries (persist it
/// with [`crate::wcoj_cost::WcojCostModel::save`]) to accumulate
/// learning. Closes Open Item §10.3.
pub fn execute_wcoj_learning(
    query: &WcojQuery,
    relations: &[(String, Relation)],
    model: &mut crate::wcoj_cost::WcojCostModel,
) -> Vec<Tuple> {
    // Feed observed cardinalities into the stats catalog first so the
    // feature map sees them.
    let cards: Vec<(String, u64)> = relations
        .iter()
        .map(|(n, r)| (n.clone(), r.tuples.len() as u64))
        .collect();
    model.observe_cardinalities(&cards);

    let order = model.best_order(query);
    let (results, work) = execute_wcoj_with_order(query, relations, &order);
    model.record(query, &order, work);
    results
}

/// Recursive enumeration with leapfrog intersection at each variable
/// depth. `work` accumulates the cost signal the learned model trains
/// on: one unit per recursive call plus one per intermediate binding
/// explored.
fn enumerate(
    atom_relations: &[(&Relation, &Atom)],
    var_order: &[String],
    depth: usize,
    prefix: &mut Vec<Vec<u8>>,
    results: &mut Vec<Tuple>,
    work: &mut u64,
) {
    *work = work.saturating_add(1);

    if depth >= var_order.len() {
        results.push(prefix.clone());
        return;
    }

    let variable = &var_order[depth];
    let iters = build_trie_iterators(atom_relations, variable, var_order, depth, prefix);

    if iters.is_empty() {
        // Variable not constrained by any atom — shouldn't happen in valid queries.
        return;
    }

    let mut join = LeapfrogJoin::new(iters);
    let matching_values = join.intersect();

    for value in matching_values {
        *work = work.saturating_add(1);
        prefix.push(value);
        enumerate(atom_relations, var_order, depth + 1, prefix, results, work);
        prefix.pop();
    }
}

// ---------------------------------------------------------------------------
// Convenience: detect cyclic join patterns
// ---------------------------------------------------------------------------

/// Check if a set of join conditions forms a cycle (e.g., triangle pattern).
/// Returns true if the join graph has a cycle.
pub fn is_cyclic_join(atoms: &[Atom]) -> bool {
    // Build adjacency from shared variables.
    let n = atoms.len();
    if n < 3 {
        return false;
    }

    let mut adj = vec![vec![]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let shared: Vec<&String> = atoms[i]
                .variables
                .iter()
                .filter(|v| atoms[j].variables.contains(v))
                .collect();
            if !shared.is_empty() {
                adj[i].push(j);
                adj[j].push(i);
            }
        }
    }

    // DFS cycle detection on undirected graph.
    let mut visited = vec![false; n];
    fn dfs(v: usize, parent: Option<usize>, adj: &[Vec<usize>], visited: &mut [bool]) -> bool {
        visited[v] = true;
        for &u in &adj[v] {
            if !visited[u] {
                if dfs(u, Some(v), adj, visited) {
                    return true;
                }
            } else if parent != Some(u) {
                return true;
            }
        }
        false
    }

    for i in 0..n {
        if !visited[i] && dfs(i, None, &adj, &mut visited) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn val(s: &str) -> Vec<u8> {
        s.as_bytes().to_vec()
    }

    fn edge(src: &str, dst: &str) -> Tuple {
        vec![val(src), val(dst)]
    }

    fn make_edges(edges: &[(&str, &str)]) -> Relation {
        let tuples: Vec<Tuple> = edges.iter().map(|(s, d)| edge(s, d)).collect();
        Relation::new(vec!["src".to_string(), "dst".to_string()], tuples)
    }

    #[test]
    fn test_triangle_query() {
        // Graph: 1→2, 2→3, 3→1, 1→3, 2→1, 3→2 (complete graph K3)
        let edges = make_edges(&[
            ("1", "2"),
            ("2", "3"),
            ("3", "1"),
            ("1", "3"),
            ("2", "1"),
            ("3", "2"),
        ]);

        // Triangle: edges(X,Y), edges(Y,Z), edges(Z,X)
        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["X".to_string(), "Y".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["Y".to_string(), "Z".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["Z".to_string(), "X".to_string()],
                },
            ],
            output_variables: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        };

        let relations = vec![("edges".to_string(), edges)];
        let results = execute_wcoj(&query, &relations);

        // K3 has 6 directed triangles (3! = 6 permutations of {1,2,3})
        assert_eq!(results.len(), 6);

        // Verify all results are valid triangles.
        let edge_set: BTreeSet<(Vec<u8>, Vec<u8>)> = [
            ("1", "2"),
            ("2", "3"),
            ("3", "1"),
            ("1", "3"),
            ("2", "1"),
            ("3", "2"),
        ]
        .iter()
        .map(|(s, d)| (val(s), val(d)))
        .collect();

        for row in &results {
            assert_eq!(row.len(), 3);
            let (x, y, z) = (&row[0], &row[1], &row[2]);
            assert!(edge_set.contains(&(x.clone(), y.clone())), "missing X→Y");
            assert!(edge_set.contains(&(y.clone(), z.clone())), "missing Y→Z");
            assert!(edge_set.contains(&(z.clone(), x.clone())), "missing Z→X");
        }
    }

    #[test]
    fn test_simple_join() {
        // Two-way join: R(X, Y) ⋈ S(Y, Z)
        let r = Relation::new(
            vec!["a".to_string(), "b".to_string()],
            vec![
                vec![val("1"), val("2")],
                vec![val("1"), val("3")],
                vec![val("2"), val("3")],
            ],
        );
        let s = Relation::new(
            vec!["c".to_string(), "d".to_string()],
            vec![
                vec![val("2"), val("4")],
                vec![val("3"), val("5")],
                vec![val("3"), val("6")],
            ],
        );

        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "R".to_string(),
                    variables: vec!["X".to_string(), "Y".to_string()],
                },
                Atom {
                    relation_name: "S".to_string(),
                    variables: vec!["Y".to_string(), "Z".to_string()],
                },
            ],
            output_variables: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        };

        let relations = vec![
            ("R".to_string(), r),
            ("S".to_string(), s),
        ];
        let mut results = execute_wcoj(&query, &relations);
        results.sort();

        // Expected: (1,2,4), (1,3,5), (1,3,6), (2,3,5), (2,3,6)
        assert_eq!(results.len(), 5);
        assert_eq!(results[0], vec![val("1"), val("2"), val("4")]);
        assert_eq!(results[1], vec![val("1"), val("3"), val("5")]);
        assert_eq!(results[2], vec![val("1"), val("3"), val("6")]);
        assert_eq!(results[3], vec![val("2"), val("3"), val("5")]);
        assert_eq!(results[4], vec![val("2"), val("3"), val("6")]);
    }

    #[test]
    fn test_four_clique() {
        // K4 complete graph: every vertex connects to every other.
        let nodes = ["a", "b", "c", "d"];
        let mut edges_data = Vec::new();
        for &s in &nodes {
            for &d in &nodes {
                if s != d {
                    edges_data.push((s, d));
                }
            }
        }
        let edges = make_edges(&edges_data);

        // 4-clique: edges(A,B), edges(A,C), edges(A,D), edges(B,C), edges(B,D), edges(C,D)
        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["A".to_string(), "B".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["A".to_string(), "C".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["A".to_string(), "D".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["B".to_string(), "C".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["B".to_string(), "D".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["C".to_string(), "D".to_string()],
                },
            ],
            output_variables: vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "D".to_string(),
            ],
        };

        let relations = vec![("edges".to_string(), edges)];
        let results = execute_wcoj(&query, &relations);

        // K4 has 4! = 24 directed 4-cliques.
        assert_eq!(results.len(), 24);
    }

    #[test]
    fn test_empty_intersection() {
        let r = Relation::new(
            vec!["a".to_string(), "b".to_string()],
            vec![vec![val("1"), val("2")]],
        );
        let s = Relation::new(
            vec!["c".to_string(), "d".to_string()],
            vec![vec![val("3"), val("4")]],
        );

        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "R".to_string(),
                    variables: vec!["X".to_string(), "Y".to_string()],
                },
                Atom {
                    relation_name: "S".to_string(),
                    variables: vec!["Y".to_string(), "Z".to_string()],
                },
            ],
            output_variables: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        };

        let relations = vec![
            ("R".to_string(), r),
            ("S".to_string(), s),
        ];
        let results = execute_wcoj(&query, &relations);
        assert!(results.is_empty());
    }

    #[test]
    fn test_self_join() {
        // Self-join: find all 2-hop paths
        let edges = make_edges(&[("a", "b"), ("b", "c"), ("c", "d"), ("a", "c")]);

        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["X".to_string(), "Y".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["Y".to_string(), "Z".to_string()],
                },
            ],
            output_variables: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        };

        let relations = vec![("edges".to_string(), edges)];
        let mut results = execute_wcoj(&query, &relations);
        results.sort();

        // 2-hop paths: a→b→c, b→c→d, a→c→d
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], vec![val("a"), val("b"), val("c")]);
        assert_eq!(results[1], vec![val("a"), val("c"), val("d")]);
        assert_eq!(results[2], vec![val("b"), val("c"), val("d")]);
    }

    #[test]
    fn test_is_cyclic_join() {
        // Triangle: edges(X,Y), edges(Y,Z), edges(Z,X) — cyclic
        let triangle_atoms = vec![
            Atom {
                relation_name: "e".to_string(),
                variables: vec!["X".to_string(), "Y".to_string()],
            },
            Atom {
                relation_name: "e".to_string(),
                variables: vec!["Y".to_string(), "Z".to_string()],
            },
            Atom {
                relation_name: "e".to_string(),
                variables: vec!["Z".to_string(), "X".to_string()],
            },
        ];
        assert!(is_cyclic_join(&triangle_atoms));

        // Chain: edges(X,Y), edges(Y,Z) — acyclic
        let chain_atoms = vec![
            Atom {
                relation_name: "e".to_string(),
                variables: vec!["X".to_string(), "Y".to_string()],
            },
            Atom {
                relation_name: "e".to_string(),
                variables: vec!["Y".to_string(), "Z".to_string()],
            },
        ];
        assert!(!is_cyclic_join(&chain_atoms));
    }

    #[test]
    fn test_triangle_count_larger_graph() {
        // Graph with known triangle count.
        // Vertices: 0..5, edges forming multiple triangles.
        let edges = make_edges(&[
            ("0", "1"),
            ("1", "0"),
            ("0", "2"),
            ("2", "0"),
            ("1", "2"),
            ("2", "1"),
            ("2", "3"),
            ("3", "2"),
            ("3", "4"),
            ("4", "3"),
            ("2", "4"),
            ("4", "2"),
        ]);

        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["X".to_string(), "Y".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["Y".to_string(), "Z".to_string()],
                },
                Atom {
                    relation_name: "edges".to_string(),
                    variables: vec!["Z".to_string(), "X".to_string()],
                },
            ],
            output_variables: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        };

        let relations = vec![("edges".to_string(), edges)];
        let results = execute_wcoj(&query, &relations);

        // Triangles: {0,1,2} (6 directed) + {2,3,4} (6 directed) = 12
        assert_eq!(results.len(), 12);
    }

    #[test]
    fn test_variable_ordering() {
        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "e".to_string(),
                    variables: vec!["X".to_string(), "Y".to_string()],
                },
                Atom {
                    relation_name: "e".to_string(),
                    variables: vec!["Y".to_string(), "Z".to_string()],
                },
                Atom {
                    relation_name: "e".to_string(),
                    variables: vec!["Z".to_string(), "X".to_string()],
                },
            ],
            output_variables: vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
        };

        let order = query.compute_variable_order();
        // All variables appear in 2 atoms each, so ordering is alphabetical (stable sort).
        assert_eq!(order.len(), 3);
        // X, Y, Z each appear 2 times — alphabetical tiebreak.
        assert_eq!(order, vec!["X", "Y", "Z"]);
    }

    #[test]
    fn test_single_atom_no_join() {
        let r = Relation::new(
            vec!["a".to_string(), "b".to_string()],
            vec![
                vec![val("1"), val("2")],
                vec![val("3"), val("4")],
            ],
        );

        let query = WcojQuery {
            atoms: vec![Atom {
                relation_name: "R".to_string(),
                variables: vec!["X".to_string(), "Y".to_string()],
            }],
            output_variables: vec!["X".to_string(), "Y".to_string()],
        };

        let relations = vec![("R".to_string(), r)];
        let results = execute_wcoj(&query, &relations);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_star_join() {
        // Star pattern: R(X, Y), S(X, Z), T(X, W) — all share X.
        let r = Relation::new(
            vec!["a".to_string(), "b".to_string()],
            vec![
                vec![val("1"), val("a")],
                vec![val("2"), val("b")],
            ],
        );
        let s = Relation::new(
            vec!["c".to_string(), "d".to_string()],
            vec![
                vec![val("1"), val("c")],
                vec![val("3"), val("d")],
            ],
        );
        let t = Relation::new(
            vec!["e".to_string(), "f".to_string()],
            vec![
                vec![val("1"), val("e")],
                vec![val("2"), val("f")],
            ],
        );

        let query = WcojQuery {
            atoms: vec![
                Atom {
                    relation_name: "R".to_string(),
                    variables: vec!["X".to_string(), "Y".to_string()],
                },
                Atom {
                    relation_name: "S".to_string(),
                    variables: vec!["X".to_string(), "Z".to_string()],
                },
                Atom {
                    relation_name: "T".to_string(),
                    variables: vec!["X".to_string(), "W".to_string()],
                },
            ],
            output_variables: vec![
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "W".to_string(),
            ],
        };

        let relations = vec![
            ("R".to_string(), r),
            ("S".to_string(), s),
            ("T".to_string(), t),
        ];
        let results = execute_wcoj(&query, &relations);

        // Only X=1 is in all three. Y=a, Z=c, W=e.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0], vec![val("1"), val("a"), val("c"), val("e")]);
    }

    // ── §10.3 learned-cost-model end-to-end validation ──────────────────

    /// A skewed triangle: e1 dense, e2 sparse, e3 dense. The variable
    /// order materially changes leapfrog work, but every order yields
    /// the same result set (the WCOJ correctness invariant).
    fn skewed_triangle() -> (WcojQuery, Vec<(String, Relation)>) {
        // e1: 0→{0..30}  (dense fan-out on X)
        let mut e1 = Vec::new();
        for x in 0..30 {
            for y in 0..30 {
                e1.push(edge(&x.to_string(), &y.to_string()));
            }
        }
        // e2: only a couple of Y→Z edges (sparse — the selective join).
        let e2 = vec![edge("5", "7"), edge("6", "8")];
        // e3: dense Z→X.
        let mut e3 = Vec::new();
        for z in 0..30 {
            for x in 0..30 {
                e3.push(edge(&z.to_string(), &x.to_string()));
            }
        }
        let query = WcojQuery {
            atoms: vec![
                Atom { relation_name: "e1".into(), variables: vec!["X".into(), "Y".into()] },
                Atom { relation_name: "e2".into(), variables: vec!["Y".into(), "Z".into()] },
                Atom { relation_name: "e3".into(), variables: vec!["Z".into(), "X".into()] },
            ],
            output_variables: vec!["X".into(), "Y".into(), "Z".into()],
        };
        let rels = vec![
            ("e1".to_string(), Relation::new(vec!["src".into(), "dst".into()], e1)),
            ("e2".to_string(), Relation::new(vec!["src".into(), "dst".into()], e2)),
            ("e3".to_string(), Relation::new(vec!["src".into(), "dst".into()], e3)),
        ];
        (query, rels)
    }

    #[test]
    fn order_is_correctness_invariant() {
        // Every permutation of the variable order must produce the same
        // result set — only the work differs.
        let (q, rels) = skewed_triangle();
        let orders = [
            vec!["X".to_string(), "Y".to_string(), "Z".to_string()],
            vec!["Y".to_string(), "Z".to_string(), "X".to_string()],
            vec!["Z".to_string(), "X".to_string(), "Y".to_string()],
        ];
        let mut canonical: Option<Vec<Tuple>> = None;
        let mut works = Vec::new();
        for o in &orders {
            let (mut res, work) = execute_wcoj_with_order(&q, &rels, o);
            res.sort();
            works.push(work);
            match &canonical {
                None => canonical = Some(res),
                Some(c) => assert_eq!(&res, c, "order {o:?} changed the result set"),
            }
        }
        // Sanity: the orders genuinely differ in cost (otherwise the
        // learned model has nothing to learn).
        assert!(
            works.iter().max() != works.iter().min(),
            "expected order to affect work; got {works:?}"
        );
    }

    #[test]
    fn learned_model_converges_and_never_regresses() {
        use crate::wcoj_cost::WcojCostModel;

        let (q, rels) = skewed_triangle();

        // Ground truth: the minimum work achievable across all orders,
        // and the frequency-baseline's work.
        let all_orders = {
            let mut vs = Vec::new();
            for a in &q.atoms {
                for v in &a.variables {
                    if !vs.contains(v) {
                        vs.push(v.clone());
                    }
                }
            }
            // 3 vars → 6 perms; brute-force the optimum.
            let mut perms = Vec::new();
            fn go(cur: &mut Vec<String>, rest: &[String], out: &mut Vec<Vec<String>>) {
                if rest.is_empty() {
                    out.push(cur.clone());
                    return;
                }
                for i in 0..rest.len() {
                    let mut r = rest.to_vec();
                    let picked = r.remove(i);
                    cur.push(picked);
                    go(cur, &r, out);
                    cur.pop();
                }
            }
            go(&mut Vec::new(), &vs, &mut perms);
            perms
        };
        let optimal_work = all_orders
            .iter()
            .map(|o| execute_wcoj_with_order(&q, &rels, o).1)
            .min()
            .unwrap();
        let baseline_order = q.frequency_order();
        let baseline_work = execute_wcoj_with_order(&q, &rels, &baseline_order).1;

        // Train the model by repeatedly running every order so it sees
        // the full cost surface (a real workload would sample this over
        // time; we compress it).
        let mut model = WcojCostModel::new();
        let cards: Vec<(String, u64)> = rels
            .iter()
            .map(|(n, r)| (n.clone(), r.tuples.len() as u64))
            .collect();
        model.observe_cardinalities(&cards);
        for _ in 0..150 {
            for o in &all_orders {
                let (_r, w) = execute_wcoj_with_order(&q, &rels, o);
                model.record(&q, o, w);
            }
        }

        // The model's chosen order, executed for real.
        let chosen = model.best_order(&q);
        let (chosen_res, chosen_work) = execute_wcoj_with_order(&q, &rels, &chosen);

        // 1. Correctness preserved.
        let (mut base_res, _) = execute_wcoj_with_order(&q, &rels, &baseline_order);
        let mut cr = chosen_res.clone();
        cr.sort();
        base_res.sort();
        assert_eq!(cr, base_res, "learned order changed the result set");

        // 2. No-regression floor: never worse than the frequency baseline.
        assert!(
            chosen_work <= baseline_work,
            "learned order regressed: chosen {chosen_work} > baseline {baseline_work}"
        );

        // 3. Convergence: the learned order is at (or very near) the
        //    optimum — strictly better than the baseline on this skewed
        //    workload, where the baseline is demonstrably suboptimal.
        assert!(baseline_work > optimal_work, "test setup: baseline should be suboptimal here");
        assert!(
            chosen_work <= optimal_work + (optimal_work / 20).max(1),
            "learned order did not converge: chosen {chosen_work}, optimal {optimal_work}, baseline {baseline_work}"
        );
    }
}
