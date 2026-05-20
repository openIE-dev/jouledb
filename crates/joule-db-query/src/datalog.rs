//! Datalog Query Language Parser
//!
//! Datalog is a declarative logic programming language for recursive queries.
//! It enables transitive closure, reachability, and rule-based inference that
//! no other query projection supports natively.
//!
//! # Syntax
//!
//! ```text
//! % Rules (head :- body)
//! reachable(X, Y) :- edge(X, Y).
//! reachable(X, Y) :- edge(X, Z), reachable(Z, Y).
//!
//! % Queries (prefix with ?-)
//! ?- reachable("a", X).
//!
//! % Facts (head with no body)
//! edge("a", "b").
//! edge("b", "c").
//!
//! % Negation (stratified)
//! not_reachable(X, Y) :- node(X), node(Y), not reachable(X, Y).
//!
//! % Aggregation
//! count_paths(X, count(Y)) :- reachable(X, Y).
//! ```

use crate::ast::{Expression, Operator, Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// A complete Datalog program: rules, facts, and queries.
#[derive(Debug, Clone)]
pub struct DatalogProgram {
    /// Rule definitions (head :- body)
    pub rules: Vec<DatalogRule>,
    /// Ground facts (head with no body)
    pub facts: Vec<DatalogAtom>,
    /// Queries (?- goal)
    pub queries: Vec<DatalogGoal>,
}

impl DatalogProgram {
    /// Convert to generic Query for unified AST compatibility.
    pub fn to_query(&self) -> Query {
        Query {
            query_type: QueryType::Traverse,
            source: None,
            columns: Vec::new(),
            filter: None,
            order_by: Vec::new(),
            group_by: Vec::new(),
            having: None,
            limit: None,
            offset: None,
            joins: Vec::new(),
            values: Vec::new(),
            returning: Vec::new(),
            ctes: Vec::new(),
            derived_columns: HashMap::new(),
            distinct: false,
            source_alias: None,
        }
    }
}

/// A Datalog rule: head :- body1, body2, ..., bodyN.
#[derive(Debug, Clone)]
pub struct DatalogRule {
    /// Head atom (what the rule derives)
    pub head: DatalogAtom,
    /// Body atoms (conditions that must hold)
    pub body: Vec<DatalogLiteral>,
}

/// A literal in a rule body — positive or negated atom, or a comparison.
#[derive(Debug, Clone)]
pub enum DatalogLiteral {
    /// Positive atom: predicate(args)
    Positive(DatalogAtom),
    /// Negated atom: not predicate(args)
    Negated(DatalogAtom),
    /// Comparison: X op Y (e.g., X > 5, X != Y)
    Comparison {
        left: DatalogTerm,
        op: ComparisonOp,
        right: DatalogTerm,
    },
    /// Aggregate: agg_func(var) — used in head, tracked in body for dependency
    Aggregate {
        function: String, // count, sum, min, max, avg
        variable: String,
    },
}

/// A Datalog atom: predicate(term1, term2, ..., termN)
#[derive(Debug, Clone)]
pub struct DatalogAtom {
    /// Predicate name (e.g., "reachable", "edge")
    pub predicate: String,
    /// Terms (arguments)
    pub terms: Vec<DatalogTerm>,
}

/// A term in a Datalog atom — variable or constant.
#[derive(Debug, Clone, PartialEq)]
pub enum DatalogTerm {
    /// Variable (uppercase identifier, e.g., X, Name)
    Variable(String),
    /// String constant
    String(String),
    /// Integer constant
    Integer(i64),
    /// Float constant
    Float(f64),
    /// Wildcard (_)
    Wildcard,
    /// Aggregate function call in head (e.g., count(X))
    Aggregate(String, String), // (function, variable)
}

/// Comparison operators for body constraints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonOp {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

/// A query goal: ?- atom1, atom2, ...
#[derive(Debug, Clone)]
pub struct DatalogGoal {
    pub literals: Vec<DatalogLiteral>,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

/// Datalog parser.
pub struct DatalogParser {
    input: Vec<char>,
    pos: usize,
}

impl DatalogParser {
    pub fn new() -> Self {
        Self {
            input: Vec::new(),
            pos: 0,
        }
    }

    /// Parse a Datalog program from input text.
    pub fn parse(&mut self, input: &str) -> QueryResult<DatalogProgram> {
        self.input = input.chars().collect();
        self.pos = 0;

        let mut rules = Vec::new();
        let mut facts = Vec::new();
        let mut queries = Vec::new();

        while self.pos < self.input.len() {
            self.skip_whitespace_and_comments();
            if self.pos >= self.input.len() {
                break;
            }

            if self.peek_str("?-") {
                // Query
                self.advance(2);
                self.skip_whitespace();
                let goal = self.parse_goal()?;
                queries.push(goal);
            } else {
                // Rule or fact: parse head first
                let head = self.parse_atom()?;
                self.skip_whitespace();

                if self.peek_str(":-") {
                    // Rule
                    self.advance(2);
                    self.skip_whitespace();
                    let body = self.parse_body()?;
                    rules.push(DatalogRule { head, body });
                } else {
                    // Fact (no body)
                    facts.push(head);
                }

                self.skip_whitespace();
                self.expect_char('.')?;
            }
        }

        Ok(DatalogProgram {
            rules,
            facts,
            queries,
        })
    }

    fn parse_atom(&mut self) -> QueryResult<DatalogAtom> {
        let predicate = self.parse_identifier()?;
        self.skip_whitespace();
        self.expect_char('(')?;

        let mut terms = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek_char() == Some(')') {
                self.advance(1);
                break;
            }
            if !terms.is_empty() {
                self.expect_char(',')?;
                self.skip_whitespace();
            }
            terms.push(self.parse_term()?);
        }

        Ok(DatalogAtom { predicate, terms })
    }

    fn parse_term(&mut self) -> QueryResult<DatalogTerm> {
        self.skip_whitespace();
        match self.peek_char() {
            Some('_') if !self.is_ident_continue_at(self.pos + 1) => {
                self.advance(1);
                Ok(DatalogTerm::Wildcard)
            }
            Some('"') => {
                // String constant
                self.advance(1);
                let mut s = String::new();
                while let Some(c) = self.peek_char() {
                    if c == '"' {
                        self.advance(1);
                        break;
                    }
                    if c == '\\' {
                        self.advance(1);
                        if let Some(escaped) = self.peek_char() {
                            s.push(escaped);
                            self.advance(1);
                        }
                    } else {
                        s.push(c);
                        self.advance(1);
                    }
                }
                Ok(DatalogTerm::String(s))
            }
            Some(c) if c.is_ascii_digit() || c == '-' => {
                // Number
                let mut num_str = String::new();
                if c == '-' {
                    num_str.push('-');
                    self.advance(1);
                }
                while let Some(c) = self.peek_char() {
                    if c.is_ascii_digit() {
                        num_str.push(c);
                        self.advance(1);
                    } else if c == '.' {
                        // Only consume '.' if followed by a digit (decimal point).
                        // Otherwise it's the statement terminator.
                        let next = self.input.get(self.pos + 1).copied();
                        if next.map(|n| n.is_ascii_digit()).unwrap_or(false) {
                            num_str.push(c);
                            self.advance(1);
                        } else {
                            break;
                        }
                    } else {
                        break;
                    }
                }
                if num_str.contains('.') {
                    let f: f64 = num_str.parse().map_err(|_| {
                        QueryError::ParseError(format!("invalid float: {}", num_str))
                    })?;
                    Ok(DatalogTerm::Float(f))
                } else {
                    let i: i64 = num_str.parse().map_err(|_| {
                        QueryError::ParseError(format!("invalid integer: {}", num_str))
                    })?;
                    Ok(DatalogTerm::Integer(i))
                }
            }
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.parse_identifier()?;
                self.skip_whitespace();

                // Check if this is an aggregate function call (e.g., count(X))
                if self.peek_char() == Some('(') {
                    let func = ident.to_lowercase();
                    if matches!(
                        func.as_str(),
                        "count" | "sum" | "min" | "max" | "avg"
                    ) {
                        self.advance(1); // (
                        self.skip_whitespace();
                        let var = self.parse_identifier()?;
                        self.skip_whitespace();
                        self.expect_char(')')?;
                        return Ok(DatalogTerm::Aggregate(func, var));
                    }
                }

                // Variable if starts with uppercase, constant otherwise
                if ident.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    Ok(DatalogTerm::Variable(ident))
                } else {
                    // Treat as string constant (unquoted atom)
                    Ok(DatalogTerm::String(ident))
                }
            }
            _ => Err(QueryError::ParseError(format!(
                "unexpected character at position {}: {:?}",
                self.pos,
                self.peek_char()
            ))),
        }
    }

    fn parse_body(&mut self) -> QueryResult<Vec<DatalogLiteral>> {
        let mut literals = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek_char() == Some('.') {
                break;
            }
            if !literals.is_empty() {
                self.expect_char(',')?;
                self.skip_whitespace();
            }
            literals.push(self.parse_literal()?);
        }
        Ok(literals)
    }

    fn parse_literal(&mut self) -> QueryResult<DatalogLiteral> {
        self.skip_whitespace();

        // Check for negation
        if self.peek_str("not ") || self.peek_str("not\t") || self.peek_str("\\+") {
            let skip = if self.peek_str("\\+") { 2 } else { 4 };
            self.advance(skip);
            self.skip_whitespace();
            let atom = self.parse_atom()?;
            return Ok(DatalogLiteral::Negated(atom));
        }

        // Try parsing as atom. If we hit a comparison operator before ',' or '.',
        // reinterpret as comparison.
        let start_pos = self.pos;

        // Check if it looks like a comparison: term op term
        // Heuristic: if we find an identifier NOT followed by '(' then check for op.
        let ident = self.parse_identifier()?;
        self.skip_whitespace();

        if self.peek_char() == Some('(') {
            // It's an atom — reparse from start
            self.pos = start_pos;
            let atom = self.parse_atom()?;
            return Ok(DatalogLiteral::Positive(atom));
        }

        // Check for comparison operator
        if let Some(op) = self.try_parse_comparison_op() {
            self.skip_whitespace();
            let left = if ident.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                DatalogTerm::Variable(ident)
            } else {
                DatalogTerm::String(ident)
            };
            let right = self.parse_term()?;
            return Ok(DatalogLiteral::Comparison { left, op, right });
        }

        // Fallback: reparse as atom
        self.pos = start_pos;
        let atom = self.parse_atom()?;
        Ok(DatalogLiteral::Positive(atom))
    }

    fn parse_goal(&mut self) -> QueryResult<DatalogGoal> {
        let mut literals = Vec::new();
        loop {
            self.skip_whitespace();
            if self.peek_char() == Some('.') {
                self.advance(1);
                break;
            }
            if self.pos >= self.input.len() {
                break;
            }
            if !literals.is_empty() {
                self.expect_char(',')?;
                self.skip_whitespace();
            }
            literals.push(self.parse_literal()?);
        }
        Ok(DatalogGoal { literals })
    }

    fn try_parse_comparison_op(&mut self) -> Option<ComparisonOp> {
        if self.peek_str("!=") {
            self.advance(2);
            Some(ComparisonOp::Neq)
        } else if self.peek_str(">=") {
            self.advance(2);
            Some(ComparisonOp::Gte)
        } else if self.peek_str("<=") {
            self.advance(2);
            Some(ComparisonOp::Lte)
        } else if self.peek_str("=") {
            self.advance(1);
            Some(ComparisonOp::Eq)
        } else if self.peek_str(">") {
            self.advance(1);
            Some(ComparisonOp::Gt)
        } else if self.peek_str("<") {
            self.advance(1);
            Some(ComparisonOp::Lt)
        } else {
            None
        }
    }

    // --- Helper methods ---

    fn parse_identifier(&mut self) -> QueryResult<String> {
        let mut ident = String::new();
        while let Some(c) = self.peek_char() {
            if c.is_alphanumeric() || c == '_' {
                ident.push(c);
                self.advance(1);
            } else {
                break;
            }
        }
        if ident.is_empty() {
            return Err(QueryError::ParseError(format!(
                "expected identifier at position {}",
                self.pos
            )));
        }
        Ok(ident)
    }

    fn skip_whitespace(&mut self) {
        while let Some(c) = self.peek_char() {
            if c.is_whitespace() {
                self.advance(1);
            } else {
                break;
            }
        }
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            self.skip_whitespace();
            if self.peek_char() == Some('%') {
                // Line comment
                while let Some(c) = self.peek_char() {
                    self.advance(1);
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn peek_char(&self) -> Option<char> {
        self.input.get(self.pos).copied()
    }

    fn peek_str(&self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        if self.pos + chars.len() > self.input.len() {
            return false;
        }
        for (i, c) in chars.iter().enumerate() {
            if self.input[self.pos + i] != *c {
                return false;
            }
        }
        true
    }

    fn advance(&mut self, n: usize) {
        self.pos += n;
    }

    fn expect_char(&mut self, expected: char) -> QueryResult<()> {
        match self.peek_char() {
            Some(c) if c == expected => {
                self.advance(1);
                Ok(())
            }
            other => Err(QueryError::ParseError(format!(
                "expected '{}' at position {}, got {:?}",
                expected, self.pos, other
            ))),
        }
    }

    fn is_ident_continue_at(&self, pos: usize) -> bool {
        self.input
            .get(pos)
            .map(|c| c.is_alphanumeric() || *c == '_')
            .unwrap_or(false)
    }
}

// ---------------------------------------------------------------------------
// Semi-naive Datalog evaluator
// ---------------------------------------------------------------------------

/// Result of evaluating a Datalog program.
#[derive(Debug, Clone)]
pub struct DatalogResult {
    /// Column names (from query variables)
    pub columns: Vec<String>,
    /// Result rows
    pub rows: Vec<Vec<Value>>,
    /// Number of iterations to reach fixpoint
    pub iterations: usize,
    /// Derived facts by predicate
    pub derived: HashMap<String, Vec<Vec<Value>>>,
}

/// Evaluate a Datalog program using semi-naive evaluation.
///
/// Semi-naive only considers NEW facts from the previous iteration
/// (delta) to derive new facts, avoiding redundant rederivation.
pub fn evaluate(program: &DatalogProgram) -> QueryResult<DatalogResult> {
    // Initialize EDB (Extensional Database) from facts.
    let mut db: HashMap<String, Vec<Vec<Value>>> = HashMap::new();

    for fact in &program.facts {
        let values: Vec<Value> = fact
            .terms
            .iter()
            .map(|t| term_to_value(t))
            .collect();
        db.entry(fact.predicate.clone())
            .or_default()
            .push(values);
    }

    // Compute strata for stratified negation.
    let strata = compute_strata(&program.rules);

    let mut iterations = 0;
    let max_iterations = 1000;

    // Fast dedup set: predicate → set of fact fingerprints (string repr for hashing).
    let mut seen: HashMap<String, HashSet<String>> = HashMap::new();

    fn fact_key(fact: &[Value]) -> String {
        fact.iter()
            .map(|v| format!("{:?}", v))
            .collect::<Vec<_>>()
            .join("|")
    }

    // Seed seen set from initial facts.
    for (pred, facts) in &db {
        let set = seen.entry(pred.clone()).or_default();
        for f in facts {
            set.insert(fact_key(f));
        }
    }

    // Evaluate each stratum in order.
    for stratum_rules in &strata {
        let mut delta: HashMap<String, Vec<Vec<Value>>> = HashMap::new();

        // Initial pass.
        for rule in stratum_rules {
            let new_facts = evaluate_rule(rule, &db, None);
            for fact in new_facts {
                let pred = &rule.head.predicate;
                let set = seen.entry(pred.clone()).or_default();
                if set.insert(fact_key(&fact)) {
                    delta.entry(pred.clone()).or_default().push(fact.clone());
                    db.entry(pred.clone()).or_default().push(fact);
                }
            }
        }

        // Semi-naive iteration.
        loop {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            let mut new_delta: HashMap<String, Vec<Vec<Value>>> = HashMap::new();
            let mut any_new = false;

            for rule in stratum_rules {
                let new_facts = evaluate_rule(rule, &db, Some(&delta));
                for fact in new_facts {
                    let pred = &rule.head.predicate;
                    let set = seen.entry(pred.clone()).or_default();
                    if set.insert(fact_key(&fact)) {
                        new_delta.entry(pred.clone()).or_default().push(fact.clone());
                        db.entry(pred.clone()).or_default().push(fact);
                        any_new = true;
                    }
                }
            }

            delta = new_delta;
            if !any_new {
                break;
            }
        }
    }

    // Execute queries.
    let mut result_columns = Vec::new();
    let mut result_rows = Vec::new();

    for query_goal in &program.queries {
        // For each positive literal in the query, look up the predicate.
        for lit in &query_goal.literals {
            if let DatalogLiteral::Positive(atom) = lit {
                // Determine output variable names.
                let mut columns: Vec<String> = Vec::new();
                let mut var_indices: Vec<(usize, String)> = Vec::new();

                for (i, term) in atom.terms.iter().enumerate() {
                    match term {
                        DatalogTerm::Variable(v) => {
                            columns.push(v.clone());
                            var_indices.push((i, v.clone()));
                        }
                        _ => {
                            columns.push(format!("_{}", i));
                        }
                    }
                }

                if let Some(facts) = db.get(&atom.predicate) {
                    for fact_row in facts {
                        // Check that constants match.
                        let mut matches = true;
                        for (i, term) in atom.terms.iter().enumerate() {
                            if i >= fact_row.len() {
                                matches = false;
                                break;
                            }
                            match term {
                                DatalogTerm::String(s) => {
                                    if fact_row[i] != Value::String(s.clone()) {
                                        matches = false;
                                    }
                                }
                                DatalogTerm::Integer(n) => {
                                    if fact_row[i] != Value::Int(*n) {
                                        matches = false;
                                    }
                                }
                                DatalogTerm::Variable(_) | DatalogTerm::Wildcard => {}
                                _ => {}
                            }
                        }
                        if matches {
                            result_rows.push(fact_row.clone());
                        }
                    }
                }

                result_columns = columns;
            }
        }
    }

    Ok(DatalogResult {
        columns: result_columns,
        rows: result_rows,
        iterations,
        derived: db,
    })
}

/// Evaluate a single rule against the database.
/// If `delta` is provided, use semi-naive: at least one body atom
/// must match from delta (not the full DB) to avoid redundant derivations.
fn evaluate_rule(
    rule: &DatalogRule,
    db: &HashMap<String, Vec<Vec<Value>>>,
    delta: Option<&HashMap<String, Vec<Vec<Value>>>>,
) -> Vec<Vec<Value>> {
    let positive_atoms: Vec<&DatalogAtom> = rule
        .body
        .iter()
        .filter_map(|l| match l {
            DatalogLiteral::Positive(a) => Some(a),
            _ => None,
        })
        .collect();

    let negated_atoms: Vec<&DatalogAtom> = rule
        .body
        .iter()
        .filter_map(|l| match l {
            DatalogLiteral::Negated(a) => Some(a),
            _ => None,
        })
        .collect();

    let comparisons: Vec<(&DatalogTerm, &ComparisonOp, &DatalogTerm)> = rule
        .body
        .iter()
        .filter_map(|l| match l {
            DatalogLiteral::Comparison { left, op, right } => Some((left, op, right)),
            _ => None,
        })
        .collect();

    if positive_atoms.is_empty() {
        return Vec::new();
    }

    // Semi-naive: if delta is provided, for each body atom position i,
    // substitute delta facts at position i and full DB facts elsewhere.
    // This ensures we only derive NEW facts based on what changed last iteration.
    // Without delta (initial pass), use full DB for all atoms.
    if let Some(delta_db) = delta {
        let mut bindings: Vec<HashMap<String, Value>> = Vec::new();

        for (i, _) in positive_atoms.iter().enumerate() {
            // Skip if this atom's predicate has no delta.
            if !delta_db.contains_key(&positive_atoms[i].predicate) {
                continue;
            }

            let mut round_bindings: Vec<HashMap<String, Value>> = vec![HashMap::new()];

            for (j, a2) in positive_atoms.iter().enumerate() {
                let facts = if i == j {
                    // Use delta for this position.
                    match delta_db.get(&a2.predicate) {
                        Some(f) => f,
                        None => continue,
                    }
                } else {
                    // Use full DB for other positions.
                    match db.get(&a2.predicate) {
                        Some(f) => f,
                        None => continue,
                    }
                };

                let mut new_bindings = Vec::new();
                for binding in &round_bindings {
                    for fact_row in facts {
                        if let Some(extended) = try_unify(a2, fact_row, binding) {
                            new_bindings.push(extended);
                        }
                    }
                }
                round_bindings = new_bindings;
            }

            for b in round_bindings {
                if !bindings.contains(&b) {
                    bindings.push(b);
                }
            }
        }

        // Fall through to negation/comparison/projection with these bindings.
        return apply_filters_and_project(rule, &bindings, &negated_atoms, &comparisons, db);
    }

    // Initial pass (no delta): join all atoms against full DB.
    let mut bindings: Vec<HashMap<String, Value>> = vec![HashMap::new()];

    for atom in &positive_atoms {
        let facts = match db.get(&atom.predicate) {
            Some(f) => f,
            None => continue,
        };

        let mut new_bindings = Vec::new();
        for binding in &bindings {
            for fact_row in facts {
                if let Some(extended) = try_unify(atom, fact_row, binding) {
                    new_bindings.push(extended);
                }
            }
        }
        bindings = new_bindings;
    }

    apply_filters_and_project(rule, &bindings, &negated_atoms, &comparisons, db)
}

/// Apply negation filters, comparison filters, and project bindings to head terms.
fn apply_filters_and_project(
    rule: &DatalogRule,
    bindings: &[HashMap<String, Value>],
    negated_atoms: &[&DatalogAtom],
    comparisons: &[(&DatalogTerm, &ComparisonOp, &DatalogTerm)],
    db: &HashMap<String, Vec<Vec<Value>>>,
) -> Vec<Vec<Value>> {
    let mut filtered: Vec<&HashMap<String, Value>> = bindings.iter().collect();

    // Apply negation.
    filtered.retain(|binding| {
        for neg_atom in negated_atoms {
            if let Some(neg_facts) = db.get(&neg_atom.predicate) {
                for fact_row in neg_facts {
                    if try_unify(neg_atom, fact_row, binding).is_some() {
                        return false;
                    }
                }
            }
        }
        true
    });

    // Apply comparisons.
    filtered.retain(|binding| {
        for (left, op, right) in comparisons {
            let lv = resolve_term(left, binding);
            let rv = resolve_term(right, binding);
            if let (Some(l), Some(r)) = (lv, rv) {
                if !compare_values(&l, op, &r) {
                    return false;
                }
            }
        }
        true
    });

    // Project bindings to head terms (with hash-based dedup).
    let mut results = Vec::new();
    let mut result_keys = HashSet::new();
    for binding in &filtered {
        let row: Vec<Value> = rule
            .head
            .terms
            .iter()
            .map(|term| match term {
                DatalogTerm::Variable(v) => {
                    binding.get(v).cloned().unwrap_or(Value::Null)
                }
                DatalogTerm::String(s) => Value::String(s.clone()),
                DatalogTerm::Integer(n) => Value::Int(*n),
                DatalogTerm::Float(f) => Value::Float(*f),
                DatalogTerm::Wildcard => Value::Null,
                DatalogTerm::Aggregate(_, _) => Value::Null,
            })
            .collect();

        let key: String = row.iter().map(|v| format!("{:?}", v)).collect::<Vec<_>>().join("|");
        if result_keys.insert(key) {
            results.push(row);
        }
    }

    results
}

/// Try to unify an atom's terms with a fact row, extending the given binding.
/// Returns None if unification fails (constant mismatch or variable conflict).
fn try_unify(
    atom: &DatalogAtom,
    fact_row: &[Value],
    binding: &HashMap<String, Value>,
) -> Option<HashMap<String, Value>> {
    if atom.terms.len() != fact_row.len() {
        return None;
    }

    let mut new_binding = binding.clone();

    for (term, value) in atom.terms.iter().zip(fact_row.iter()) {
        match term {
            DatalogTerm::Variable(v) => {
                if let Some(existing) = new_binding.get(v) {
                    if existing != value {
                        return None; // Variable already bound to different value.
                    }
                } else {
                    new_binding.insert(v.clone(), value.clone());
                }
            }
            DatalogTerm::String(s) => {
                if *value != Value::String(s.clone()) {
                    return None;
                }
            }
            DatalogTerm::Integer(n) => {
                if *value != Value::Int(*n) {
                    return None;
                }
            }
            DatalogTerm::Float(f) => {
                if *value != Value::Float(*f) {
                    return None;
                }
            }
            DatalogTerm::Wildcard => {} // matches anything
            DatalogTerm::Aggregate(_, _) => {} // skip
        }
    }

    Some(new_binding)
}

fn term_to_value(term: &DatalogTerm) -> Value {
    match term {
        DatalogTerm::String(s) => Value::String(s.clone()),
        DatalogTerm::Integer(n) => Value::Int(*n),
        DatalogTerm::Float(f) => Value::Float(*f),
        DatalogTerm::Variable(v) => Value::String(v.clone()),
        DatalogTerm::Wildcard => Value::Null,
        DatalogTerm::Aggregate(_, _) => Value::Null,
    }
}

fn resolve_term(term: &DatalogTerm, binding: &HashMap<String, Value>) -> Option<Value> {
    match term {
        DatalogTerm::Variable(v) => binding.get(v).cloned(),
        DatalogTerm::String(s) => Some(Value::String(s.clone())),
        DatalogTerm::Integer(n) => Some(Value::Int(*n)),
        DatalogTerm::Float(f) => Some(Value::Float(*f)),
        _ => None,
    }
}

fn compare_values(left: &Value, op: &ComparisonOp, right: &Value) -> bool {
    match (left, right) {
        (Value::Int(l), Value::Int(r)) => match op {
            ComparisonOp::Eq => l == r,
            ComparisonOp::Neq => l != r,
            ComparisonOp::Lt => l < r,
            ComparisonOp::Gt => l > r,
            ComparisonOp::Lte => l <= r,
            ComparisonOp::Gte => l >= r,
        },
        (Value::Float(l), Value::Float(r)) => match op {
            ComparisonOp::Eq => l == r,
            ComparisonOp::Neq => l != r,
            ComparisonOp::Lt => l < r,
            ComparisonOp::Gt => l > r,
            ComparisonOp::Lte => l <= r,
            ComparisonOp::Gte => l >= r,
        },
        (Value::String(l), Value::String(r)) => match op {
            ComparisonOp::Eq => l == r,
            ComparisonOp::Neq => l != r,
            ComparisonOp::Lt => l < r,
            ComparisonOp::Gt => l > r,
            ComparisonOp::Lte => l <= r,
            ComparisonOp::Gte => l >= r,
        },
        _ => false,
    }
}

/// Compute stratification for rules with negation.
/// Returns strata in evaluation order (lower strata first).
fn compute_strata(rules: &[DatalogRule]) -> Vec<Vec<&DatalogRule>> {
    // Simple stratification: rules without negation in stratum 0,
    // rules with negation whose negated predicates are all in lower strata go next.

    let mut strata: Vec<Vec<&DatalogRule>> = Vec::new();
    let mut assigned: HashMap<&str, usize> = HashMap::new();
    let mut remaining: Vec<&DatalogRule> = rules.iter().collect();

    // Stratum 0: rules with no negation
    let (no_neg, has_neg): (Vec<_>, Vec<_>) = remaining.into_iter().partition(|r| {
        !r.body
            .iter()
            .any(|l| matches!(l, DatalogLiteral::Negated(_)))
    });

    if !no_neg.is_empty() {
        for rule in &no_neg {
            assigned.insert(&rule.head.predicate, 0);
        }
        strata.push(no_neg);
    }

    remaining = has_neg;

    // Subsequent strata: assign rules whose negated predicates are in earlier strata.
    let mut stratum_idx = 1;
    let max_strata = 100;
    while !remaining.is_empty() && stratum_idx < max_strata {
        let mut can_assign = Vec::new();
        let mut cannot = Vec::new();

        for rule in remaining {
            let assignable = rule.body.iter().all(|l| {
                if let DatalogLiteral::Negated(atom) = l {
                    assigned.contains_key(atom.predicate.as_str())
                } else {
                    true
                }
            });
            if assignable {
                can_assign.push(rule);
            } else {
                cannot.push(rule);
            }
        }

        if can_assign.is_empty() {
            // Remaining rules form a cycle through negation — add them all.
            strata.push(cannot);
            remaining = Vec::new();
            break;
        }

        for rule in &can_assign {
            assigned.insert(&rule.head.predicate, stratum_idx);
        }
        strata.push(can_assign);
        remaining = cannot;
        stratum_idx += 1;
    }

    if !remaining.is_empty() {
        strata.push(remaining);
    }

    strata
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_facts() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(r#"edge("a", "b"). edge("b", "c")."#)
            .unwrap();
        assert_eq!(prog.facts.len(), 2);
        assert_eq!(prog.facts[0].predicate, "edge");
        assert_eq!(prog.facts[0].terms.len(), 2);
    }

    #[test]
    fn test_parse_rule() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse("reachable(X, Y) :- edge(X, Y).")
            .unwrap();
        assert_eq!(prog.rules.len(), 1);
        assert_eq!(prog.rules[0].head.predicate, "reachable");
        assert_eq!(prog.rules[0].body.len(), 1);
    }

    #[test]
    fn test_parse_recursive_rule() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                "reachable(X, Y) :- edge(X, Y).\n\
                 reachable(X, Y) :- edge(X, Z), reachable(Z, Y).",
            )
            .unwrap();
        assert_eq!(prog.rules.len(), 2);
        assert_eq!(prog.rules[1].body.len(), 2);
    }

    #[test]
    fn test_parse_query() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(r#"?- reachable("a", X)."#)
            .unwrap();
        assert_eq!(prog.queries.len(), 1);
        assert_eq!(prog.queries[0].literals.len(), 1);
    }

    #[test]
    fn test_parse_negation() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse("unreachable(X, Y) :- node(X), node(Y), not reachable(X, Y).")
            .unwrap();
        assert_eq!(prog.rules.len(), 1);
        let negated_count = prog.rules[0]
            .body
            .iter()
            .filter(|l| matches!(l, DatalogLiteral::Negated(_)))
            .count();
        assert_eq!(negated_count, 1);
    }

    #[test]
    fn test_parse_comparison() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse("big_edge(X, Y) :- edge(X, Y), Y > 5.")
            .unwrap();
        assert_eq!(prog.rules.len(), 1);
        let comp_count = prog.rules[0]
            .body
            .iter()
            .filter(|l| matches!(l, DatalogLiteral::Comparison { .. }))
            .count();
        assert_eq!(comp_count, 1);
    }

    #[test]
    fn test_parse_comments() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                "% This is a comment\n\
                 edge(\"a\", \"b\").\n\
                 % Another comment\n\
                 edge(\"b\", \"c\").",
            )
            .unwrap();
        assert_eq!(prog.facts.len(), 2);
    }

    #[test]
    fn test_evaluate_transitive_closure() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                r#"
                edge("a", "b").
                edge("b", "c").
                edge("c", "d").

                reachable(X, Y) :- edge(X, Y).
                reachable(X, Y) :- edge(X, Z), reachable(Z, Y).

                ?- reachable("a", X).
                "#,
            )
            .unwrap();

        let result = evaluate(&prog).unwrap();

        // From "a": can reach "b", "c", "d"
        assert_eq!(result.rows.len(), 3);
        let reached: Vec<&Value> = result.rows.iter().map(|r| &r[1]).collect();
        assert!(reached.contains(&&Value::String("b".to_string())));
        assert!(reached.contains(&&Value::String("c".to_string())));
        assert!(reached.contains(&&Value::String("d".to_string())));
    }

    #[test]
    fn test_evaluate_all_reachable() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                r#"
                edge("a", "b").
                edge("b", "c").

                reachable(X, Y) :- edge(X, Y).
                reachable(X, Y) :- edge(X, Z), reachable(Z, Y).

                ?- reachable(X, Y).
                "#,
            )
            .unwrap();

        let result = evaluate(&prog).unwrap();

        // (a,b), (b,c), (a,c) = 3 reachable pairs
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_evaluate_with_negation() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                r#"
                node("a").
                node("b").
                node("c").
                edge("a", "b").
                edge("b", "c").

                connected(X, Y) :- edge(X, Y).
                unconnected(X, Y) :- node(X), node(Y), not edge(X, Y).

                ?- unconnected(X, Y).
                "#,
            )
            .unwrap();

        let result = evaluate(&prog).unwrap();

        // Nodes: a, b, c. Edges: a→b, b→c.
        // Unconnected (direct): a→a, a→c, b→a, b→b, c→a, c→b, c→c = 7
        assert_eq!(result.rows.len(), 7);
    }

    #[test]
    fn test_evaluate_same_generation() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                r#"
                parent("alice", "bob").
                parent("alice", "carol").
                parent("bob", "dave").
                parent("carol", "eve").

                same_gen(X, Y) :- parent(Z, X), parent(Z, Y), X != Y.
                same_gen(X, Y) :- parent(Zx, X), parent(Zy, Y), same_gen(Zx, Zy), X != Y.

                ?- same_gen(X, Y).
                "#,
            )
            .unwrap();

        let result = evaluate(&prog).unwrap();

        // Gen 1: bob & carol (same parent alice)
        // Gen 2: dave & eve (parents bob & carol are same_gen)
        // Each pair counted twice (X,Y) and (Y,X)
        assert!(result.rows.len() >= 4); // (bob,carol), (carol,bob), (dave,eve), (eve,dave)
    }

    #[test]
    fn test_evaluate_with_constants_in_query() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                r#"
                edge("a", "b").
                edge("a", "c").
                edge("b", "d").

                ?- edge("a", X).
                "#,
            )
            .unwrap();

        let result = evaluate(&prog).unwrap();
        // Only edges from "a": (a,b) and (a,c)
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_evaluate_fixpoint_convergence() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                r#"
                edge("a", "b").
                edge("b", "a").

                reachable(X, Y) :- edge(X, Y).
                reachable(X, Y) :- edge(X, Z), reachable(Z, Y).

                ?- reachable(X, Y).
                "#,
            )
            .unwrap();

        let result = evaluate(&prog).unwrap();

        // Cycle: a→b, b→a. Reachable: (a,b), (b,a), (a,a), (b,b) = 4
        assert_eq!(result.rows.len(), 4);
        // Should converge quickly despite the cycle.
        assert!(result.iterations < 10);
    }

    #[test]
    fn test_evaluate_integers() {
        let mut parser = DatalogParser::new();
        let prog = parser
            .parse(
                r#"
                score("alice", 95).
                score("bob", 80).
                score("carol", 92).

                high_score(Name, S) :- score(Name, S), S > 90.

                ?- high_score(Name, S).
                "#,
            )
            .unwrap();

        let result = evaluate(&prog).unwrap();
        assert_eq!(result.rows.len(), 2); // alice(95) and carol(92)
    }
}
