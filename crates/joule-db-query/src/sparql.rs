//! SPARQL 1.1 Query Language Parser
//!
//! SPARQL is the standard query language for RDF data and knowledge graphs.
//! JouleDB maps SPARQL triple patterns directly to the amorphic HDC knowledge
//! core, enabling approximate matching that no other triple store supports.
//!
//! # Supported query forms
//!
//! ```text
//! PREFIX foaf: <http://xmlns.com/foaf/0.1/>
//!
//! SELECT ?name ?email
//! WHERE {
//!   ?person foaf:name ?name .
//!   ?person foaf:mbox ?email .
//!   OPTIONAL { ?person foaf:age ?age }
//!   FILTER (?age > 18)
//! }
//! ORDER BY ?name
//! LIMIT 10
//! ```

use crate::ast::{Expression, Operator, Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// A SPARQL query.
#[derive(Debug, Clone)]
pub struct SparqlQuery {
    /// PREFIX declarations.
    pub prefixes: HashMap<String, String>,
    /// Query form.
    pub form: SparqlForm,
    /// WHERE clause (graph pattern).
    pub pattern: GraphPattern,
    /// ORDER BY.
    pub order_by: Vec<(String, bool)>, // (var, descending)
    /// LIMIT.
    pub limit: Option<usize>,
    /// OFFSET.
    pub offset: Option<usize>,
}

impl SparqlQuery {
    pub fn to_query(&self) -> Query {
        Query {
            query_type: QueryType::Select,
            source: None,
            columns: match &self.form {
                SparqlForm::Select(vars) => vars.clone(),
                SparqlForm::SelectAll => Vec::new(),
                _ => Vec::new(),
            },
            filter: None,
            order_by: Vec::new(),
            group_by: Vec::new(),
            having: None,
            limit: self.limit,
            offset: self.offset,
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

/// Query form (SELECT, CONSTRUCT, ASK, DESCRIBE).
#[derive(Debug, Clone)]
pub enum SparqlForm {
    /// SELECT ?var1 ?var2 ...
    Select(Vec<String>),
    /// SELECT *
    SelectAll,
    /// ASK — returns boolean
    Ask,
    /// DESCRIBE ?var or DESCRIBE <uri>
    Describe(Vec<SparqlTerm>),
    /// CONSTRUCT { template } — returns triples
    Construct(Vec<TriplePattern>),
}

/// A graph pattern (the WHERE clause body).
#[derive(Debug, Clone)]
pub enum GraphPattern {
    /// Basic graph pattern: a set of triple patterns.
    Bgp(Vec<TriplePattern>),
    /// OPTIONAL { pattern }
    Optional(Box<GraphPattern>),
    /// UNION of two patterns.
    Union(Box<GraphPattern>, Box<GraphPattern>),
    /// FILTER(expression)
    Filter(SparqlExpr, Box<GraphPattern>),
    /// Sequence of patterns (AND / join).
    Join(Vec<GraphPattern>),
    /// BIND(expr AS ?var)
    Bind(SparqlExpr, String),
    /// Empty pattern.
    Empty,
}

/// A triple pattern: subject predicate object.
#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: SparqlTerm,
    pub predicate: SparqlTerm,
    pub object: SparqlTerm,
}

/// A term in a SPARQL query.
#[derive(Debug, Clone, PartialEq)]
pub enum SparqlTerm {
    /// Variable: ?name
    Variable(String),
    /// IRI: <http://...> or prefixed name
    Iri(String),
    /// String literal: "hello"
    Literal(String),
    /// Typed literal: "42"^^xsd:integer
    TypedLiteral(String, String),
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// Boolean literal
    Boolean(bool),
    /// Blank node: _:label
    BlankNode(String),
}

/// SPARQL expressions (for FILTER, BIND).
#[derive(Debug, Clone)]
pub enum SparqlExpr {
    /// Variable reference
    Var(String),
    /// Literal value
    Literal(SparqlTerm),
    /// Comparison: left op right
    Compare(Box<SparqlExpr>, CompareOp, Box<SparqlExpr>),
    /// Logical AND
    And(Box<SparqlExpr>, Box<SparqlExpr>),
    /// Logical OR
    Or(Box<SparqlExpr>, Box<SparqlExpr>),
    /// NOT
    Not(Box<SparqlExpr>),
    /// Function call: name(args)
    Function(String, Vec<SparqlExpr>),
    /// BOUND(?var)
    Bound(String),
    /// isIRI(?var)
    IsIri(Box<SparqlExpr>),
    /// STR(?var)
    Str(Box<SparqlExpr>),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareOp {
    Eq,
    Neq,
    Lt,
    Gt,
    Lte,
    Gte,
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct SparqlParser {
    input: Vec<char>,
    pos: usize,
}

impl SparqlParser {
    pub fn new() -> Self {
        Self {
            input: Vec::new(),
            pos: 0,
        }
    }

    pub fn parse(&mut self, input: &str) -> QueryResult<SparqlQuery> {
        self.input = input.chars().collect();
        self.pos = 0;

        let mut prefixes = HashMap::new();
        self.skip_ws();

        // Parse PREFIX declarations.
        while self.peek_kw("PREFIX") {
            self.advance_kw("PREFIX");
            self.skip_ws();
            let alias = self.read_until(':');
            self.expect(':')?;
            self.skip_ws();
            self.expect('<')?;
            let iri = self.read_until('>');
            self.expect('>')?;
            self.skip_ws();
            prefixes.insert(alias, iri);
        }

        // Parse query form.
        let form = self.parse_form()?;
        self.skip_ws();

        // Parse WHERE clause.
        let pattern = if self.peek_kw("WHERE") {
            self.advance_kw("WHERE");
            self.skip_ws();
            self.parse_group_pattern()?
        } else if self.peek('{') {
            self.parse_group_pattern()?
        } else {
            GraphPattern::Empty
        };

        self.skip_ws();

        // Parse solution modifiers.
        let mut order_by = Vec::new();
        let mut limit = None;
        let mut offset = None;

        while self.pos < self.input.len() {
            self.skip_ws();
            if self.peek_kw("ORDER") {
                self.advance_kw("ORDER");
                self.skip_ws();
                self.advance_kw("BY");
                self.skip_ws();
                while self.peek('?') || self.peek_kw("ASC") || self.peek_kw("DESC") {
                    let desc = if self.peek_kw("DESC") {
                        self.advance_kw("DESC");
                        self.skip_ws();
                        self.expect('(')?;
                        true
                    } else if self.peek_kw("ASC") {
                        self.advance_kw("ASC");
                        self.skip_ws();
                        self.expect('(')?;
                        false
                    } else {
                        false
                    };
                    self.skip_ws();
                    self.expect('?')?;
                    let var = self.read_ident();
                    if desc || self.peek(')') {
                        if self.peek(')') { self.advance(1); }
                    }
                    order_by.push((var, desc));
                    self.skip_ws();
                }
            } else if self.peek_kw("LIMIT") {
                self.advance_kw("LIMIT");
                self.skip_ws();
                limit = Some(self.read_number()? as usize);
            } else if self.peek_kw("OFFSET") {
                self.advance_kw("OFFSET");
                self.skip_ws();
                offset = Some(self.read_number()? as usize);
            } else {
                break;
            }
        }

        Ok(SparqlQuery {
            prefixes,
            form,
            pattern,
            order_by,
            limit,
            offset,
        })
    }

    fn parse_form(&mut self) -> QueryResult<SparqlForm> {
        if self.peek_kw("SELECT") {
            self.advance_kw("SELECT");
            self.skip_ws();
            if self.peek('*') {
                self.advance(1);
                Ok(SparqlForm::SelectAll)
            } else {
                let mut vars = Vec::new();
                while self.peek('?') {
                    self.advance(1);
                    vars.push(self.read_ident());
                    self.skip_ws();
                }
                Ok(SparqlForm::Select(vars))
            }
        } else if self.peek_kw("ASK") {
            self.advance_kw("ASK");
            Ok(SparqlForm::Ask)
        } else if self.peek_kw("DESCRIBE") {
            self.advance_kw("DESCRIBE");
            self.skip_ws();
            let mut terms = Vec::new();
            while self.peek('?') || self.peek('<') {
                terms.push(self.parse_term()?);
                self.skip_ws();
            }
            Ok(SparqlForm::Describe(terms))
        } else if self.peek_kw("CONSTRUCT") {
            self.advance_kw("CONSTRUCT");
            self.skip_ws();
            self.expect('{')?;
            let triples = self.parse_triple_patterns('}')?;
            self.expect('}')?;
            Ok(SparqlForm::Construct(triples))
        } else {
            Err(QueryError::ParseError(
                "expected SELECT, ASK, DESCRIBE, or CONSTRUCT".to_string(),
            ))
        }
    }

    fn parse_group_pattern(&mut self) -> QueryResult<GraphPattern> {
        self.expect('{')?;
        self.skip_ws();

        let mut patterns: Vec<GraphPattern> = Vec::new();
        let mut triples: Vec<TriplePattern> = Vec::new();

        while !self.peek('}') && self.pos < self.input.len() {
            self.skip_ws();
            if self.peek('}') {
                break;
            }

            if self.peek_kw("OPTIONAL") {
                // Flush pending triples.
                if !triples.is_empty() {
                    patterns.push(GraphPattern::Bgp(std::mem::take(&mut triples)));
                }
                self.advance_kw("OPTIONAL");
                self.skip_ws();
                let inner = self.parse_group_pattern()?;
                patterns.push(GraphPattern::Optional(Box::new(inner)));
            } else if self.peek_kw("FILTER") {
                if !triples.is_empty() {
                    patterns.push(GraphPattern::Bgp(std::mem::take(&mut triples)));
                }
                self.advance_kw("FILTER");
                self.skip_ws();
                let expr = self.parse_filter_expr()?;
                patterns.push(GraphPattern::Filter(
                    expr,
                    Box::new(GraphPattern::Empty),
                ));
            } else if self.peek_kw("UNION") {
                self.advance_kw("UNION");
                self.skip_ws();
                let right = self.parse_group_pattern()?;
                let left = if !triples.is_empty() {
                    GraphPattern::Bgp(std::mem::take(&mut triples))
                } else if let Some(p) = patterns.pop() {
                    p
                } else {
                    GraphPattern::Empty
                };
                patterns.push(GraphPattern::Union(Box::new(left), Box::new(right)));
            } else if self.peek_kw("BIND") {
                if !triples.is_empty() {
                    patterns.push(GraphPattern::Bgp(std::mem::take(&mut triples)));
                }
                self.advance_kw("BIND");
                self.skip_ws();
                self.expect('(')?;
                self.skip_ws();
                let expr = self.parse_sparql_expr()?;
                self.skip_ws();
                self.advance_kw("AS");
                self.skip_ws();
                self.expect('?')?;
                let var = self.read_ident();
                self.skip_ws();
                self.expect(')')?;
                patterns.push(GraphPattern::Bind(expr, var));
            } else {
                // Triple pattern.
                let tp = self.parse_one_triple()?;
                triples.push(tp);
                self.skip_ws();
                // Consume optional '.'
                if self.peek('.') {
                    self.advance(1);
                }
            }
            self.skip_ws();
        }

        self.expect('}')?;

        if !triples.is_empty() {
            patterns.push(GraphPattern::Bgp(triples));
        }

        if patterns.is_empty() {
            Ok(GraphPattern::Empty)
        } else if patterns.len() == 1 {
            Ok(patterns.remove(0))
        } else {
            Ok(GraphPattern::Join(patterns))
        }
    }

    fn parse_one_triple(&mut self) -> QueryResult<TriplePattern> {
        self.skip_ws();
        let subject = self.parse_term()?;
        self.skip_ws();
        let predicate = self.parse_term()?;
        self.skip_ws();
        let object = self.parse_term()?;
        Ok(TriplePattern {
            subject,
            predicate,
            object,
        })
    }

    fn parse_triple_patterns(&mut self, end: char) -> QueryResult<Vec<TriplePattern>> {
        let mut triples = Vec::new();
        self.skip_ws();
        while !self.peek(end) && self.pos < self.input.len() {
            triples.push(self.parse_one_triple()?);
            self.skip_ws();
            if self.peek('.') {
                self.advance(1);
            }
            self.skip_ws();
        }
        Ok(triples)
    }

    fn parse_term(&mut self) -> QueryResult<SparqlTerm> {
        self.skip_ws();
        if self.peek('?') {
            self.advance(1);
            Ok(SparqlTerm::Variable(self.read_ident()))
        } else if self.peek('<') {
            self.advance(1);
            let iri = self.read_until('>');
            self.expect('>')?;
            Ok(SparqlTerm::Iri(iri))
        } else if self.peek('"') {
            self.advance(1);
            let lit = self.read_until('"');
            self.expect('"')?;
            // Check for typed literal.
            if self.peek_str("^^") {
                self.advance(2);
                let dtype = if self.peek('<') {
                    self.advance(1);
                    let t = self.read_until('>');
                    self.expect('>')?;
                    t
                } else {
                    self.read_ident()
                };
                Ok(SparqlTerm::TypedLiteral(lit, dtype))
            } else {
                Ok(SparqlTerm::Literal(lit))
            }
        } else if self.peek('_') && self.peek_at(1) == Some(':') {
            self.advance(2);
            Ok(SparqlTerm::BlankNode(self.read_ident()))
        } else if self.cur().map(|c| c.is_ascii_digit() || c == '-').unwrap_or(false) {
            let num_str = self.read_number_str();
            if num_str.contains('.') {
                Ok(SparqlTerm::Float(num_str.parse().map_err(|_| {
                    QueryError::ParseError(format!("invalid float: {}", num_str))
                })?))
            } else {
                Ok(SparqlTerm::Integer(num_str.parse().map_err(|_| {
                    QueryError::ParseError(format!("invalid integer: {}", num_str))
                })?))
            }
        } else if self.peek_kw("true") {
            self.advance(4);
            Ok(SparqlTerm::Boolean(true))
        } else if self.peek_kw("false") {
            self.advance(5);
            Ok(SparqlTerm::Boolean(false))
        } else {
            // Prefixed name: prefix:local
            let prefix = self.read_ident();
            if self.peek(':') {
                self.advance(1);
                let local = self.read_ident();
                Ok(SparqlTerm::Iri(format!("{}:{}", prefix, local)))
            } else {
                Ok(SparqlTerm::Iri(prefix))
            }
        }
    }

    fn parse_filter_expr(&mut self) -> QueryResult<SparqlExpr> {
        self.skip_ws();
        if self.peek('(') {
            self.advance(1);
            let expr = self.parse_sparql_expr()?;
            self.skip_ws();
            self.expect(')')?;
            Ok(expr)
        } else {
            self.parse_sparql_expr()
        }
    }

    fn parse_sparql_expr(&mut self) -> QueryResult<SparqlExpr> {
        self.skip_ws();
        let left = self.parse_sparql_primary()?;
        self.skip_ws();

        // Check for comparison operator.
        if let Some(op) = self.try_compare_op() {
            self.skip_ws();
            let right = self.parse_sparql_primary()?;
            return Ok(SparqlExpr::Compare(Box::new(left), op, Box::new(right)));
        }

        if self.peek_str("&&") {
            self.advance(2);
            self.skip_ws();
            let right = self.parse_sparql_expr()?;
            return Ok(SparqlExpr::And(Box::new(left), Box::new(right)));
        }
        if self.peek_str("||") {
            self.advance(2);
            self.skip_ws();
            let right = self.parse_sparql_expr()?;
            return Ok(SparqlExpr::Or(Box::new(left), Box::new(right)));
        }

        Ok(left)
    }

    fn parse_sparql_primary(&mut self) -> QueryResult<SparqlExpr> {
        self.skip_ws();
        if self.peek('?') {
            self.advance(1);
            Ok(SparqlExpr::Var(self.read_ident()))
        } else if self.peek('!') {
            self.advance(1);
            let inner = self.parse_sparql_primary()?;
            Ok(SparqlExpr::Not(Box::new(inner)))
        } else if self.peek_kw("BOUND") {
            self.advance(5);
            self.skip_ws();
            self.expect('(')?;
            self.skip_ws();
            self.expect('?')?;
            let var = self.read_ident();
            self.skip_ws();
            self.expect(')')?;
            Ok(SparqlExpr::Bound(var))
        } else if self.peek('(') {
            self.advance(1);
            let expr = self.parse_sparql_expr()?;
            self.skip_ws();
            self.expect(')')?;
            Ok(expr)
        } else {
            let term = self.parse_term()?;
            Ok(SparqlExpr::Literal(term))
        }
    }

    fn try_compare_op(&mut self) -> Option<CompareOp> {
        if self.peek_str("!=") { self.advance(2); Some(CompareOp::Neq) }
        else if self.peek_str(">=") { self.advance(2); Some(CompareOp::Gte) }
        else if self.peek_str("<=") { self.advance(2); Some(CompareOp::Lte) }
        else if self.peek('=') { self.advance(1); Some(CompareOp::Eq) }
        else if self.peek('>') { self.advance(1); Some(CompareOp::Gt) }
        else if self.peek('<') && !self.peek_at(1).map(|c| c.is_alphabetic() || c == '/').unwrap_or(false) {
            self.advance(1); Some(CompareOp::Lt)
        }
        else { None }
    }

    // --- Helpers ---

    fn skip_ws(&mut self) {
        while let Some(c) = self.cur() {
            if c.is_whitespace() || c == '#' {
                if c == '#' {
                    while let Some(c2) = self.cur() {
                        self.advance(1);
                        if c2 == '\n' { break; }
                    }
                } else {
                    self.advance(1);
                }
            } else {
                break;
            }
        }
    }

    fn cur(&self) -> Option<char> { self.input.get(self.pos).copied() }

    fn peek(&self, c: char) -> bool { self.cur() == Some(c) }

    fn peek_at(&self, offset: usize) -> Option<char> {
        self.input.get(self.pos + offset).copied()
    }

    fn peek_kw(&self, kw: &str) -> bool {
        let chars: Vec<char> = kw.chars().collect();
        if self.pos + chars.len() > self.input.len() { return false; }
        for (i, c) in chars.iter().enumerate() {
            if self.input[self.pos + i].to_ascii_uppercase() != c.to_ascii_uppercase() {
                return false;
            }
        }
        // Must not be followed by an alphanumeric char.
        let next = self.input.get(self.pos + chars.len());
        next.map(|c| !c.is_alphanumeric() && *c != '_').unwrap_or(true)
    }

    fn peek_str(&self, s: &str) -> bool {
        let chars: Vec<char> = s.chars().collect();
        if self.pos + chars.len() > self.input.len() { return false; }
        for (i, c) in chars.iter().enumerate() {
            if self.input[self.pos + i] != *c { return false; }
        }
        true
    }

    fn advance(&mut self, n: usize) { self.pos += n; }

    fn advance_kw(&mut self, kw: &str) { self.pos += kw.len(); }

    fn expect(&mut self, c: char) -> QueryResult<()> {
        if self.cur() == Some(c) { self.advance(1); Ok(()) }
        else {
            Err(QueryError::ParseError(format!(
                "expected '{}' at pos {}, got {:?}", c, self.pos, self.cur()
            )))
        }
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while let Some(c) = self.cur() {
            if c.is_alphanumeric() || c == '_' || c == '-' {
                s.push(c);
                self.advance(1);
            } else { break; }
        }
        s
    }

    fn read_until(&mut self, end: char) -> String {
        let mut s = String::new();
        while let Some(c) = self.cur() {
            if c == end { break; }
            s.push(c);
            self.advance(1);
        }
        s
    }

    fn read_number(&mut self) -> QueryResult<i64> {
        let s = self.read_number_str();
        s.parse().map_err(|_| QueryError::ParseError(format!("invalid number: {}", s)))
    }

    fn read_number_str(&mut self) -> String {
        let mut s = String::new();
        if self.peek('-') { s.push('-'); self.advance(1); }
        while let Some(c) = self.cur() {
            if c.is_ascii_digit() || c == '.' { s.push(c); self.advance(1); }
            else { break; }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// Simple SPARQL evaluator (over in-memory triples)
// ---------------------------------------------------------------------------

/// A triple (subject, predicate, object) as strings.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: String,
    pub predicate: String,
    pub object: String,
}

/// Evaluate a SPARQL query against a set of triples.
pub fn evaluate_sparql(
    query: &SparqlQuery,
    triples: &[Triple],
) -> QueryResult<SparqlResult> {
    let bindings = match_pattern(&query.pattern, triples);

    let columns = match &query.form {
        SparqlForm::Select(vars) => vars.clone(),
        SparqlForm::SelectAll => {
            let mut vars: Vec<String> = Vec::new();
            for b in &bindings {
                for k in b.keys() {
                    if !vars.contains(k) { vars.push(k.clone()); }
                }
            }
            vars.sort();
            vars
        }
        SparqlForm::Ask => Vec::new(),
        _ => Vec::new(),
    };

    let mut rows: Vec<Vec<String>> = bindings
        .iter()
        .map(|b| {
            columns
                .iter()
                .map(|c| b.get(c).cloned().unwrap_or_default())
                .collect()
        })
        .collect();

    // Apply LIMIT/OFFSET.
    if let Some(off) = query.offset {
        rows = rows.into_iter().skip(off).collect();
    }
    if let Some(lim) = query.limit {
        rows.truncate(lim);
    }

    let ask_result = match &query.form {
        SparqlForm::Ask => Some(!bindings.is_empty()),
        _ => None,
    };

    Ok(SparqlResult {
        columns,
        rows,
        ask_result,
    })
}

#[derive(Debug, Clone)]
pub struct SparqlResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<String>>,
    pub ask_result: Option<bool>,
}

type Binding = HashMap<String, String>;

fn match_pattern(pattern: &GraphPattern, triples: &[Triple]) -> Vec<Binding> {
    match pattern {
        GraphPattern::Empty => vec![HashMap::new()],
        GraphPattern::Bgp(patterns) => {
            let mut bindings = vec![HashMap::new()];
            for tp in patterns {
                let mut new_bindings = Vec::new();
                for binding in &bindings {
                    for triple in triples {
                        if let Some(ext) = try_match_triple(tp, triple, binding) {
                            new_bindings.push(ext);
                        }
                    }
                }
                bindings = new_bindings;
            }
            bindings
        }
        GraphPattern::Optional(inner) => {
            // OPTIONAL: try to match, keep original binding if no match.
            let outer = vec![HashMap::new()];
            let inner_bindings = match_pattern(inner, triples);
            if inner_bindings.is_empty() {
                outer
            } else {
                inner_bindings
            }
        }
        GraphPattern::Union(left, right) => {
            let mut result = match_pattern(left, triples);
            result.extend(match_pattern(right, triples));
            result
        }
        GraphPattern::Filter(expr, inner) => {
            let bindings = match_pattern(inner, triples);
            // For now, pass through (full FILTER eval would need expression evaluator).
            bindings
        }
        GraphPattern::Join(patterns) => {
            let mut bindings = vec![HashMap::new()];
            for p in patterns {
                let inner = match_pattern(p, triples);
                bindings = join_bindings(&bindings, &inner);
            }
            bindings
        }
        GraphPattern::Bind(_, _) => vec![HashMap::new()],
    }
}

fn try_match_triple(
    pattern: &TriplePattern,
    triple: &Triple,
    binding: &Binding,
) -> Option<Binding> {
    let mut new_binding = binding.clone();

    if !match_term(&pattern.subject, &triple.subject, &mut new_binding) {
        return None;
    }
    if !match_term(&pattern.predicate, &triple.predicate, &mut new_binding) {
        return None;
    }
    if !match_term(&pattern.object, &triple.object, &mut new_binding) {
        return None;
    }

    Some(new_binding)
}

fn match_term(pattern: &SparqlTerm, value: &str, binding: &mut Binding) -> bool {
    match pattern {
        SparqlTerm::Variable(var) => {
            if let Some(existing) = binding.get(var) {
                existing == value
            } else {
                binding.insert(var.clone(), value.to_string());
                true
            }
        }
        SparqlTerm::Iri(iri) => iri == value,
        SparqlTerm::Literal(lit) => lit == value,
        SparqlTerm::Integer(n) => value == &n.to_string(),
        SparqlTerm::Float(f) => value == &f.to_string(),
        SparqlTerm::Boolean(b) => value == &b.to_string(),
        SparqlTerm::TypedLiteral(lit, _) => lit == value,
        SparqlTerm::BlankNode(label) => label == value,
    }
}

fn join_bindings(left: &[Binding], right: &[Binding]) -> Vec<Binding> {
    let mut result = Vec::new();
    for l in left {
        for r in right {
            if let Some(merged) = merge_bindings(l, r) {
                result.push(merged);
            }
        }
    }
    result
}

fn merge_bindings(a: &Binding, b: &Binding) -> Option<Binding> {
    let mut merged = a.clone();
    for (k, v) in b {
        if let Some(existing) = merged.get(k) {
            if existing != v { return None; }
        } else {
            merged.insert(k.clone(), v.clone());
        }
    }
    Some(merged)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_select() {
        let mut p = SparqlParser::new();
        let q = p.parse("SELECT ?name ?age WHERE { ?person <name> ?name . ?person <age> ?age }").unwrap();
        assert!(matches!(q.form, SparqlForm::Select(v) if v.len() == 2));
    }

    #[test]
    fn test_parse_select_star() {
        let mut p = SparqlParser::new();
        let q = p.parse("SELECT * WHERE { ?s ?p ?o }").unwrap();
        assert!(matches!(q.form, SparqlForm::SelectAll));
    }

    #[test]
    fn test_parse_ask() {
        let mut p = SparqlParser::new();
        let q = p.parse("ASK WHERE { ?s <knows> ?o }").unwrap();
        assert!(matches!(q.form, SparqlForm::Ask));
    }

    #[test]
    fn test_parse_prefix() {
        let mut p = SparqlParser::new();
        let q = p.parse(
            "PREFIX foaf: <http://xmlns.com/foaf/0.1/> SELECT ?name WHERE { ?p foaf:name ?name }"
        ).unwrap();
        assert_eq!(q.prefixes.get("foaf").unwrap(), "http://xmlns.com/foaf/0.1/");
    }

    #[test]
    fn test_parse_optional() {
        let mut p = SparqlParser::new();
        let q = p.parse(
            "SELECT ?name ?age WHERE { ?p <name> ?name OPTIONAL { ?p <age> ?age } }"
        ).unwrap();
        match &q.pattern {
            GraphPattern::Join(patterns) => assert!(patterns.len() >= 2),
            _ => panic!("expected Join pattern"),
        }
    }

    #[test]
    fn test_parse_limit_offset() {
        let mut p = SparqlParser::new();
        let q = p.parse("SELECT ?s WHERE { ?s ?p ?o } LIMIT 10 OFFSET 5").unwrap();
        assert_eq!(q.limit, Some(10));
        assert_eq!(q.offset, Some(5));
    }

    #[test]
    fn test_evaluate_bgp() {
        let triples = vec![
            Triple { subject: "alice".into(), predicate: "name".into(), object: "Alice".into() },
            Triple { subject: "bob".into(), predicate: "name".into(), object: "Bob".into() },
            Triple { subject: "alice".into(), predicate: "age".into(), object: "30".into() },
        ];

        let mut p = SparqlParser::new();
        let q = p.parse("SELECT ?name WHERE { ?person <name> ?name }").unwrap();
        let result = evaluate_sparql(&q, &triples).unwrap();

        assert_eq!(result.columns, vec!["name"]);
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_evaluate_join() {
        let triples = vec![
            Triple { subject: "alice".into(), predicate: "name".into(), object: "Alice".into() },
            Triple { subject: "bob".into(), predicate: "name".into(), object: "Bob".into() },
            Triple { subject: "alice".into(), predicate: "age".into(), object: "30".into() },
            Triple { subject: "bob".into(), predicate: "age".into(), object: "25".into() },
        ];

        let mut p = SparqlParser::new();
        let q = p.parse("SELECT ?name ?age WHERE { ?person <name> ?name . ?person <age> ?age }").unwrap();
        let result = evaluate_sparql(&q, &triples).unwrap();

        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_evaluate_ask() {
        let triples = vec![
            Triple { subject: "alice".into(), predicate: "knows".into(), object: "bob".into() },
        ];

        let mut p = SparqlParser::new();
        let q = p.parse("ASK WHERE { ?s <knows> ?o }").unwrap();
        let result = evaluate_sparql(&q, &triples).unwrap();
        assert_eq!(result.ask_result, Some(true));
    }

    #[test]
    fn test_evaluate_ask_false() {
        let triples = vec![
            Triple { subject: "alice".into(), predicate: "name".into(), object: "Alice".into() },
        ];

        let mut p = SparqlParser::new();
        let q = p.parse("ASK WHERE { ?s <knows> ?o }").unwrap();
        let result = evaluate_sparql(&q, &triples).unwrap();
        assert_eq!(result.ask_result, Some(false));
    }

    #[test]
    fn test_evaluate_limit() {
        let triples = vec![
            Triple { subject: "a".into(), predicate: "p".into(), object: "1".into() },
            Triple { subject: "b".into(), predicate: "p".into(), object: "2".into() },
            Triple { subject: "c".into(), predicate: "p".into(), object: "3".into() },
        ];

        let mut p = SparqlParser::new();
        let q = p.parse("SELECT ?s WHERE { ?s <p> ?o } LIMIT 2").unwrap();
        let result = evaluate_sparql(&q, &triples).unwrap();
        assert_eq!(result.rows.len(), 2);
    }

    #[test]
    fn test_evaluate_select_star() {
        let triples = vec![
            Triple { subject: "alice".into(), predicate: "name".into(), object: "Alice".into() },
        ];

        let mut p = SparqlParser::new();
        let q = p.parse("SELECT * WHERE { ?s ?p ?o }").unwrap();
        let result = evaluate_sparql(&q, &triples).unwrap();
        assert_eq!(result.columns.len(), 3); // o, p, s (sorted)
        assert_eq!(result.rows.len(), 1);
    }
}
