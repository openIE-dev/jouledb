//! Gremlin Query Language Parser
//!
//! Gremlin is the graph traversal language of Apache TinkerPop.
//! JouleDB supports the core traversal steps for graph navigation.
//!
//! # Supported syntax
//!
//! ```text
//! g.V().has("name", "Alice").out("knows").values("name")
//! g.V(1).outE("created").inV().path()
//! g.V().hasLabel("person").count()
//! g.addV("person").property("name", "Dave")
//! ```

use crate::ast::{Expression, Operator, Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// AST
// ---------------------------------------------------------------------------

/// A Gremlin traversal query.
#[derive(Debug, Clone)]
pub struct GremlinQuery {
    /// The traversal source (V, E, or addV/addE).
    pub source: GremlinSource,
    /// Sequence of traversal steps.
    pub steps: Vec<GremlinStep>,
}

impl GremlinQuery {
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

/// Traversal source.
#[derive(Debug, Clone)]
pub enum GremlinSource {
    /// g.V() or g.V(id)
    V(Option<GremlinValue>),
    /// g.E() or g.E(id)
    E(Option<GremlinValue>),
    /// g.addV("label")
    AddV(String),
    /// g.addE("label")
    AddE(String),
}

/// A single traversal step.
#[derive(Debug, Clone)]
pub enum GremlinStep {
    // --- Map steps ---
    /// .out("label") or .out()
    Out(Option<String>),
    /// .in("label") or .in()
    In(Option<String>),
    /// .both("label") or .both()
    Both(Option<String>),
    /// .outE("label")
    OutE(Option<String>),
    /// .inE("label")
    InE(Option<String>),
    /// .bothE("label")
    BothE(Option<String>),
    /// .inV()
    InV,
    /// .outV()
    OutV,
    /// .otherV()
    OtherV,
    /// .values("key1", "key2", ...)
    Values(Vec<String>),
    /// .valueMap() or .valueMap("key1", ...)
    ValueMap(Vec<String>),
    /// .id()
    Id,
    /// .label()
    Label,
    /// .select("a", "b")
    Select(Vec<String>),
    /// .project("a", "b")
    Project(Vec<String>),
    /// .as("label")
    As(String),
    /// .path()
    Path,

    // --- Filter steps ---
    /// .has("key", value) or .has("label", "key", value)
    Has(String, Option<GremlinValue>),
    /// .hasLabel("label")
    HasLabel(String),
    /// .hasId(id)
    HasId(GremlinValue),
    /// .where(predicate)
    Where(GremlinPredicate),
    /// .dedup()
    Dedup,
    /// .limit(n)
    Limit(usize),
    /// .range(start, end)
    Range(usize, usize),

    // --- Reduce steps ---
    /// .count()
    Count,
    /// .sum()
    Sum,
    /// .min()
    Min,
    /// .max()
    Max,
    /// .mean()
    Mean,
    /// .fold()
    Fold,
    /// .unfold()
    Unfold,

    // --- Side-effect steps ---
    /// .group().by(key)
    Group,
    /// .groupCount()
    GroupCount,
    /// .by("key") — modifier for group/order
    By(String),
    /// .order()
    Order,

    // --- Mutation steps ---
    /// .property("key", value)
    Property(String, GremlinValue),
    /// .drop()
    Drop,
    /// .from(traversal)
    From(GremlinValue),
    /// .to(traversal)
    To(GremlinValue),
}

/// A Gremlin value (argument to steps).
#[derive(Debug, Clone, PartialEq)]
pub enum GremlinValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
}

/// A Gremlin predicate (for .where() and .has() with predicates).
#[derive(Debug, Clone)]
pub enum GremlinPredicate {
    Eq(GremlinValue),
    Neq(GremlinValue),
    Lt(GremlinValue),
    Gt(GremlinValue),
    Lte(GremlinValue),
    Gte(GremlinValue),
    Within(Vec<GremlinValue>),
    Without(Vec<GremlinValue>),
}

// ---------------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------------

pub struct GremlinParser {
    input: Vec<char>,
    pos: usize,
}

impl GremlinParser {
    pub fn new() -> Self {
        Self {
            input: Vec::new(),
            pos: 0,
        }
    }

    pub fn parse(&mut self, input: &str) -> QueryResult<GremlinQuery> {
        self.input = input.chars().collect();
        self.pos = 0;

        self.skip_ws();
        // Expect "g."
        self.expect_str("g.")?;

        // Parse source.
        let source = self.parse_source()?;

        // Parse steps.
        let mut steps = Vec::new();
        while self.pos < self.input.len() {
            self.skip_ws();
            if !self.peek('.') {
                break;
            }
            self.advance(1); // skip '.'
            steps.push(self.parse_step()?);
        }

        Ok(GremlinQuery { source, steps })
    }

    fn parse_source(&mut self) -> QueryResult<GremlinSource> {
        let name = self.read_ident();
        match name.as_str() {
            "V" => {
                self.expect_char('(')?;
                self.skip_ws();
                let arg = if !self.peek(')') {
                    Some(self.parse_value()?)
                } else {
                    None
                };
                self.skip_ws();
                self.expect_char(')')?;
                Ok(GremlinSource::V(arg))
            }
            "E" => {
                self.expect_char('(')?;
                self.skip_ws();
                let arg = if !self.peek(')') {
                    Some(self.parse_value()?)
                } else {
                    None
                };
                self.skip_ws();
                self.expect_char(')')?;
                Ok(GremlinSource::E(arg))
            }
            "addV" => {
                self.expect_char('(')?;
                self.skip_ws();
                let label = self.parse_string_arg()?;
                self.skip_ws();
                self.expect_char(')')?;
                Ok(GremlinSource::AddV(label))
            }
            "addE" => {
                self.expect_char('(')?;
                self.skip_ws();
                let label = self.parse_string_arg()?;
                self.skip_ws();
                self.expect_char(')')?;
                Ok(GremlinSource::AddE(label))
            }
            _ => Err(QueryError::ParseError(format!("unknown source: {}", name))),
        }
    }

    fn parse_step(&mut self) -> QueryResult<GremlinStep> {
        let name = self.read_ident();
        self.expect_char('(')?;
        self.skip_ws();

        let step = match name.as_str() {
            "out" => GremlinStep::Out(self.try_parse_string_arg()?),
            "in" | "in_" => GremlinStep::In(self.try_parse_string_arg()?),
            "both" => GremlinStep::Both(self.try_parse_string_arg()?),
            "outE" => GremlinStep::OutE(self.try_parse_string_arg()?),
            "inE" => GremlinStep::InE(self.try_parse_string_arg()?),
            "bothE" => GremlinStep::BothE(self.try_parse_string_arg()?),
            "inV" => GremlinStep::InV,
            "outV" => GremlinStep::OutV,
            "otherV" => GremlinStep::OtherV,
            "values" => {
                let args = self.parse_string_list()?;
                GremlinStep::Values(args)
            }
            "valueMap" => {
                let args = self.parse_string_list()?;
                GremlinStep::ValueMap(args)
            }
            "id" => GremlinStep::Id,
            "label" => GremlinStep::Label,
            "select" => {
                let args = self.parse_string_list()?;
                GremlinStep::Select(args)
            }
            "project" => {
                let args = self.parse_string_list()?;
                GremlinStep::Project(args)
            }
            "as" | "as_" => {
                let label = self.parse_string_arg()?;
                GremlinStep::As(label)
            }
            "path" => GremlinStep::Path,
            "has" => {
                let key = self.parse_string_arg()?;
                self.skip_ws();
                let value = if self.peek(',') {
                    self.advance(1);
                    self.skip_ws();
                    Some(self.parse_value()?)
                } else {
                    None
                };
                GremlinStep::Has(key, value)
            }
            "hasLabel" => {
                let label = self.parse_string_arg()?;
                GremlinStep::HasLabel(label)
            }
            "hasId" => {
                let id = self.parse_value()?;
                GremlinStep::HasId(id)
            }
            "dedup" => GremlinStep::Dedup,
            "limit" => {
                let n = self.parse_int_arg()?;
                GremlinStep::Limit(n as usize)
            }
            "range" => {
                let start = self.parse_int_arg()?;
                self.skip_ws();
                self.expect_char(',')?;
                self.skip_ws();
                let end = self.parse_int_arg()?;
                GremlinStep::Range(start as usize, end as usize)
            }
            "count" => GremlinStep::Count,
            "sum" => GremlinStep::Sum,
            "min" => GremlinStep::Min,
            "max" => GremlinStep::Max,
            "mean" => GremlinStep::Mean,
            "fold" => GremlinStep::Fold,
            "unfold" => GremlinStep::Unfold,
            "group" => GremlinStep::Group,
            "groupCount" => GremlinStep::GroupCount,
            "by" => {
                let key = self.parse_string_arg()?;
                GremlinStep::By(key)
            }
            "order" => GremlinStep::Order,
            "property" => {
                let key = self.parse_string_arg()?;
                self.skip_ws();
                self.expect_char(',')?;
                self.skip_ws();
                let val = self.parse_value()?;
                GremlinStep::Property(key, val)
            }
            "drop" => GremlinStep::Drop,
            "from" | "from_" => {
                let val = self.parse_value()?;
                GremlinStep::From(val)
            }
            "to" => {
                let val = self.parse_value()?;
                GremlinStep::To(val)
            }
            _ => {
                return Err(QueryError::ParseError(format!(
                    "unknown Gremlin step: {}", name
                )));
            }
        };

        self.skip_ws();
        self.expect_char(')')?;
        Ok(step)
    }

    fn parse_value(&mut self) -> QueryResult<GremlinValue> {
        self.skip_ws();
        if self.peek('"') || self.peek('\'') {
            let s = self.parse_string_arg()?;
            Ok(GremlinValue::String(s))
        } else if self.peek_kw("true") {
            self.advance(4);
            Ok(GremlinValue::Boolean(true))
        } else if self.peek_kw("false") {
            self.advance(5);
            Ok(GremlinValue::Boolean(false))
        } else {
            let num = self.read_number_str();
            if num.contains('.') {
                Ok(GremlinValue::Float(num.parse().map_err(|_| {
                    QueryError::ParseError(format!("invalid float: {}", num))
                })?))
            } else {
                Ok(GremlinValue::Integer(num.parse().map_err(|_| {
                    QueryError::ParseError(format!("invalid integer: {}", num))
                })?))
            }
        }
    }

    fn parse_string_arg(&mut self) -> QueryResult<String> {
        self.skip_ws();
        let quote = self.cur().ok_or_else(|| {
            QueryError::ParseError("expected string".to_string())
        })?;
        if quote != '"' && quote != '\'' {
            return Err(QueryError::ParseError(format!("expected quote, got {:?}", quote)));
        }
        self.advance(1);
        let mut s = String::new();
        while let Some(c) = self.cur() {
            if c == quote { self.advance(1); return Ok(s); }
            if c == '\\' { self.advance(1); if let Some(e) = self.cur() { s.push(e); self.advance(1); } }
            else { s.push(c); self.advance(1); }
        }
        Ok(s)
    }

    fn try_parse_string_arg(&mut self) -> QueryResult<Option<String>> {
        self.skip_ws();
        if self.peek('"') || self.peek('\'') {
            Ok(Some(self.parse_string_arg()?))
        } else {
            Ok(None)
        }
    }

    fn parse_string_list(&mut self) -> QueryResult<Vec<String>> {
        let mut args = Vec::new();
        self.skip_ws();
        while self.peek('"') || self.peek('\'') {
            args.push(self.parse_string_arg()?);
            self.skip_ws();
            if self.peek(',') { self.advance(1); self.skip_ws(); }
        }
        Ok(args)
    }

    fn parse_int_arg(&mut self) -> QueryResult<i64> {
        let s = self.read_number_str();
        s.parse().map_err(|_| QueryError::ParseError(format!("invalid integer: {}", s)))
    }

    // --- Helpers ---

    fn skip_ws(&mut self) {
        while let Some(c) = self.cur() {
            if c.is_whitespace() { self.advance(1); } else { break; }
        }
    }

    fn cur(&self) -> Option<char> { self.input.get(self.pos).copied() }
    fn peek(&self, c: char) -> bool { self.cur() == Some(c) }
    fn advance(&mut self, n: usize) { self.pos += n; }

    fn peek_kw(&self, kw: &str) -> bool {
        let chars: Vec<char> = kw.chars().collect();
        if self.pos + chars.len() > self.input.len() { return false; }
        for (i, c) in chars.iter().enumerate() {
            if self.input[self.pos + i] != *c { return false; }
        }
        true
    }

    fn expect_str(&mut self, s: &str) -> QueryResult<()> {
        for c in s.chars() {
            self.expect_char(c)?;
        }
        Ok(())
    }

    fn expect_char(&mut self, c: char) -> QueryResult<()> {
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
            if c.is_alphanumeric() || c == '_' { s.push(c); self.advance(1); }
            else { break; }
        }
        s
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
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_v() {
        let mut p = GremlinParser::new();
        let q = p.parse("g.V()").unwrap();
        assert!(matches!(q.source, GremlinSource::V(None)));
        assert!(q.steps.is_empty());
    }

    #[test]
    fn test_parse_v_with_id() {
        let mut p = GremlinParser::new();
        let q = p.parse("g.V(1)").unwrap();
        assert!(matches!(q.source, GremlinSource::V(Some(GremlinValue::Integer(1)))));
    }

    #[test]
    fn test_parse_out() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().out("knows")"#).unwrap();
        assert_eq!(q.steps.len(), 1);
        assert!(matches!(&q.steps[0], GremlinStep::Out(Some(s)) if s == "knows"));
    }

    #[test]
    fn test_parse_has_label() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().hasLabel("person")"#).unwrap();
        assert_eq!(q.steps.len(), 1);
        assert!(matches!(&q.steps[0], GremlinStep::HasLabel(s) if s == "person"));
    }

    #[test]
    fn test_parse_has_property() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().has("name", "Alice")"#).unwrap();
        assert_eq!(q.steps.len(), 1);
        match &q.steps[0] {
            GremlinStep::Has(key, Some(GremlinValue::String(val))) => {
                assert_eq!(key, "name");
                assert_eq!(val, "Alice");
            }
            _ => panic!("expected Has step"),
        }
    }

    #[test]
    fn test_parse_chain() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().has("name", "Alice").out("knows").values("name")"#).unwrap();
        assert_eq!(q.steps.len(), 3);
        assert!(matches!(&q.steps[0], GremlinStep::Has(..)));
        assert!(matches!(&q.steps[1], GremlinStep::Out(..)));
        assert!(matches!(&q.steps[2], GremlinStep::Values(..)));
    }

    #[test]
    fn test_parse_count() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().hasLabel("person").count()"#).unwrap();
        assert_eq!(q.steps.len(), 2);
        assert!(matches!(&q.steps[1], GremlinStep::Count));
    }

    #[test]
    fn test_parse_add_v() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.addV("person").property("name", "Dave")"#).unwrap();
        assert!(matches!(&q.source, GremlinSource::AddV(s) if s == "person"));
        assert_eq!(q.steps.len(), 1);
        match &q.steps[0] {
            GremlinStep::Property(k, GremlinValue::String(v)) => {
                assert_eq!(k, "name");
                assert_eq!(v, "Dave");
            }
            _ => panic!("expected Property step"),
        }
    }

    #[test]
    fn test_parse_path() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V(1).out("created").path()"#).unwrap();
        assert_eq!(q.steps.len(), 2);
        assert!(matches!(&q.steps[1], GremlinStep::Path));
    }

    #[test]
    fn test_parse_limit() {
        let mut p = GremlinParser::new();
        let q = p.parse("g.V().limit(10)").unwrap();
        assert!(matches!(&q.steps[0], GremlinStep::Limit(10)));
    }

    #[test]
    fn test_parse_dedup() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().out("knows").dedup()"#).unwrap();
        assert_eq!(q.steps.len(), 2);
        assert!(matches!(&q.steps[1], GremlinStep::Dedup));
    }

    #[test]
    fn test_parse_range() {
        let mut p = GremlinParser::new();
        let q = p.parse("g.V().range(5, 10)").unwrap();
        assert!(matches!(&q.steps[0], GremlinStep::Range(5, 10)));
    }

    #[test]
    fn test_parse_group_count() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().out("knows").groupCount()"#).unwrap();
        assert_eq!(q.steps.len(), 2);
        assert!(matches!(&q.steps[1], GremlinStep::GroupCount));
    }

    #[test]
    fn test_parse_e() {
        let mut p = GremlinParser::new();
        let q = p.parse("g.E()").unwrap();
        assert!(matches!(q.source, GremlinSource::E(None)));
    }

    #[test]
    fn test_parse_both() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().both("knows")"#).unwrap();
        assert!(matches!(&q.steps[0], GremlinStep::Both(Some(s)) if s == "knows"));
    }

    #[test]
    fn test_parse_in_out_e_v() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V(1).outE("created").inV()"#).unwrap();
        assert_eq!(q.steps.len(), 2);
        assert!(matches!(&q.steps[0], GremlinStep::OutE(Some(s)) if s == "created"));
        assert!(matches!(&q.steps[1], GremlinStep::InV));
    }

    #[test]
    fn test_parse_drop() {
        let mut p = GremlinParser::new();
        let q = p.parse(r#"g.V().has("name", "old").drop()"#).unwrap();
        assert!(matches!(&q.steps[1], GremlinStep::Drop));
    }
}
