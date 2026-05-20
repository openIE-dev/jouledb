//! GraphQL Parser
//!
//! Parses GraphQL queries for database access.

use crate::ast::{Query, QueryType, Value};
use crate::error::{QueryError, QueryResult};
use std::collections::HashMap;

/// GraphQL query
#[derive(Debug, Clone)]
pub struct GraphqlQuery {
    pub operations: Vec<GraphqlOperation>,
    pub fragments: HashMap<String, GraphqlFragment>,
}

impl GraphqlQuery {
    /// Convert to generic Query (first operation)
    pub fn to_query(&self) -> Query {
        Query {
            query_type: QueryType::Select,
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
            distinct: false, source_alias: None,
        }
    }
}

/// GraphQL operation
#[derive(Debug, Clone)]
pub struct GraphqlOperation {
    pub operation_type: GraphqlOperationType,
    pub name: Option<String>,
    pub variables: Vec<GraphqlVariable>,
    pub directives: Vec<GraphqlDirective>,
    pub selection_set: Vec<GraphqlSelection>,
}

/// Operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphqlOperationType {
    Query,
    Mutation,
    Subscription,
}

/// GraphQL variable definition
#[derive(Debug, Clone)]
pub struct GraphqlVariable {
    pub name: String,
    pub var_type: GraphqlType,
    pub default_value: Option<GraphqlValue>,
}

/// GraphQL type
#[derive(Debug, Clone)]
pub struct GraphqlType {
    pub name: String,
    pub non_null: bool,
    pub list: bool,
    pub list_non_null: bool,
}

/// GraphQL selection
#[derive(Debug, Clone)]
pub enum GraphqlSelection {
    Field(GraphqlField),
    FragmentSpread(String),
    InlineFragment(GraphqlInlineFragment),
}

/// GraphQL field
#[derive(Debug, Clone)]
pub struct GraphqlField {
    pub alias: Option<String>,
    pub name: String,
    pub arguments: Vec<GraphqlArgument>,
    pub directives: Vec<GraphqlDirective>,
    pub selection_set: Vec<GraphqlSelection>,
}

impl GraphqlField {
    /// Create new field
    pub fn new(name: &str) -> Self {
        Self {
            alias: None,
            name: name.to_string(),
            arguments: Vec::new(),
            directives: Vec::new(),
            selection_set: Vec::new(),
        }
    }

    /// Set alias
    pub fn with_alias(mut self, alias: &str) -> Self {
        self.alias = Some(alias.to_string());
        self
    }

    /// Add argument
    pub fn with_argument(mut self, name: &str, value: GraphqlValue) -> Self {
        self.arguments.push(GraphqlArgument {
            name: name.to_string(),
            value,
        });
        self
    }

    /// Add selection
    pub fn with_selection(mut self, selection: GraphqlSelection) -> Self {
        self.selection_set.push(selection);
        self
    }
}

/// GraphQL argument
#[derive(Debug, Clone)]
pub struct GraphqlArgument {
    pub name: String,
    pub value: GraphqlValue,
}

/// GraphQL value
#[derive(Debug, Clone)]
pub enum GraphqlValue {
    Variable(String),
    Int(i64),
    Float(f64),
    String(String),
    Boolean(bool),
    Null,
    Enum(String),
    List(Vec<GraphqlValue>),
    Object(Vec<(String, GraphqlValue)>),
}

impl GraphqlValue {
    /// Convert to AST Value
    pub fn to_value(&self) -> Value {
        match self {
            Self::Variable(_) => Value::Null,
            Self::Int(n) => Value::Int(*n),
            Self::Float(n) => Value::Float(*n),
            Self::String(s) => Value::String(s.clone()),
            Self::Boolean(b) => Value::Bool(*b),
            Self::Null => Value::Null,
            Self::Enum(s) => Value::String(s.clone()),
            Self::List(items) => Value::Array(items.iter().map(|v| v.to_value()).collect()),
            Self::Object(fields) => {
                let mut map = HashMap::new();
                for (k, v) in fields {
                    map.insert(k.clone(), v.to_value());
                }
                Value::Object(map)
            }
        }
    }
}

/// GraphQL directive
#[derive(Debug, Clone)]
pub struct GraphqlDirective {
    pub name: String,
    pub arguments: Vec<GraphqlArgument>,
}

/// GraphQL fragment
#[derive(Debug, Clone)]
pub struct GraphqlFragment {
    pub name: String,
    pub type_condition: String,
    pub directives: Vec<GraphqlDirective>,
    pub selection_set: Vec<GraphqlSelection>,
}

/// GraphQL inline fragment
#[derive(Debug, Clone)]
pub struct GraphqlInlineFragment {
    pub type_condition: Option<String>,
    pub directives: Vec<GraphqlDirective>,
    pub selection_set: Vec<GraphqlSelection>,
}

/// GraphQL Parser
/// Maximum nesting depth for selection sets and values to prevent stack overflow.
const MAX_NESTING_DEPTH: usize = 128;

/// Maximum query length in bytes (1 MB).
const MAX_QUERY_LENGTH: usize = 1_048_576;

pub struct GraphqlParser {
    input: String,
    pos: usize,
    /// Current nesting depth (prevents stack overflow).
    nesting_depth: usize,
}

impl GraphqlParser {
    /// Create new parser
    pub fn new() -> Self {
        Self {
            input: String::new(),
            pos: 0,
            nesting_depth: 0,
        }
    }

    /// Parse GraphQL document
    pub fn parse(&mut self, graphql: &str) -> QueryResult<GraphqlQuery> {
        if graphql.len() > MAX_QUERY_LENGTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Query too long: {} bytes exceeds maximum of {} bytes",
                graphql.len(),
                MAX_QUERY_LENGTH
            )));
        }
        self.input = graphql.to_string();
        self.pos = 0;
        self.nesting_depth = 0;

        let mut operations = Vec::new();
        let mut fragments = HashMap::new();

        while self.pos < self.input.len() {
            self.skip_whitespace();
            if self.pos >= self.input.len() {
                break;
            }

            if self.peek_keyword("fragment") {
                let fragment = self.parse_fragment()?;
                fragments.insert(fragment.name.clone(), fragment);
            } else {
                operations.push(self.parse_operation()?);
            }
        }

        Ok(GraphqlQuery {
            operations,
            fragments,
        })
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
            if c.is_whitespace() || c == ',' {
                self.pos += c.len_utf8();
            } else if self.input[self.pos..].starts_with('#') {
                // Comment
                while self.pos < self.input.len() && !self.input[self.pos..].starts_with('\n') {
                    let ch = self.input[self.pos..].chars().next().expect("pos < len");
                    self.pos += ch.len_utf8();
                }
            } else {
                break;
            }
        }
    }

    fn peek_keyword(&self, keyword: &str) -> bool {
        self.input[self.pos..].to_lowercase().starts_with(keyword)
    }

    fn try_consume(&mut self, s: &str) -> bool {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    fn consume(&mut self, s: &str) -> QueryResult<()> {
        self.skip_whitespace();
        if self.input[self.pos..].starts_with(s) {
            self.pos += s.len();
            Ok(())
        } else {
            Err(QueryError::ParseError(format!("Expected '{}'", s)))
        }
    }

    fn parse_name(&mut self) -> QueryResult<String> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
            if c.is_alphanumeric() || c == '_' {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
        if self.pos > start {
            Ok(self.input[start..self.pos].to_string())
        } else {
            Err(QueryError::ParseError("Expected name".to_string()))
        }
    }

    fn parse_operation(&mut self) -> QueryResult<GraphqlOperation> {
        self.skip_whitespace();

        // Check for shorthand query
        if self.input[self.pos..].starts_with('{') {
            return Ok(GraphqlOperation {
                operation_type: GraphqlOperationType::Query,
                name: None,
                variables: Vec::new(),
                directives: Vec::new(),
                selection_set: self.parse_selection_set()?,
            });
        }

        // Parse operation type
        let operation_type = if self.try_consume("query") {
            GraphqlOperationType::Query
        } else if self.try_consume("mutation") {
            GraphqlOperationType::Mutation
        } else if self.try_consume("subscription") {
            GraphqlOperationType::Subscription
        } else {
            return Err(QueryError::ParseError(
                "Expected operation type".to_string(),
            ));
        };

        // Parse name (optional)
        let name = if !self.input[self.pos..].trim_start().starts_with('(')
            && !self.input[self.pos..].trim_start().starts_with('{')
            && !self.input[self.pos..].trim_start().starts_with('@')
        {
            Some(self.parse_name()?)
        } else {
            None
        };

        // Parse variables (optional)
        let variables = if self.try_consume("(") {
            let vars = self.parse_variable_definitions()?;
            self.consume(")")?;
            vars
        } else {
            Vec::new()
        };

        // Parse directives (optional)
        let directives = self.parse_directives()?;

        // Parse selection set
        let selection_set = self.parse_selection_set()?;

        Ok(GraphqlOperation {
            operation_type,
            name,
            variables,
            directives,
            selection_set,
        })
    }

    fn parse_variable_definitions(&mut self) -> QueryResult<Vec<GraphqlVariable>> {
        let mut vars = Vec::new();

        loop {
            self.skip_whitespace();
            if self.input[self.pos..].starts_with(')') {
                break;
            }

            self.consume("$")?;
            let name = self.parse_name()?;
            self.consume(":")?;
            let var_type = self.parse_type()?;

            let default_value = if self.try_consume("=") {
                Some(self.parse_value()?)
            } else {
                None
            };

            vars.push(GraphqlVariable {
                name,
                var_type,
                default_value,
            });
            // Note: skip_whitespace() already consumes commas (GraphQL insignificant separators),
            // so the loop-top check for ')' handles termination.
        }

        Ok(vars)
    }

    fn parse_type(&mut self) -> QueryResult<GraphqlType> {
        self.skip_whitespace();

        let list = self.try_consume("[");
        let name = self.parse_name()?;
        let inner_non_null = self.try_consume("!");

        let list_non_null = if list {
            self.consume("]")?;
            self.try_consume("!")
        } else {
            false
        };

        Ok(GraphqlType {
            name,
            non_null: if list { inner_non_null } else { inner_non_null },
            list,
            list_non_null,
        })
    }

    fn parse_selection_set(&mut self) -> QueryResult<Vec<GraphqlSelection>> {
        self.nesting_depth += 1;
        if self.nesting_depth > MAX_NESTING_DEPTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Selection set nesting too deep: exceeds maximum depth of {}",
                MAX_NESTING_DEPTH
            )));
        }
        self.consume("{")?;
        let mut selections = Vec::new();

        loop {
            self.skip_whitespace();
            if self.input[self.pos..].starts_with('}') {
                break;
            }

            if self.try_consume("...") {
                self.skip_whitespace();
                if self.peek_keyword("on")
                    || self.input[self.pos..].starts_with('{')
                    || self.input[self.pos..].starts_with('@')
                {
                    // Inline fragment
                    let type_condition = if self.try_consume("on") {
                        Some(self.parse_name()?)
                    } else {
                        None
                    };
                    let directives = self.parse_directives()?;
                    let selection_set = self.parse_selection_set()?;
                    selections.push(GraphqlSelection::InlineFragment(GraphqlInlineFragment {
                        type_condition,
                        directives,
                        selection_set,
                    }));
                } else {
                    // Fragment spread
                    let name = self.parse_name()?;
                    selections.push(GraphqlSelection::FragmentSpread(name));
                }
            } else {
                selections.push(GraphqlSelection::Field(self.parse_field()?));
            }
        }

        self.consume("}")?;
        self.nesting_depth -= 1;
        Ok(selections)
    }

    fn parse_field(&mut self) -> QueryResult<GraphqlField> {
        let name_or_alias = self.parse_name()?;

        let (alias, name) = if self.try_consume(":") {
            (Some(name_or_alias), self.parse_name()?)
        } else {
            (None, name_or_alias)
        };

        let arguments = if self.try_consume("(") {
            let args = self.parse_arguments()?;
            self.consume(")")?;
            args
        } else {
            Vec::new()
        };

        let directives = self.parse_directives()?;

        let selection_set = if self.input[self.pos..].trim_start().starts_with('{') {
            self.parse_selection_set()?
        } else {
            Vec::new()
        };

        Ok(GraphqlField {
            alias,
            name,
            arguments,
            directives,
            selection_set,
        })
    }

    fn parse_arguments(&mut self) -> QueryResult<Vec<GraphqlArgument>> {
        let mut args = Vec::new();

        loop {
            self.skip_whitespace();
            if self.input[self.pos..].starts_with(')') {
                break;
            }

            let name = self.parse_name()?;
            self.consume(":")?;
            let value = self.parse_value()?;

            args.push(GraphqlArgument { name, value });
            // Note: skip_whitespace() already consumes commas (GraphQL insignificant separators),
            // so the loop-top check for ')' handles termination.
        }

        Ok(args)
    }

    fn parse_value(&mut self) -> QueryResult<GraphqlValue> {
        self.nesting_depth += 1;
        if self.nesting_depth > MAX_NESTING_DEPTH {
            return Err(crate::error::QueryError::ParseError(format!(
                "Value nesting too deep: exceeds maximum depth of {}",
                MAX_NESTING_DEPTH
            )));
        }
        let result = self.parse_value_inner();
        self.nesting_depth -= 1;
        result
    }

    fn parse_value_inner(&mut self) -> QueryResult<GraphqlValue> {
        self.skip_whitespace();

        // Variable
        if self.try_consume("$") {
            return Ok(GraphqlValue::Variable(self.parse_name()?));
        }

        // String
        if self.input[self.pos..].starts_with('"') {
            return self.parse_string_value();
        }

        // List
        if self.try_consume("[") {
            let mut items = Vec::new();
            loop {
                self.skip_whitespace();
                if self.input[self.pos..].starts_with(']') {
                    break;
                }
                items.push(self.parse_value()?);
                self.try_consume(",");
            }
            self.consume("]")?;
            return Ok(GraphqlValue::List(items));
        }

        // Object
        if self.try_consume("{") {
            let mut fields = Vec::new();
            loop {
                self.skip_whitespace();
                if self.input[self.pos..].starts_with('}') {
                    break;
                }
                let name = self.parse_name()?;
                self.consume(":")?;
                let value = self.parse_value()?;
                fields.push((name, value));
                self.try_consume(",");
            }
            self.consume("}")?;
            return Ok(GraphqlValue::Object(fields));
        }

        // Number or boolean/null/enum
        let start = self.pos;
        let mut has_dot = false;
        let mut has_exp = false;
        let mut is_negative = false;

        if self.input[self.pos..].starts_with('-') {
            is_negative = true;
            self.pos += 1;
        }

        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
            if c.is_ascii_digit() {
                self.pos += 1;
            } else if c == '.' && !has_dot {
                has_dot = true;
                self.pos += 1;
            } else if (c == 'e' || c == 'E') && !has_exp {
                has_exp = true;
                self.pos += 1;
                if self.input[self.pos..].starts_with('+')
                    || self.input[self.pos..].starts_with('-')
                {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }

        if self.pos > start + (if is_negative { 1 } else { 0 }) {
            let num_str = &self.input[start..self.pos];
            if has_dot || has_exp {
                return Ok(GraphqlValue::Float(num_str.parse().unwrap_or(0.0)));
            } else {
                return Ok(GraphqlValue::Int(num_str.parse().unwrap_or(0)));
            }
        }

        // Reset if not a number
        self.pos = start;

        // Name (boolean, null, or enum)
        let name = self.parse_name()?;
        match name.as_str() {
            "true" => Ok(GraphqlValue::Boolean(true)),
            "false" => Ok(GraphqlValue::Boolean(false)),
            "null" => Ok(GraphqlValue::Null),
            _ => Ok(GraphqlValue::Enum(name)),
        }
    }

    fn parse_string_value(&mut self) -> QueryResult<GraphqlValue> {
        // Check for block string
        if self.input[self.pos..].starts_with("\"\"\"") {
            self.pos += 3;
            let start = self.pos;
            while self.pos < self.input.len() && !self.input[self.pos..].starts_with("\"\"\"") {
                let ch = self.input[self.pos..].chars().next().expect("pos < len");
                self.pos += ch.len_utf8();
            }
            let s = self.input[start..self.pos].to_string();
            self.pos += 3;
            return Ok(GraphqlValue::String(s));
        }

        self.consume("\"")?;
        let mut s = String::new();
        while self.pos < self.input.len() {
            let c = self.input[self.pos..]
                .chars()
                .next()
                .expect("pos < len guarantees char exists");
            if c == '"' {
                break;
            } else if c == '\\' {
                self.pos += 1;
                if let Some(escaped) = self.input[self.pos..].chars().next() {
                    self.pos += escaped.len_utf8();
                    match escaped {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        'r' => s.push('\r'),
                        '\\' => s.push('\\'),
                        '"' => s.push('"'),
                        '/' => s.push('/'),
                        'b' => s.push('\x08'),
                        'f' => s.push('\x0C'),
                        'u' => {
                            // Unicode escape
                            if self.pos + 4 <= self.input.len() {
                                let hex = &self.input[self.pos..self.pos + 4];
                                if let Ok(code) = u32::from_str_radix(hex, 16) {
                                    if let Some(ch) = char::from_u32(code) {
                                        s.push(ch);
                                    }
                                }
                                self.pos += 4;
                            }
                        }
                        _ => s.push(escaped),
                    }
                }
            } else {
                s.push(c);
                self.pos += c.len_utf8();
            }
        }
        self.consume("\"")?;
        Ok(GraphqlValue::String(s))
    }

    fn parse_directives(&mut self) -> QueryResult<Vec<GraphqlDirective>> {
        let mut directives = Vec::new();

        while self.try_consume("@") {
            let name = self.parse_name()?;
            let arguments = if self.try_consume("(") {
                let args = self.parse_arguments()?;
                self.consume(")")?;
                args
            } else {
                Vec::new()
            };

            directives.push(GraphqlDirective { name, arguments });
        }

        Ok(directives)
    }

    fn parse_fragment(&mut self) -> QueryResult<GraphqlFragment> {
        self.consume("fragment")?;
        let name = self.parse_name()?;
        self.consume("on")?;
        let type_condition = self.parse_name()?;
        let directives = self.parse_directives()?;
        let selection_set = self.parse_selection_set()?;

        Ok(GraphqlFragment {
            name,
            type_condition,
            directives,
            selection_set,
        })
    }
}

impl Default for GraphqlParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_query() {
        let mut parser = GraphqlParser::new();
        let query = parser.parse("{ user { id name } }").unwrap();

        assert_eq!(query.operations.len(), 1);
        assert_eq!(
            query.operations[0].operation_type,
            GraphqlOperationType::Query
        );
    }

    #[test]
    fn test_named_query() {
        let mut parser = GraphqlParser::new();
        let query = parser.parse("query GetUser { user { id } }").unwrap();

        assert_eq!(query.operations[0].name, Some("GetUser".to_string()));
    }

    #[test]
    fn test_query_with_variables() {
        let mut parser = GraphqlParser::new();
        let query = parser
            .parse("query GetUser($id: ID!) { user(id: $id) { name } }")
            .unwrap();

        assert_eq!(query.operations[0].variables.len(), 1);
        assert_eq!(query.operations[0].variables[0].name, "id");
    }

    #[test]
    fn test_mutation() {
        let mut parser = GraphqlParser::new();
        let query = parser
            .parse("mutation CreateUser { createUser(name: \"Alice\") { id } }")
            .unwrap();

        assert_eq!(
            query.operations[0].operation_type,
            GraphqlOperationType::Mutation
        );
    }

    #[test]
    fn test_field_with_alias() {
        let mut parser = GraphqlParser::new();
        let query = parser.parse("{ admin: user(role: ADMIN) { id } }").unwrap();

        if let GraphqlSelection::Field(f) = &query.operations[0].selection_set[0] {
            assert_eq!(f.alias, Some("admin".to_string()));
            assert_eq!(f.name, "user");
        }
    }

    #[test]
    fn test_fragment() {
        let mut parser = GraphqlParser::new();
        let query = parser
            .parse(
                "
                fragment UserFields on User { id name }
                query { user { ...UserFields } }
            ",
            )
            .unwrap();

        assert_eq!(query.fragments.len(), 1);
        assert!(query.fragments.contains_key("UserFields"));
    }

    #[test]
    fn test_nested_selection() {
        let mut parser = GraphqlParser::new();
        let query = parser
            .parse("{ user { posts { id title comments { text } } } }")
            .unwrap();

        if let GraphqlSelection::Field(user) = &query.operations[0].selection_set[0] {
            assert_eq!(user.selection_set.len(), 1);
            if let GraphqlSelection::Field(posts) = &user.selection_set[0] {
                assert_eq!(posts.selection_set.len(), 3);
            }
        }
    }

    #[test]
    fn test_directives() {
        let mut parser = GraphqlParser::new();
        let query = parser.parse("{ user @include(if: true) { id } }").unwrap();

        if let GraphqlSelection::Field(f) = &query.operations[0].selection_set[0] {
            assert_eq!(f.directives.len(), 1);
            assert_eq!(f.directives[0].name, "include");
        }
    }

    #[test]
    fn test_various_value_types() {
        let mut parser = GraphqlParser::new();
        let query = parser
            .parse(
                "{ 
                    a(int: 42)
                    b(float: 3.14)
                    c(string: \"hello\")
                    d(bool: true)
                    e(null: null)
                    f(enum: ACTIVE)
                    g(list: [1, 2, 3])
                    h(object: {key: \"value\"})
                }",
            )
            .unwrap();

        assert_eq!(query.operations[0].selection_set.len(), 8);
    }
}
