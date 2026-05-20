//! GraphQL query parser — full parse of queries, mutations, subscriptions,
//! selection sets, arguments, variables, fragments (named and inline),
//! directives, and the type system (scalar/object/enum/union/interface/input).
//!
//! Replaces `graphql-js` parser, `graphql-tag`, and similar JS parsing
//! libraries with a pure-Rust hand-written recursive-descent parser.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ───────────────────────────────────────────────────────

/// Parse error with location information.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub line: usize,
    pub column: usize,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}: {}", self.line, self.column, self.message)
    }
}

impl std::error::Error for ParseError {}

// ── AST Types ────────────────────────────────────────────────────

/// A complete GraphQL document — one or more definitions.
#[derive(Debug, Clone, PartialEq)]
pub struct Document {
    pub definitions: Vec<Definition>,
}

/// Top-level definition: operation, fragment, or type system.
#[derive(Debug, Clone, PartialEq)]
pub enum Definition {
    Operation(OperationDefinition),
    Fragment(FragmentDefinition),
    TypeSystem(TypeSystemDefinition),
}

/// An operation (query / mutation / subscription) or shorthand query.
#[derive(Debug, Clone, PartialEq)]
pub struct OperationDefinition {
    pub operation: OperationType,
    pub name: Option<String>,
    pub variables: Vec<VariableDefinition>,
    pub directives: Vec<Directive>,
    pub selection_set: SelectionSet,
}

/// Operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OperationType {
    Query,
    Mutation,
    Subscription,
}

impl fmt::Display for OperationType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Query => f.write_str("query"),
            Self::Mutation => f.write_str("mutation"),
            Self::Subscription => f.write_str("subscription"),
        }
    }
}

/// A variable definition: `$name: Type = defaultValue`.
#[derive(Debug, Clone, PartialEq)]
pub struct VariableDefinition {
    pub name: String,
    pub var_type: TypeRef,
    pub default_value: Option<Value>,
    pub directives: Vec<Directive>,
}

/// Type reference: named, list, or non-null wrapper.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeRef {
    Named(String),
    List(Box<TypeRef>),
    NonNull(Box<TypeRef>),
}

impl fmt::Display for TypeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Named(n) => write!(f, "{n}"),
            Self::List(inner) => write!(f, "[{inner}]"),
            Self::NonNull(inner) => write!(f, "{inner}!"),
        }
    }
}

/// Selection set — a list of fields, fragment spreads, or inline fragments.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionSet {
    pub selections: Vec<Selection>,
}

/// A single selection.
#[derive(Debug, Clone, PartialEq)]
pub enum Selection {
    Field(Field),
    FragmentSpread(FragmentSpread),
    InlineFragment(InlineFragment),
}

/// Field selection with optional alias, arguments, directives, and sub-selection.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub alias: Option<String>,
    pub name: String,
    pub arguments: Vec<Argument>,
    pub directives: Vec<Directive>,
    pub selection_set: Option<SelectionSet>,
}

/// A named argument: `key: value`.
#[derive(Debug, Clone, PartialEq)]
pub struct Argument {
    pub name: String,
    pub value: Value,
}

/// A directive application: `@name(args)`.
#[derive(Debug, Clone, PartialEq)]
pub struct Directive {
    pub name: String,
    pub arguments: Vec<Argument>,
}

/// Fragment spread: `...FragmentName`.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentSpread {
    pub name: String,
    pub directives: Vec<Directive>,
}

/// Inline fragment: `... on Type { fields }`.
#[derive(Debug, Clone, PartialEq)]
pub struct InlineFragment {
    pub type_condition: Option<String>,
    pub directives: Vec<Directive>,
    pub selection_set: SelectionSet,
}

/// Fragment definition: `fragment Name on Type { fields }`.
#[derive(Debug, Clone, PartialEq)]
pub struct FragmentDefinition {
    pub name: String,
    pub type_condition: String,
    pub directives: Vec<Directive>,
    pub selection_set: SelectionSet,
}

/// A GraphQL value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Variable(String),
    Int(i64),
    Float(f64),
    StringValue(String),
    Boolean(bool),
    Null,
    Enum(String),
    List(Vec<Value>),
    Object(Vec<(String, Value)>),
}

// ── Type System Definitions ──────────────────────────────────────

/// Type system definition (schema-level).
#[derive(Debug, Clone, PartialEq)]
pub enum TypeSystemDefinition {
    Schema(SchemaDefinition),
    Scalar(ScalarDefinition),
    Object(ObjectDefinition),
    Interface(InterfaceDefinition),
    Union(UnionDefinition),
    Enum(EnumDefinition),
    InputObject(InputObjectDefinition),
}

/// Schema definition: `schema { query: Query ... }`.
#[derive(Debug, Clone, PartialEq)]
pub struct SchemaDefinition {
    pub directives: Vec<Directive>,
    pub query: Option<String>,
    pub mutation: Option<String>,
    pub subscription: Option<String>,
}

/// Scalar type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ScalarDefinition {
    pub name: String,
    pub description: Option<String>,
    pub directives: Vec<Directive>,
}

/// Object type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectDefinition {
    pub name: String,
    pub description: Option<String>,
    pub interfaces: Vec<String>,
    pub directives: Vec<Directive>,
    pub fields: Vec<FieldDefinition>,
}

/// Interface type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct InterfaceDefinition {
    pub name: String,
    pub description: Option<String>,
    pub directives: Vec<Directive>,
    pub fields: Vec<FieldDefinition>,
}

/// Union type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct UnionDefinition {
    pub name: String,
    pub description: Option<String>,
    pub directives: Vec<Directive>,
    pub members: Vec<String>,
}

/// Enum type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumDefinition {
    pub name: String,
    pub description: Option<String>,
    pub directives: Vec<Directive>,
    pub values: Vec<EnumValueDefinition>,
}

/// A single enum value.
#[derive(Debug, Clone, PartialEq)]
pub struct EnumValueDefinition {
    pub name: String,
    pub description: Option<String>,
    pub directives: Vec<Directive>,
}

/// Input object type definition.
#[derive(Debug, Clone, PartialEq)]
pub struct InputObjectDefinition {
    pub name: String,
    pub description: Option<String>,
    pub directives: Vec<Directive>,
    pub fields: Vec<InputValueDefinition>,
}

/// A field in an object or interface definition.
#[derive(Debug, Clone, PartialEq)]
pub struct FieldDefinition {
    pub name: String,
    pub description: Option<String>,
    pub arguments: Vec<InputValueDefinition>,
    pub field_type: TypeRef,
    pub directives: Vec<Directive>,
}

/// Input value (argument or input object field).
#[derive(Debug, Clone, PartialEq)]
pub struct InputValueDefinition {
    pub name: String,
    pub description: Option<String>,
    pub input_type: TypeRef,
    pub default_value: Option<Value>,
    pub directives: Vec<Directive>,
}

// ── Tokenizer ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Name(String),
    IntValue(i64),
    FloatValue(f64),
    StringValue(String),
    Punctuation(char),
    Spread,   // "..."
    Dollar,   // "$"
    At,       // "@"
    Equals,   // "="
    Colon,    // ":"
    Bang,     // "!"
    Pipe,     // "|"
    Amp,      // "&"
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Self {
            chars: input.chars().collect(),
            pos: 0,
            line: 1,
            col: 1,
        }
    }

    fn peek_token(&mut self) -> Result<Token, ParseError> {
        let saved_pos = self.pos;
        let saved_line = self.line;
        let saved_col = self.col;
        let tok = self.next_token();
        self.pos = saved_pos;
        self.line = saved_line;
        self.col = saved_col;
        tok
    }

    fn next_token(&mut self) -> Result<Token, ParseError> {
        self.skip_whitespace_and_comments();
        if self.pos >= self.chars.len() {
            return Ok(Token::Eof);
        }
        let ch = self.chars[self.pos];
        match ch {
            '{' | '}' | '(' | ')' | '[' | ']' => {
                self.advance();
                Ok(Token::Punctuation(ch))
            }
            '$' => { self.advance(); Ok(Token::Dollar) }
            '@' => { self.advance(); Ok(Token::At) }
            '=' => { self.advance(); Ok(Token::Equals) }
            ':' => { self.advance(); Ok(Token::Colon) }
            '!' => { self.advance(); Ok(Token::Bang) }
            '|' => { self.advance(); Ok(Token::Pipe) }
            '&' => { self.advance(); Ok(Token::Amp) }
            '.' => {
                if self.pos + 2 < self.chars.len()
                    && self.chars[self.pos + 1] == '.'
                    && self.chars[self.pos + 2] == '.'
                {
                    self.advance();
                    self.advance();
                    self.advance();
                    Ok(Token::Spread)
                } else {
                    Err(self.error("unexpected '.'"))
                }
            }
            '"' => self.read_string(),
            c if c == '-' || c.is_ascii_digit() => self.read_number(),
            c if c == '_' || c.is_ascii_alphabetic() => self.read_name(),
            _ => Err(self.error(&format!("unexpected character '{ch}'"))),
        }
    }

    fn read_name(&mut self) -> Result<Token, ParseError> {
        let start = self.pos;
        while self.pos < self.chars.len()
            && (self.chars[self.pos] == '_' || self.chars[self.pos].is_ascii_alphanumeric())
        {
            self.advance();
        }
        let name: String = self.chars[start..self.pos].iter().collect();
        Ok(Token::Name(name))
    }

    fn read_number(&mut self) -> Result<Token, ParseError> {
        let start = self.pos;
        if self.pos < self.chars.len() && self.chars[self.pos] == '-' {
            self.advance();
        }
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
            self.advance();
        }
        let mut is_float = false;
        if self.pos < self.chars.len() && self.chars[self.pos] == '.' {
            is_float = true;
            self.advance();
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                self.advance();
            }
        }
        if self.pos < self.chars.len()
            && (self.chars[self.pos] == 'e' || self.chars[self.pos] == 'E')
        {
            is_float = true;
            self.advance();
            if self.pos < self.chars.len()
                && (self.chars[self.pos] == '+' || self.chars[self.pos] == '-')
            {
                self.advance();
            }
            while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_digit() {
                self.advance();
            }
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        if is_float {
            text.parse::<f64>()
                .map(Token::FloatValue)
                .map_err(|_| self.error(&format!("invalid float: {text}")))
        } else {
            text.parse::<i64>()
                .map(Token::IntValue)
                .map_err(|_| self.error(&format!("invalid int: {text}")))
        }
    }

    fn read_string(&mut self) -> Result<Token, ParseError> {
        self.advance(); // opening quote
        let mut buf = String::new();
        while self.pos < self.chars.len() && self.chars[self.pos] != '"' {
            if self.chars[self.pos] == '\\' {
                self.advance();
                if self.pos >= self.chars.len() {
                    return Err(self.error("unexpected end of string"));
                }
                match self.chars[self.pos] {
                    '"' => buf.push('"'),
                    '\\' => buf.push('\\'),
                    '/' => buf.push('/'),
                    'n' => buf.push('\n'),
                    'r' => buf.push('\r'),
                    't' => buf.push('\t'),
                    c => buf.push(c),
                }
            } else {
                buf.push(self.chars[self.pos]);
            }
            self.advance();
        }
        if self.pos >= self.chars.len() {
            return Err(self.error("unterminated string"));
        }
        self.advance(); // closing quote
        Ok(Token::StringValue(buf))
    }

    fn skip_whitespace_and_comments(&mut self) {
        while self.pos < self.chars.len() {
            let ch = self.chars[self.pos];
            if ch == '\n' {
                self.line += 1;
                self.col = 0;
                self.pos += 1;
            } else if ch.is_ascii_whitespace() || ch == ',' {
                self.advance();
            } else if ch == '#' {
                // skip comment to end of line
                while self.pos < self.chars.len() && self.chars[self.pos] != '\n' {
                    self.pos += 1;
                }
            } else {
                break;
            }
        }
    }

    fn advance(&mut self) {
        if self.pos < self.chars.len() {
            self.pos += 1;
            self.col += 1;
        }
    }

    fn error(&self, msg: &str) -> ParseError {
        ParseError {
            message: msg.to_string(),
            line: self.line,
            column: self.col,
        }
    }

    fn expect_name(&mut self) -> Result<String, ParseError> {
        match self.next_token()? {
            Token::Name(n) => Ok(n),
            other => Err(self.error(&format!("expected name, got {other:?}"))),
        }
    }

    fn expect_punctuation(&mut self, expected: char) -> Result<(), ParseError> {
        match self.next_token()? {
            Token::Punctuation(c) if c == expected => Ok(()),
            other => Err(self.error(&format!("expected '{expected}', got {other:?}"))),
        }
    }
}

// ── Parser ───────────────────────────────────────────────────────

/// Parse a GraphQL document string into an AST.
pub fn parse(input: &str) -> Result<Document, ParseError> {
    let mut lexer = Lexer::new(input);
    let mut definitions = Vec::new();
    loop {
        let tok = lexer.peek_token()?;
        match tok {
            Token::Eof => break,
            Token::Punctuation('{') => {
                // shorthand query
                let sel = parse_selection_set(&mut lexer)?;
                definitions.push(Definition::Operation(OperationDefinition {
                    operation: OperationType::Query,
                    name: None,
                    variables: Vec::new(),
                    directives: Vec::new(),
                    selection_set: sel,
                }));
            }
            Token::Name(ref n) => {
                let name_lower = n.to_lowercase();
                match name_lower.as_str() {
                    "query" | "mutation" | "subscription" => {
                        definitions.push(Definition::Operation(
                            parse_operation_definition(&mut lexer)?,
                        ));
                    }
                    "fragment" => {
                        definitions.push(Definition::Fragment(
                            parse_fragment_definition(&mut lexer)?,
                        ));
                    }
                    "schema" => {
                        definitions.push(Definition::TypeSystem(
                            parse_type_system_definition(&mut lexer)?,
                        ));
                    }
                    "scalar" | "type" | "interface" | "union" | "enum" | "input" => {
                        definitions.push(Definition::TypeSystem(
                            parse_type_system_definition(&mut lexer)?,
                        ));
                    }
                    _ => return Err(lexer.error(&format!("unexpected keyword: {n}"))),
                }
            }
            Token::StringValue(_) => {
                // description string before type definition
                definitions.push(Definition::TypeSystem(
                    parse_type_system_definition(&mut lexer)?,
                ));
            }
            _ => return Err(lexer.error(&format!("unexpected token: {tok:?}"))),
        }
    }
    Ok(Document { definitions })
}

fn parse_operation_definition(lexer: &mut Lexer) -> Result<OperationDefinition, ParseError> {
    let op_token = lexer.expect_name()?;
    let operation = match op_token.as_str() {
        "query" => OperationType::Query,
        "mutation" => OperationType::Mutation,
        "subscription" => OperationType::Subscription,
        _ => return Err(lexer.error(&format!("expected operation type, got {op_token}"))),
    };
    // optional name
    let name = match lexer.peek_token()? {
        Token::Name(_) => Some(lexer.expect_name()?),
        _ => None,
    };
    // optional variable definitions
    let variables = if matches!(lexer.peek_token()?, Token::Punctuation('(')) {
        parse_variable_definitions(lexer)?
    } else {
        Vec::new()
    };
    let directives = parse_directives(lexer)?;
    let selection_set = parse_selection_set(lexer)?;
    Ok(OperationDefinition {
        operation,
        name,
        variables,
        directives,
        selection_set,
    })
}

fn parse_variable_definitions(lexer: &mut Lexer) -> Result<Vec<VariableDefinition>, ParseError> {
    lexer.expect_punctuation('(')?;
    let mut vars = Vec::new();
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation(')')) {
            lexer.next_token()?;
            break;
        }
        // expect "$name"
        match lexer.next_token()? {
            Token::Dollar => {}
            other => return Err(lexer.error(&format!("expected '$', got {other:?}"))),
        }
        let name = lexer.expect_name()?;
        match lexer.next_token()? {
            Token::Colon => {}
            other => return Err(lexer.error(&format!("expected ':', got {other:?}"))),
        }
        let var_type = parse_type_ref(lexer)?;
        let default_value = if matches!(lexer.peek_token()?, Token::Equals) {
            lexer.next_token()?; // consume '='
            Some(parse_value(lexer)?)
        } else {
            None
        };
        let directives = parse_directives(lexer)?;
        vars.push(VariableDefinition {
            name,
            var_type,
            default_value,
            directives,
        });
    }
    Ok(vars)
}

fn parse_type_ref(lexer: &mut Lexer) -> Result<TypeRef, ParseError> {
    let base = match lexer.peek_token()? {
        Token::Punctuation('[') => {
            lexer.next_token()?;
            let inner = parse_type_ref(lexer)?;
            lexer.expect_punctuation(']')?;
            TypeRef::List(Box::new(inner))
        }
        Token::Name(_) => {
            let name = lexer.expect_name()?;
            TypeRef::Named(name)
        }
        other => return Err(lexer.error(&format!("expected type, got {other:?}"))),
    };
    // check for non-null
    if matches!(lexer.peek_token()?, Token::Bang) {
        lexer.next_token()?;
        Ok(TypeRef::NonNull(Box::new(base)))
    } else {
        Ok(base)
    }
}

fn parse_selection_set(lexer: &mut Lexer) -> Result<SelectionSet, ParseError> {
    lexer.expect_punctuation('{')?;
    let mut selections = Vec::new();
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation('}')) {
            lexer.next_token()?;
            break;
        }
        selections.push(parse_selection(lexer)?);
    }
    Ok(SelectionSet { selections })
}

fn parse_selection(lexer: &mut Lexer) -> Result<Selection, ParseError> {
    if matches!(lexer.peek_token()?, Token::Spread) {
        lexer.next_token()?; // consume "..."
        // inline fragment or named spread?
        let peeked = lexer.peek_token()?;
        match peeked {
            Token::Name(ref n) if n != "on" => {
                let name = lexer.expect_name()?;
                let directives = parse_directives(lexer)?;
                Ok(Selection::FragmentSpread(FragmentSpread { name, directives }))
            }
            _ => {
                // inline fragment
                let type_condition = if matches!(lexer.peek_token()?, Token::Name(ref n) if n == "on")
                {
                    lexer.next_token()?; // consume "on"
                    Some(lexer.expect_name()?)
                } else {
                    None
                };
                let directives = parse_directives(lexer)?;
                let selection_set = parse_selection_set(lexer)?;
                Ok(Selection::InlineFragment(InlineFragment {
                    type_condition,
                    directives,
                    selection_set,
                }))
            }
        }
    } else {
        parse_field(lexer).map(Selection::Field)
    }
}

fn parse_field(lexer: &mut Lexer) -> Result<Field, ParseError> {
    let first_name = lexer.expect_name()?;
    // check for alias
    let (alias, name) = if matches!(lexer.peek_token()?, Token::Colon) {
        lexer.next_token()?; // consume ':'
        let actual_name = lexer.expect_name()?;
        (Some(first_name), actual_name)
    } else {
        (None, first_name)
    };
    let arguments = if matches!(lexer.peek_token()?, Token::Punctuation('(')) {
        parse_arguments(lexer)?
    } else {
        Vec::new()
    };
    let directives = parse_directives(lexer)?;
    let selection_set = if matches!(lexer.peek_token()?, Token::Punctuation('{')) {
        Some(parse_selection_set(lexer)?)
    } else {
        None
    };
    Ok(Field {
        alias,
        name,
        arguments,
        directives,
        selection_set,
    })
}

fn parse_arguments(lexer: &mut Lexer) -> Result<Vec<Argument>, ParseError> {
    lexer.expect_punctuation('(')?;
    let mut args = Vec::new();
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation(')')) {
            lexer.next_token()?;
            break;
        }
        let name = lexer.expect_name()?;
        match lexer.next_token()? {
            Token::Colon => {}
            other => return Err(lexer.error(&format!("expected ':', got {other:?}"))),
        }
        let value = parse_value(lexer)?;
        args.push(Argument { name, value });
    }
    Ok(args)
}

fn parse_directives(lexer: &mut Lexer) -> Result<Vec<Directive>, ParseError> {
    let mut directives = Vec::new();
    while matches!(lexer.peek_token()?, Token::At) {
        lexer.next_token()?; // consume '@'
        let name = lexer.expect_name()?;
        let arguments = if matches!(lexer.peek_token()?, Token::Punctuation('(')) {
            parse_arguments(lexer)?
        } else {
            Vec::new()
        };
        directives.push(Directive { name, arguments });
    }
    Ok(directives)
}

fn parse_value(lexer: &mut Lexer) -> Result<Value, ParseError> {
    let tok = lexer.peek_token()?;
    match tok {
        Token::Dollar => {
            lexer.next_token()?;
            let name = lexer.expect_name()?;
            Ok(Value::Variable(name))
        }
        Token::IntValue(_) => {
            if let Token::IntValue(v) = lexer.next_token()? {
                Ok(Value::Int(v))
            } else {
                unreachable!()
            }
        }
        Token::FloatValue(_) => {
            if let Token::FloatValue(v) = lexer.next_token()? {
                Ok(Value::Float(v))
            } else {
                unreachable!()
            }
        }
        Token::StringValue(_) => {
            if let Token::StringValue(s) = lexer.next_token()? {
                Ok(Value::StringValue(s))
            } else {
                unreachable!()
            }
        }
        Token::Name(ref n) if n == "true" => {
            lexer.next_token()?;
            Ok(Value::Boolean(true))
        }
        Token::Name(ref n) if n == "false" => {
            lexer.next_token()?;
            Ok(Value::Boolean(false))
        }
        Token::Name(ref n) if n == "null" => {
            lexer.next_token()?;
            Ok(Value::Null)
        }
        Token::Name(_) => {
            let name = lexer.expect_name()?;
            Ok(Value::Enum(name))
        }
        Token::Punctuation('[') => {
            lexer.next_token()?;
            let mut items = Vec::new();
            loop {
                if matches!(lexer.peek_token()?, Token::Punctuation(']')) {
                    lexer.next_token()?;
                    break;
                }
                items.push(parse_value(lexer)?);
            }
            Ok(Value::List(items))
        }
        Token::Punctuation('{') => {
            lexer.next_token()?;
            let mut fields = Vec::new();
            loop {
                if matches!(lexer.peek_token()?, Token::Punctuation('}')) {
                    lexer.next_token()?;
                    break;
                }
                let key = lexer.expect_name()?;
                match lexer.next_token()? {
                    Token::Colon => {}
                    other => {
                        return Err(lexer.error(&format!("expected ':', got {other:?}")));
                    }
                }
                let val = parse_value(lexer)?;
                fields.push((key, val));
            }
            Ok(Value::Object(fields))
        }
        _ => Err(lexer.error(&format!("unexpected token in value: {tok:?}"))),
    }
}

fn parse_fragment_definition(lexer: &mut Lexer) -> Result<FragmentDefinition, ParseError> {
    lexer.expect_name()?; // consume "fragment"
    let name = lexer.expect_name()?;
    let on_kw = lexer.expect_name()?;
    if on_kw != "on" {
        return Err(lexer.error(&format!("expected 'on', got '{on_kw}'")));
    }
    let type_condition = lexer.expect_name()?;
    let directives = parse_directives(lexer)?;
    let selection_set = parse_selection_set(lexer)?;
    Ok(FragmentDefinition {
        name,
        type_condition,
        directives,
        selection_set,
    })
}

// ── Type System Parsing ──────────────────────────────────────────

fn parse_optional_description(lexer: &mut Lexer) -> Result<Option<String>, ParseError> {
    if matches!(lexer.peek_token()?, Token::StringValue(_)) {
        if let Token::StringValue(s) = lexer.next_token()? {
            return Ok(Some(s));
        }
    }
    Ok(None)
}

fn parse_type_system_definition(
    lexer: &mut Lexer,
) -> Result<TypeSystemDefinition, ParseError> {
    let description = parse_optional_description(lexer)?;
    let keyword = lexer.expect_name()?;
    match keyword.as_str() {
        "schema" => parse_schema_definition(lexer).map(TypeSystemDefinition::Schema),
        "scalar" => parse_scalar_definition(lexer, description).map(TypeSystemDefinition::Scalar),
        "type" => parse_object_definition(lexer, description).map(TypeSystemDefinition::Object),
        "interface" => {
            parse_interface_definition(lexer, description).map(TypeSystemDefinition::Interface)
        }
        "union" => parse_union_definition(lexer, description).map(TypeSystemDefinition::Union),
        "enum" => parse_enum_definition(lexer, description).map(TypeSystemDefinition::Enum),
        "input" => {
            parse_input_object_definition(lexer, description)
                .map(TypeSystemDefinition::InputObject)
        }
        _ => Err(lexer.error(&format!("unexpected type system keyword: {keyword}"))),
    }
}

fn parse_schema_definition(lexer: &mut Lexer) -> Result<SchemaDefinition, ParseError> {
    let directives = parse_directives(lexer)?;
    lexer.expect_punctuation('{')?;
    let mut query = None;
    let mut mutation = None;
    let mut subscription = None;
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation('}')) {
            lexer.next_token()?;
            break;
        }
        let key = lexer.expect_name()?;
        match lexer.next_token()? {
            Token::Colon => {}
            other => return Err(lexer.error(&format!("expected ':', got {other:?}"))),
        }
        let val = lexer.expect_name()?;
        match key.as_str() {
            "query" => query = Some(val),
            "mutation" => mutation = Some(val),
            "subscription" => subscription = Some(val),
            _ => return Err(lexer.error(&format!("unexpected schema field: {key}"))),
        }
    }
    Ok(SchemaDefinition {
        directives,
        query,
        mutation,
        subscription,
    })
}

fn parse_scalar_definition(
    lexer: &mut Lexer,
    description: Option<String>,
) -> Result<ScalarDefinition, ParseError> {
    let name = lexer.expect_name()?;
    let directives = parse_directives(lexer)?;
    Ok(ScalarDefinition {
        name,
        description,
        directives,
    })
}

fn parse_object_definition(
    lexer: &mut Lexer,
    description: Option<String>,
) -> Result<ObjectDefinition, ParseError> {
    let name = lexer.expect_name()?;
    let mut interfaces = Vec::new();
    if matches!(lexer.peek_token()?, Token::Name(ref n) if n == "implements") {
        lexer.next_token()?;
        // optional leading '&'
        if matches!(lexer.peek_token()?, Token::Amp) {
            lexer.next_token()?;
        }
        interfaces.push(lexer.expect_name()?);
        while matches!(lexer.peek_token()?, Token::Amp) {
            lexer.next_token()?;
            interfaces.push(lexer.expect_name()?);
        }
    }
    let directives = parse_directives(lexer)?;
    let fields = parse_field_definitions(lexer)?;
    Ok(ObjectDefinition {
        name,
        description,
        interfaces,
        directives,
        fields,
    })
}

fn parse_interface_definition(
    lexer: &mut Lexer,
    description: Option<String>,
) -> Result<InterfaceDefinition, ParseError> {
    let name = lexer.expect_name()?;
    let directives = parse_directives(lexer)?;
    let fields = parse_field_definitions(lexer)?;
    Ok(InterfaceDefinition {
        name,
        description,
        directives,
        fields,
    })
}

fn parse_union_definition(
    lexer: &mut Lexer,
    description: Option<String>,
) -> Result<UnionDefinition, ParseError> {
    let name = lexer.expect_name()?;
    let directives = parse_directives(lexer)?;
    let mut members = Vec::new();
    match lexer.next_token()? {
        Token::Equals => {}
        other => return Err(lexer.error(&format!("expected '=', got {other:?}"))),
    }
    // optional leading pipe
    if matches!(lexer.peek_token()?, Token::Pipe) {
        lexer.next_token()?;
    }
    members.push(lexer.expect_name()?);
    while matches!(lexer.peek_token()?, Token::Pipe) {
        lexer.next_token()?;
        members.push(lexer.expect_name()?);
    }
    Ok(UnionDefinition {
        name,
        description,
        directives,
        members,
    })
}

fn parse_enum_definition(
    lexer: &mut Lexer,
    description: Option<String>,
) -> Result<EnumDefinition, ParseError> {
    let name = lexer.expect_name()?;
    let directives = parse_directives(lexer)?;
    lexer.expect_punctuation('{')?;
    let mut values = Vec::new();
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation('}')) {
            lexer.next_token()?;
            break;
        }
        let val_desc = parse_optional_description(lexer)?;
        let val_name = lexer.expect_name()?;
        let val_directives = parse_directives(lexer)?;
        values.push(EnumValueDefinition {
            name: val_name,
            description: val_desc,
            directives: val_directives,
        });
    }
    Ok(EnumDefinition {
        name,
        description,
        directives,
        values,
    })
}

fn parse_input_object_definition(
    lexer: &mut Lexer,
    description: Option<String>,
) -> Result<InputObjectDefinition, ParseError> {
    let name = lexer.expect_name()?;
    let directives = parse_directives(lexer)?;
    lexer.expect_punctuation('{')?;
    let mut fields = Vec::new();
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation('}')) {
            lexer.next_token()?;
            break;
        }
        fields.push(parse_input_value_definition(lexer)?);
    }
    Ok(InputObjectDefinition {
        name,
        description,
        directives,
        fields,
    })
}

fn parse_field_definitions(lexer: &mut Lexer) -> Result<Vec<FieldDefinition>, ParseError> {
    lexer.expect_punctuation('{')?;
    let mut fields = Vec::new();
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation('}')) {
            lexer.next_token()?;
            break;
        }
        let desc = parse_optional_description(lexer)?;
        let fname = lexer.expect_name()?;
        let arguments = if matches!(lexer.peek_token()?, Token::Punctuation('(')) {
            parse_input_value_definitions_parens(lexer)?
        } else {
            Vec::new()
        };
        match lexer.next_token()? {
            Token::Colon => {}
            other => return Err(lexer.error(&format!("expected ':', got {other:?}"))),
        }
        let field_type = parse_type_ref(lexer)?;
        let dir = parse_directives(lexer)?;
        fields.push(FieldDefinition {
            name: fname,
            description: desc,
            arguments,
            field_type,
            directives: dir,
        });
    }
    Ok(fields)
}

fn parse_input_value_definitions_parens(
    lexer: &mut Lexer,
) -> Result<Vec<InputValueDefinition>, ParseError> {
    lexer.expect_punctuation('(')?;
    let mut defs = Vec::new();
    loop {
        if matches!(lexer.peek_token()?, Token::Punctuation(')')) {
            lexer.next_token()?;
            break;
        }
        defs.push(parse_input_value_definition(lexer)?);
    }
    Ok(defs)
}

fn parse_input_value_definition(lexer: &mut Lexer) -> Result<InputValueDefinition, ParseError> {
    let desc = parse_optional_description(lexer)?;
    let name = lexer.expect_name()?;
    match lexer.next_token()? {
        Token::Colon => {}
        other => return Err(lexer.error(&format!("expected ':', got {other:?}"))),
    }
    let input_type = parse_type_ref(lexer)?;
    let default_value = if matches!(lexer.peek_token()?, Token::Equals) {
        lexer.next_token()?;
        Some(parse_value(lexer)?)
    } else {
        None
    };
    let directives = parse_directives(lexer)?;
    Ok(InputValueDefinition {
        name,
        description: desc,
        input_type,
        default_value,
        directives,
    })
}

// ── Validation ───────────────────────────────────────────────────

/// Validation error.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidationError {
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Validate a parsed document for common issues:
/// - undefined fragment references
/// - unused fragments
/// - duplicate operation names
/// - duplicate fragment names
pub fn validate(doc: &Document) -> Vec<ValidationError> {
    let mut errors = Vec::new();

    // Collect fragment names
    let mut fragment_names: HashMap<String, usize> = HashMap::new();
    let mut operation_names: HashMap<String, usize> = HashMap::new();

    for def in &doc.definitions {
        match def {
            Definition::Fragment(f) => {
                let count = fragment_names.entry(f.name.clone()).or_insert(0);
                *count += 1;
            }
            Definition::Operation(op) => {
                if let Some(ref name) = op.name {
                    let count = operation_names.entry(name.clone()).or_insert(0);
                    *count += 1;
                }
            }
            _ => {}
        }
    }

    // Check duplicate operation names
    for (name, count) in &operation_names {
        if *count > 1 {
            errors.push(ValidationError {
                message: format!("duplicate operation name: {name}"),
            });
        }
    }

    // Check duplicate fragment names
    for (name, count) in &fragment_names {
        if *count > 1 {
            errors.push(ValidationError {
                message: format!("duplicate fragment name: {name}"),
            });
        }
    }

    // Collect all spread references
    let mut referenced_fragments: HashMap<String, bool> = HashMap::new();
    for def in &doc.definitions {
        match def {
            Definition::Operation(op) => {
                collect_fragment_refs(&op.selection_set, &mut referenced_fragments);
            }
            Definition::Fragment(f) => {
                collect_fragment_refs(&f.selection_set, &mut referenced_fragments);
            }
            _ => {}
        }
    }

    // Check undefined references
    for name in referenced_fragments.keys() {
        if !fragment_names.contains_key(name) {
            errors.push(ValidationError {
                message: format!("undefined fragment: {name}"),
            });
        }
    }

    // Check unused fragments
    for name in fragment_names.keys() {
        if !referenced_fragments.contains_key(name) {
            errors.push(ValidationError {
                message: format!("unused fragment: {name}"),
            });
        }
    }

    errors
}

fn collect_fragment_refs(ss: &SelectionSet, refs: &mut HashMap<String, bool>) {
    for sel in &ss.selections {
        match sel {
            Selection::FragmentSpread(spread) => {
                refs.insert(spread.name.clone(), true);
            }
            Selection::Field(f) => {
                if let Some(ref sub) = f.selection_set {
                    collect_fragment_refs(sub, refs);
                }
            }
            Selection::InlineFragment(inf) => {
                collect_fragment_refs(&inf.selection_set, refs);
            }
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_query() {
        let doc = parse("{ hero { name } }").unwrap();
        assert_eq!(doc.definitions.len(), 1);
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.operation, OperationType::Query);
            assert!(op.name.is_none());
            assert_eq!(op.selection_set.selections.len(), 1);
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_named_query() {
        let doc = parse("query HeroQuery { hero { name } }").unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.name.as_deref(), Some("HeroQuery"));
            assert_eq!(op.operation, OperationType::Query);
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_mutation() {
        let doc = parse("mutation CreateUser { createUser(name: \"Alice\") { id } }").unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.operation, OperationType::Mutation);
            assert_eq!(op.name.as_deref(), Some("CreateUser"));
        } else {
            panic!("expected mutation");
        }
    }

    #[test]
    fn parse_subscription() {
        let doc = parse("subscription OnMessage { messageAdded { text } }").unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.operation, OperationType::Subscription);
        } else {
            panic!("expected subscription");
        }
    }

    #[test]
    fn parse_variables() {
        let doc = parse("query Get($id: ID!, $limit: Int = 10) { user(id: $id) { name } }")
            .unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.variables.len(), 2);
            assert_eq!(op.variables[0].name, "id");
            assert_eq!(op.variables[0].var_type, TypeRef::NonNull(Box::new(TypeRef::Named("ID".into()))));
            assert_eq!(op.variables[1].name, "limit");
            assert_eq!(op.variables[1].default_value, Some(Value::Int(10)));
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_field_alias() {
        let doc = parse("{ smallPic: profilePic(size: 64) }").unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(f) = &op.selection_set.selections[0] {
                assert_eq!(f.alias.as_deref(), Some("smallPic"));
                assert_eq!(f.name, "profilePic");
                assert_eq!(f.arguments.len(), 1);
                assert_eq!(f.arguments[0].name, "size");
                assert_eq!(f.arguments[0].value, Value::Int(64));
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_fragment_spread() {
        let input = "query { user { ...UserFields } } fragment UserFields on User { name email }";
        let doc = parse(input).unwrap();
        assert_eq!(doc.definitions.len(), 2);
        if let Definition::Fragment(f) = &doc.definitions[1] {
            assert_eq!(f.name, "UserFields");
            assert_eq!(f.type_condition, "User");
            assert_eq!(f.selection_set.selections.len(), 2);
        } else {
            panic!("expected fragment");
        }
    }

    #[test]
    fn parse_inline_fragment() {
        let input = "{ hero { name ... on Droid { primaryFunction } } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(hero) = &op.selection_set.selections[0] {
                let sels = hero.selection_set.as_ref().unwrap();
                assert_eq!(sels.selections.len(), 2);
                if let Selection::InlineFragment(inf) = &sels.selections[1] {
                    assert_eq!(inf.type_condition.as_deref(), Some("Droid"));
                } else {
                    panic!("expected inline fragment");
                }
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_directives() {
        let input = "query @cached(maxAge: 60) { user { name @uppercase } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            assert_eq!(op.directives.len(), 1);
            assert_eq!(op.directives[0].name, "cached");
            assert_eq!(op.directives[0].arguments.len(), 1);
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_list_values() {
        let input = "{ users(ids: [1, 2, 3]) { name } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(f) = &op.selection_set.selections[0] {
                assert_eq!(f.arguments[0].value, Value::List(vec![
                    Value::Int(1), Value::Int(2), Value::Int(3),
                ]));
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_object_value() {
        let input = "{ createUser(input: {name: \"Alice\", age: 30}) { id } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(f) = &op.selection_set.selections[0] {
                if let Value::Object(fields) = &f.arguments[0].value {
                    assert_eq!(fields.len(), 2);
                    assert_eq!(fields[0].0, "name");
                    assert_eq!(fields[1].0, "age");
                } else {
                    panic!("expected object value");
                }
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_enum_value() {
        let input = "{ users(role: ADMIN) { name } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(f) = &op.selection_set.selections[0] {
                assert_eq!(f.arguments[0].value, Value::Enum("ADMIN".into()));
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_boolean_and_null() {
        let input = "{ user(active: true, deleted: false, alias: null) { name } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(f) = &op.selection_set.selections[0] {
                assert_eq!(f.arguments[0].value, Value::Boolean(true));
                assert_eq!(f.arguments[1].value, Value::Boolean(false));
                assert_eq!(f.arguments[2].value, Value::Null);
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_scalar_type() {
        let input = "scalar DateTime";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Scalar(s)) = &doc.definitions[0] {
            assert_eq!(s.name, "DateTime");
        } else {
            panic!("expected scalar");
        }
    }

    #[test]
    fn parse_object_type() {
        let input = "type User implements Node { id: ID! name: String age: Int }";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Object(obj)) = &doc.definitions[0] {
            assert_eq!(obj.name, "User");
            assert_eq!(obj.interfaces, vec!["Node".to_string()]);
            assert_eq!(obj.fields.len(), 3);
            assert_eq!(obj.fields[0].name, "id");
        } else {
            panic!("expected object type");
        }
    }

    #[test]
    fn parse_enum_type() {
        let input = "enum Role { ADMIN USER GUEST }";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Enum(e)) = &doc.definitions[0] {
            assert_eq!(e.name, "Role");
            assert_eq!(e.values.len(), 3);
            assert_eq!(e.values[0].name, "ADMIN");
        } else {
            panic!("expected enum type");
        }
    }

    #[test]
    fn parse_union_type() {
        let input = "union SearchResult = User | Post | Comment";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Union(u)) = &doc.definitions[0] {
            assert_eq!(u.name, "SearchResult");
            assert_eq!(u.members, vec!["User", "Post", "Comment"]);
        } else {
            panic!("expected union type");
        }
    }

    #[test]
    fn parse_input_type() {
        let input = "input CreateUserInput { name: String! email: String }";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::InputObject(inp)) = &doc.definitions[0]
        {
            assert_eq!(inp.name, "CreateUserInput");
            assert_eq!(inp.fields.len(), 2);
        } else {
            panic!("expected input type");
        }
    }

    #[test]
    fn parse_interface_type() {
        let input = "interface Node { id: ID! }";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Interface(i)) = &doc.definitions[0] {
            assert_eq!(i.name, "Node");
            assert_eq!(i.fields.len(), 1);
        } else {
            panic!("expected interface");
        }
    }

    #[test]
    fn validate_undefined_fragment() {
        let doc = parse("{ user { ...MissingFragment } }").unwrap();
        let errors = validate(&doc);
        assert!(errors.iter().any(|e| e.message.contains("undefined fragment")));
    }

    #[test]
    fn validate_unused_fragment() {
        let input = "query { user { name } } fragment Unused on User { email }";
        let doc = parse(input).unwrap();
        let errors = validate(&doc);
        assert!(errors.iter().any(|e| e.message.contains("unused fragment")));
    }

    #[test]
    fn validate_duplicate_operation_names() {
        let input = "query Foo { a } query Foo { b }";
        let doc = parse(input).unwrap();
        let errors = validate(&doc);
        assert!(errors.iter().any(|e| e.message.contains("duplicate operation name")));
    }

    #[test]
    fn parse_comments_ignored() {
        let input = "# This is a comment\n{ hero { name } }";
        let doc = parse(input).unwrap();
        assert_eq!(doc.definitions.len(), 1);
    }

    #[test]
    fn parse_nested_selections() {
        let input = "{ user { friends { name posts { title } } } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(user) = &op.selection_set.selections[0] {
                let friends_sel = user.selection_set.as_ref().unwrap();
                if let Selection::Field(friends) = &friends_sel.selections[0] {
                    assert_eq!(friends.name, "friends");
                    let inner = friends.selection_set.as_ref().unwrap();
                    assert_eq!(inner.selections.len(), 2);
                } else {
                    panic!("expected friends field");
                }
            } else {
                panic!("expected user field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_error_unterminated() {
        let result = parse("{ hero { name");
        assert!(result.is_err());
    }

    #[test]
    fn parse_schema_definition() {
        let input = "schema { query: Query mutation: Mutation }";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Schema(s)) = &doc.definitions[0] {
            assert_eq!(s.query.as_deref(), Some("Query"));
            assert_eq!(s.mutation.as_deref(), Some("Mutation"));
            assert!(s.subscription.is_none());
        } else {
            panic!("expected schema definition");
        }
    }

    #[test]
    fn type_ref_display() {
        let t = TypeRef::NonNull(Box::new(TypeRef::List(Box::new(TypeRef::Named("String".into())))));
        assert_eq!(t.to_string(), "[String]!");
    }

    #[test]
    fn parse_float_value() {
        let input = "{ field(x: 3.14) { name } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(f) = &op.selection_set.selections[0] {
                if let Value::Float(v) = f.arguments[0].value {
                    assert!((v - 3.14).abs() < 1e-10);
                } else {
                    panic!("expected float");
                }
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_string_with_escapes() {
        let input = "{ field(msg: \"hello\\nworld\") { ok } }";
        let doc = parse(input).unwrap();
        if let Definition::Operation(op) = &doc.definitions[0] {
            if let Selection::Field(f) = &op.selection_set.selections[0] {
                assert_eq!(f.arguments[0].value, Value::StringValue("hello\nworld".into()));
            } else {
                panic!("expected field");
            }
        } else {
            panic!("expected operation");
        }
    }

    #[test]
    fn parse_description_on_type() {
        let input = "\"A user in the system\" type User { name: String }";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Object(obj)) = &doc.definitions[0] {
            assert_eq!(obj.description.as_deref(), Some("A user in the system"));
        } else {
            panic!("expected object");
        }
    }

    #[test]
    fn parse_field_arguments_in_type() {
        let input = "type Query { user(id: ID!): User }";
        let doc = parse(input).unwrap();
        if let Definition::TypeSystem(TypeSystemDefinition::Object(obj)) = &doc.definitions[0] {
            assert_eq!(obj.fields[0].arguments.len(), 1);
            assert_eq!(obj.fields[0].arguments[0].name, "id");
        } else {
            panic!("expected object");
        }
    }
}
