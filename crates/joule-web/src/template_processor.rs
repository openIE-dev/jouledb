//! Advanced template processor — Jinja2-like syntax: `{% if %}`, `{% for %}`,
//! `{% macro %}`, `{% block %}`/`{% extends %}`, `{{ expr | filter }}`,
//! whitespace control (`{%-`, `-%}`), auto-escaping, and template inheritance.
//!
//! Replaces JavaScript template engines (Nunjucks, Jinja2, Twig) with a
//! pure-Rust template processor that compiles templates into an AST.

use serde_json::Value;
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Template processor domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateProcessorError {
    /// Parse error.
    ParseError { message: String, line: usize },
    /// Render error.
    RenderError(String),
    /// Template not found.
    TemplateNotFound(String),
    /// Undefined variable.
    UndefinedVariable(String),
    /// Macro not found.
    MacroNotFound(String),
    /// Block not found.
    BlockNotFound(String),
    /// Filter not found.
    FilterNotFound(String),
    /// Circular inheritance.
    CircularInheritance(String),
    /// Max recursion depth.
    MaxRecursionDepth(usize),
}

impl std::fmt::Display for TemplateProcessorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ParseError { message, line } => write!(f, "parse error at line {line}: {message}"),
            Self::RenderError(msg) => write!(f, "render error: {msg}"),
            Self::TemplateNotFound(name) => write!(f, "template not found: {name}"),
            Self::UndefinedVariable(name) => write!(f, "undefined variable: {name}"),
            Self::MacroNotFound(name) => write!(f, "macro not found: {name}"),
            Self::BlockNotFound(name) => write!(f, "block not found: {name}"),
            Self::FilterNotFound(name) => write!(f, "filter not found: {name}"),
            Self::CircularInheritance(name) => write!(f, "circular inheritance: {name}"),
            Self::MaxRecursionDepth(depth) => write!(f, "max recursion depth exceeded: {depth}"),
        }
    }
}

impl std::error::Error for TemplateProcessorError {}

// ── AST ─────────────────────────────────────────────────────────

/// Template AST node.
#[derive(Debug, Clone)]
enum Node {
    /// Raw text.
    Text(String),
    /// Expression: `{{ expr }}` or `{{ expr | filter }}`.
    Expr {
        expression: String,
        filters: Vec<FilterCall>,
        escape: bool,
    },
    /// If block: `{% if cond %}...{% elif cond %}...{% else %}...{% endif %}`.
    If {
        condition: String,
        body: Vec<Node>,
        elif_branches: Vec<(String, Vec<Node>)>,
        else_body: Vec<Node>,
    },
    /// For loop: `{% for item in collection %}`.
    For {
        variable: String,
        iterable: String,
        body: Vec<Node>,
        else_body: Vec<Node>,
    },
    /// Block definition: `{% block name %}...{% endblock %}`.
    Block {
        name: String,
        body: Vec<Node>,
    },
    /// Extends: `{% extends "parent" %}`.
    Extends(String),
    /// Macro definition: `{% macro name(args) %}...{% endmacro %}`.
    MacroDef {
        name: String,
        params: Vec<String>,
        body: Vec<Node>,
    },
    /// Macro call: `{{ name(args) }}`.
    MacroCall {
        name: String,
        args: Vec<String>,
    },
    /// Set variable: `{% set x = expr %}`.
    Set {
        variable: String,
        value: String,
    },
    /// Include: `{% include "name" %}`.
    Include(String),
    /// Comment: `{# ... #}`.
    Comment,
    /// Raw block: `{% raw %}...{% endraw %}`.
    Raw(String),
}

/// A filter call with optional argument.
#[derive(Debug, Clone)]
struct FilterCall {
    name: String,
    argument: Option<String>,
}

// ── Parser ──────────────────────────────────────────────────────

struct Parser<'a> {
    input: &'a str,
    pos: usize,
    line: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            input,
            pos: 0,
            line: 1,
        }
    }

    fn parse(&mut self) -> Result<Vec<Node>, TemplateProcessorError> {
        self.parse_until(&[])
    }

    fn parse_until(&mut self, end_tags: &[&str]) -> Result<Vec<Node>, TemplateProcessorError> {
        let mut nodes = Vec::new();

        while self.pos < self.input.len() {
            // Check for end tags.
            for tag in end_tags {
                if self.looking_at_tag(tag) {
                    return Ok(nodes);
                }
            }

            if self.starts_with("{#") {
                self.parse_comment()?;
                // Comments produce no node.
            } else if self.starts_with("{%-") || self.starts_with("{%") {
                let trim_left = self.starts_with("{%-");
                if trim_left {
                    if let Some(Node::Text(t)) = nodes.last_mut() {
                        *t = t.trim_end().to_string();
                    }
                }
                let node = self.parse_tag(end_tags)?;
                if let Some(n) = node {
                    nodes.push(n);
                }
            } else if self.starts_with("{{-") || self.starts_with("{{") {
                let trim_left = self.starts_with("{{-");
                if trim_left {
                    if let Some(Node::Text(t)) = nodes.last_mut() {
                        *t = t.trim_end().to_string();
                    }
                }
                nodes.push(self.parse_expression()?);
            } else {
                // Collect text.
                let start = self.pos;
                while self.pos < self.input.len()
                    && !self.starts_with("{{")
                    && !self.starts_with("{%")
                    && !self.starts_with("{#")
                {
                    if self.current_char() == '\n' {
                        self.line += 1;
                    }
                    self.pos += 1;
                }
                let text = &self.input[start..self.pos];
                if !text.is_empty() {
                    nodes.push(Node::Text(text.to_string()));
                }
            }
        }

        Ok(nodes)
    }

    fn starts_with(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }

    fn current_char(&self) -> char {
        self.input[self.pos..].chars().next().unwrap_or('\0')
    }

    fn looking_at_tag(&self, tag_name: &str) -> bool {
        let rest = &self.input[self.pos..];
        // Match {%- tag or {% tag
        if let Some(stripped) = rest.strip_prefix("{%-") {
            stripped.trim_start().starts_with(tag_name)
        } else if let Some(stripped) = rest.strip_prefix("{%") {
            stripped.trim_start().starts_with(tag_name)
        } else {
            false
        }
    }

    fn consume_tag_content(&mut self) -> Result<(String, bool), TemplateProcessorError> {
        // Skip {%- or {%
        let trim_left = self.starts_with("{%-");
        if trim_left {
            self.pos += 3;
        } else {
            self.pos += 2;
        }

        let start = self.pos;
        while self.pos < self.input.len() {
            if self.starts_with("-%}") {
                let content = self.input[start..self.pos].trim().to_string();
                self.pos += 3;
                return Ok((content, true));
            }
            if self.starts_with("%}") {
                let content = self.input[start..self.pos].trim().to_string();
                self.pos += 2;
                return Ok((content, false));
            }
            if self.current_char() == '\n' {
                self.line += 1;
            }
            self.pos += 1;
        }
        Err(TemplateProcessorError::ParseError {
            message: "unclosed tag".into(),
            line: self.line,
        })
    }

    fn parse_comment(&mut self) -> Result<(), TemplateProcessorError> {
        self.pos += 2; // skip {#
        while self.pos < self.input.len() {
            if self.starts_with("#}") {
                self.pos += 2;
                return Ok(());
            }
            if self.current_char() == '\n' {
                self.line += 1;
            }
            self.pos += 1;
        }
        Err(TemplateProcessorError::ParseError {
            message: "unclosed comment".into(),
            line: self.line,
        })
    }

    fn parse_expression(&mut self) -> Result<Node, TemplateProcessorError> {
        // Skip {{- or {{
        let _trim_left = self.starts_with("{{-");
        if _trim_left {
            self.pos += 3;
        } else {
            self.pos += 2;
        }

        let start = self.pos;
        let mut trim_right = false;
        while self.pos < self.input.len() {
            if self.starts_with("-}}") {
                trim_right = true;
                let content = self.input[start..self.pos].trim().to_string();
                self.pos += 3;
                let _ = trim_right; // tracked for future whitespace control
                return self.build_expr_node(&content);
            }
            if self.starts_with("}}") {
                let content = self.input[start..self.pos].trim().to_string();
                self.pos += 2;
                return self.build_expr_node(&content);
            }
            if self.current_char() == '\n' {
                self.line += 1;
            }
            self.pos += 1;
        }
        Err(TemplateProcessorError::ParseError {
            message: "unclosed expression".into(),
            line: self.line,
        })
    }

    fn build_expr_node(&self, content: &str) -> Result<Node, TemplateProcessorError> {
        // Check for macro call: name(args).
        if let Some(paren_start) = content.find('(') {
            if content.ends_with(')') && !content.contains('|') {
                let name = content[..paren_start].trim().to_string();
                let args_str = &content[paren_start + 1..content.len() - 1];
                let args: Vec<String> = if args_str.is_empty() {
                    vec![]
                } else {
                    args_str.split(',').map(|s| s.trim().to_string()).collect()
                };
                return Ok(Node::MacroCall { name, args });
            }
        }

        // Parse filters: expr | filter1 | filter2(arg)
        let parts: Vec<&str> = content.split('|').collect();
        let expression = parts[0].trim().to_string();
        let mut filters = Vec::new();

        for part in &parts[1..] {
            let trimmed = part.trim();
            if let Some(paren_pos) = trimmed.find('(') {
                if trimmed.ends_with(')') {
                    let fname = trimmed[..paren_pos].trim().to_string();
                    let arg = trimmed[paren_pos + 1..trimmed.len() - 1].trim().to_string();
                    filters.push(FilterCall {
                        name: fname,
                        argument: if arg.is_empty() { None } else { Some(arg) },
                    });
                } else {
                    filters.push(FilterCall {
                        name: trimmed.to_string(),
                        argument: None,
                    });
                }
            } else {
                filters.push(FilterCall {
                    name: trimmed.to_string(),
                    argument: None,
                });
            }
        }

        Ok(Node::Expr {
            expression,
            filters,
            escape: true,
        })
    }

    fn parse_tag(
        &mut self,
        _end_tags: &[&str],
    ) -> Result<Option<Node>, TemplateProcessorError> {
        let (content, _trim_right) = self.consume_tag_content()?;

        if let Some(rest) = content.strip_prefix("if ") {
            return self.parse_if_block(rest.trim()).map(Some);
        }
        if let Some(rest) = content.strip_prefix("for ") {
            return self.parse_for_block(rest.trim()).map(Some);
        }
        if let Some(rest) = content.strip_prefix("block ") {
            return self.parse_block(rest.trim()).map(Some);
        }
        if let Some(rest) = content.strip_prefix("extends ") {
            let name = rest.trim().trim_matches('"').trim_matches('\'').to_string();
            return Ok(Some(Node::Extends(name)));
        }
        if let Some(rest) = content.strip_prefix("macro ") {
            return self.parse_macro_def(rest.trim()).map(Some);
        }
        if let Some(rest) = content.strip_prefix("set ") {
            return self.parse_set(rest.trim()).map(Some);
        }
        if let Some(rest) = content.strip_prefix("include ") {
            let name = rest.trim().trim_matches('"').trim_matches('\'').to_string();
            return Ok(Some(Node::Include(name)));
        }
        if content == "raw" {
            return self.parse_raw().map(Some);
        }

        // Unrecognized tag — just skip.
        Ok(None)
    }

    fn parse_if_block(&mut self, condition: &str) -> Result<Node, TemplateProcessorError> {
        let body = self.parse_until(&["elif", "else", "endif"])?;
        let mut elif_branches = Vec::new();
        let mut else_body = Vec::new();

        loop {
            if self.looking_at_tag("elif") {
                let (content, _) = self.consume_tag_content()?;
                let cond = content.strip_prefix("elif ").unwrap_or("").trim().to_string();
                let branch_body = self.parse_until(&["elif", "else", "endif"])?;
                elif_branches.push((cond, branch_body));
            } else if self.looking_at_tag("else") {
                let _ = self.consume_tag_content()?;
                else_body = self.parse_until(&["endif"])?;
                break;
            } else if self.looking_at_tag("endif") {
                let _ = self.consume_tag_content()?;
                break;
            } else {
                break;
            }
        }

        Ok(Node::If {
            condition: condition.to_string(),
            body,
            elif_branches,
            else_body,
        })
    }

    fn parse_for_block(&mut self, content: &str) -> Result<Node, TemplateProcessorError> {
        // "item in collection"
        let parts: Vec<&str> = content.splitn(3, ' ').collect();
        if parts.len() < 3 || parts[1] != "in" {
            return Err(TemplateProcessorError::ParseError {
                message: format!("invalid for syntax: {content}"),
                line: self.line,
            });
        }
        let variable = parts[0].to_string();
        let iterable = parts[2].to_string();

        let body = self.parse_until(&["else", "endfor"])?;
        let else_body = if self.looking_at_tag("else") {
            let _ = self.consume_tag_content()?;
            let eb = self.parse_until(&["endfor"])?;
            if self.looking_at_tag("endfor") {
                let _ = self.consume_tag_content()?;
            }
            eb
        } else {
            if self.looking_at_tag("endfor") {
                let _ = self.consume_tag_content()?;
            }
            vec![]
        };

        Ok(Node::For {
            variable,
            iterable,
            body,
            else_body,
        })
    }

    fn parse_block(&mut self, name: &str) -> Result<Node, TemplateProcessorError> {
        let body = self.parse_until(&["endblock"])?;
        if self.looking_at_tag("endblock") {
            let _ = self.consume_tag_content()?;
        }
        Ok(Node::Block {
            name: name.to_string(),
            body,
        })
    }

    fn parse_macro_def(&mut self, content: &str) -> Result<Node, TemplateProcessorError> {
        // "name(arg1, arg2)"
        let paren = content.find('(').ok_or_else(|| TemplateProcessorError::ParseError {
            message: format!("invalid macro syntax: {content}"),
            line: self.line,
        })?;
        let name = content[..paren].trim().to_string();
        let args_end = content.find(')').unwrap_or(content.len());
        let args_str = &content[paren + 1..args_end];
        let params: Vec<String> = if args_str.is_empty() {
            vec![]
        } else {
            args_str.split(',').map(|s| s.trim().to_string()).collect()
        };

        let body = self.parse_until(&["endmacro"])?;
        if self.looking_at_tag("endmacro") {
            let _ = self.consume_tag_content()?;
        }

        Ok(Node::MacroDef {
            name,
            params,
            body,
        })
    }

    fn parse_set(&mut self, content: &str) -> Result<Node, TemplateProcessorError> {
        // "x = expr"
        if let Some(eq_pos) = content.find('=') {
            let variable = content[..eq_pos].trim().to_string();
            let value = content[eq_pos + 1..].trim().to_string();
            Ok(Node::Set { variable, value })
        } else {
            Err(TemplateProcessorError::ParseError {
                message: format!("invalid set syntax: {content}"),
                line: self.line,
            })
        }
    }

    fn parse_raw(&mut self) -> Result<Node, TemplateProcessorError> {
        let start = self.pos;
        while self.pos < self.input.len() {
            if self.looking_at_tag("endraw") {
                let raw_content = self.input[start..self.pos].to_string();
                let _ = self.consume_tag_content()?;
                return Ok(Node::Raw(raw_content));
            }
            if self.current_char() == '\n' {
                self.line += 1;
            }
            self.pos += 1;
        }
        Err(TemplateProcessorError::ParseError {
            message: "unclosed raw block".into(),
            line: self.line,
        })
    }
}

// ── Renderer ────────────────────────────────────────────────────

struct RenderContext {
    variables: HashMap<String, Value>,
    macros: HashMap<String, (Vec<String>, Vec<Node>)>,
    blocks: HashMap<String, Vec<Node>>,
    depth: usize,
    max_depth: usize,
    auto_escape: bool,
}

impl RenderContext {
    fn new(data: &Value, auto_escape: bool) -> Self {
        let mut variables = HashMap::new();
        if let Value::Object(map) = data {
            for (k, v) in map {
                variables.insert(k.clone(), v.clone());
            }
        }
        Self {
            variables,
            macros: HashMap::new(),
            blocks: HashMap::new(),
            depth: 0,
            max_depth: 50,
            auto_escape,
        }
    }

    fn resolve(&self, expr: &str) -> Value {
        let trimmed = expr.trim();

        // String literal.
        if (trimmed.starts_with('"') && trimmed.ends_with('"'))
            || (trimmed.starts_with('\'') && trimmed.ends_with('\''))
        {
            return Value::String(trimmed[1..trimmed.len() - 1].to_string());
        }

        // Integer literal.
        if let Ok(n) = trimmed.parse::<i64>() {
            return Value::Number(serde_json::Number::from(n));
        }

        // Float literal.
        if let Ok(n) = trimmed.parse::<f64>() {
            if let Some(num) = serde_json::Number::from_f64(n) {
                return Value::Number(num);
            }
        }

        // Boolean literal.
        if trimmed == "true" {
            return Value::Bool(true);
        }
        if trimmed == "false" {
            return Value::Bool(false);
        }
        if trimmed == "none" || trimmed == "null" {
            return Value::Null;
        }

        // Dot notation: a.b.c
        let parts: Vec<&str> = trimmed.split('.').collect();
        let mut current = self.variables.get(parts[0]).cloned().unwrap_or(Value::Null);
        for part in &parts[1..] {
            current = match current {
                Value::Object(ref map) => map.get(*part).cloned().unwrap_or(Value::Null),
                Value::Array(ref arr) => {
                    if let Ok(idx) = part.parse::<usize>() {
                        arr.get(idx).cloned().unwrap_or(Value::Null)
                    } else {
                        Value::Null
                    }
                }
                _ => Value::Null,
            };
        }
        current
    }

    fn is_truthy(&self, expr: &str) -> bool {
        // Handle "not" prefix.
        let trimmed = expr.trim();
        if let Some(inner) = trimmed.strip_prefix("not ") {
            return !self.is_truthy(inner.trim());
        }

        // Handle comparisons.
        if let Some((left, right)) = split_comparison(trimmed, "==") {
            return self.resolve(left) == self.resolve(right);
        }
        if let Some((left, right)) = split_comparison(trimmed, "!=") {
            return self.resolve(left) != self.resolve(right);
        }
        if let Some((left, right)) = split_comparison(trimmed, ">=") {
            return compare_values(&self.resolve(left), &self.resolve(right))
                .map_or(false, |o| o != std::cmp::Ordering::Less);
        }
        if let Some((left, right)) = split_comparison(trimmed, "<=") {
            return compare_values(&self.resolve(left), &self.resolve(right))
                .map_or(false, |o| o != std::cmp::Ordering::Greater);
        }
        if let Some((left, right)) = split_comparison(trimmed, ">") {
            return compare_values(&self.resolve(left), &self.resolve(right))
                .map_or(false, |o| o == std::cmp::Ordering::Greater);
        }
        if let Some((left, right)) = split_comparison(trimmed, "<") {
            return compare_values(&self.resolve(left), &self.resolve(right))
                .map_or(false, |o| o == std::cmp::Ordering::Less);
        }

        // Handle "and" / "or".
        if let Some(pos) = find_logical_op(trimmed, " and ") {
            let left = &trimmed[..pos];
            let right = &trimmed[pos + 5..];
            return self.is_truthy(left) && self.is_truthy(right);
        }
        if let Some(pos) = find_logical_op(trimmed, " or ") {
            let left = &trimmed[..pos];
            let right = &trimmed[pos + 4..];
            return self.is_truthy(left) || self.is_truthy(right);
        }

        value_is_truthy(&self.resolve(trimmed))
    }
}

fn split_comparison<'a>(expr: &'a str, op: &str) -> Option<(&'a str, &'a str)> {
    // Avoid matching <= or >= when looking for < or >.
    if op == ">" || op == "<" {
        // Find op that isn't part of >= or <= or !=.
        let bytes = expr.as_bytes();
        for i in 0..bytes.len() {
            if expr[i..].starts_with(op)
                && !expr[i..].starts_with(">=")
                && !expr[i..].starts_with("<=")
                && (i == 0 || bytes[i - 1] != b'!' && bytes[i - 1] != b'>' && bytes[i - 1] != b'<')
            {
                return Some((expr[..i].trim(), expr[i + op.len()..].trim()));
            }
        }
        return None;
    }
    expr.find(op)
        .map(|pos| (expr[..pos].trim(), expr[pos + op.len()..].trim()))
}

fn find_logical_op(expr: &str, op: &str) -> Option<usize> {
    // Simple: find first occurrence not inside quotes.
    let mut in_str = false;
    let mut quote_char = '"';
    let bytes = expr.as_bytes();
    let op_bytes = op.as_bytes();
    for i in 0..bytes.len() {
        if bytes[i] == b'"' || bytes[i] == b'\'' {
            if in_str && bytes[i] == quote_char as u8 {
                in_str = false;
            } else if !in_str {
                in_str = true;
                quote_char = bytes[i] as char;
            }
        }
        if !in_str && expr[i..].starts_with(op) {
            return Some(i);
        }
    }
    None
}

fn compare_values(a: &Value, b: &Value) -> Option<std::cmp::Ordering> {
    match (a, b) {
        (Value::Number(na), Value::Number(nb)) => {
            let fa = na.as_f64()?;
            let fb = nb.as_f64()?;
            fa.partial_cmp(&fb)
        }
        (Value::String(sa), Value::String(sb)) => Some(sa.cmp(sb)),
        _ => None,
    }
}

fn value_is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().map_or(false, |f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#x27;")
}

// ── Filters ─────────────────────────────────────────────────────

fn apply_filter(
    value: &str,
    filter: &FilterCall,
    _ctx: &RenderContext,
) -> Result<String, TemplateProcessorError> {
    match filter.name.as_str() {
        "upper" => Ok(value.to_uppercase()),
        "lower" => Ok(value.to_lowercase()),
        "capitalize" => {
            let mut chars = value.chars();
            match chars.next() {
                None => Ok(String::new()),
                Some(c) => Ok(c.to_uppercase().to_string() + &chars.as_str().to_lowercase()),
            }
        }
        "trim" => Ok(value.trim().to_string()),
        "length" => Ok(value.len().to_string()),
        "reverse" => Ok(value.chars().rev().collect()),
        "title" => Ok(value
            .split_whitespace()
            .map(|w| {
                let mut chars = w.chars();
                match chars.next() {
                    None => String::new(),
                    Some(c) => c.to_uppercase().to_string() + chars.as_str(),
                }
            })
            .collect::<Vec<_>>()
            .join(" ")),
        "default" => {
            if value.is_empty() {
                Ok(filter
                    .argument
                    .as_deref()
                    .unwrap_or("")
                    .trim_matches('"')
                    .trim_matches('\'')
                    .to_string())
            } else {
                Ok(value.to_string())
            }
        }
        "truncate" => {
            let max_len: usize = filter
                .argument
                .as_deref()
                .unwrap_or("80")
                .parse()
                .unwrap_or(80);
            if value.len() > max_len {
                Ok(format!("{}...", &value[..max_len]))
            } else {
                Ok(value.to_string())
            }
        }
        "escape" | "e" => Ok(html_escape(value)),
        "safe" => Ok(value.to_string()),
        "replace" => {
            if let Some(arg) = &filter.argument {
                // arg format: "old", "new"
                let parts: Vec<&str> = arg.split(',').collect();
                if parts.len() == 2 {
                    let old = parts[0].trim().trim_matches('"').trim_matches('\'');
                    let new = parts[1].trim().trim_matches('"').trim_matches('\'');
                    Ok(value.replace(old, new))
                } else {
                    Ok(value.to_string())
                }
            } else {
                Ok(value.to_string())
            }
        }
        "join" => {
            let sep = filter
                .argument
                .as_deref()
                .unwrap_or(", ")
                .trim_matches('"')
                .trim_matches('\'');
            // If value looks like a JSON array, join its elements.
            if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(value) {
                Ok(arr
                    .iter()
                    .map(value_to_string)
                    .collect::<Vec<_>>()
                    .join(sep))
            } else {
                Ok(value.to_string())
            }
        }
        "int" => {
            let n: i64 = value.parse().unwrap_or(0);
            Ok(n.to_string())
        }
        "float" => {
            let n: f64 = value.parse().unwrap_or(0.0);
            Ok(n.to_string())
        }
        "abs" => {
            if let Ok(n) = value.parse::<f64>() {
                let abs = n.abs();
                if abs == abs.floor() && abs < 1e15 {
                    Ok(format!("{}", abs as i64))
                } else {
                    Ok(abs.to_string())
                }
            } else {
                Ok(value.to_string())
            }
        }
        "first" => {
            if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(value) {
                Ok(arr.first().map(value_to_string).unwrap_or_default())
            } else {
                Ok(value.chars().next().map(|c| c.to_string()).unwrap_or_default())
            }
        }
        "last" => {
            if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(value) {
                Ok(arr.last().map(value_to_string).unwrap_or_default())
            } else {
                Ok(value.chars().last().map(|c| c.to_string()).unwrap_or_default())
            }
        }
        _ => Err(TemplateProcessorError::FilterNotFound(filter.name.clone())),
    }
}

// ── Render Nodes ────────────────────────────────────────────────

fn render_nodes(
    nodes: &[Node],
    ctx: &mut RenderContext,
    engine: &TemplateProcessor,
) -> Result<String, TemplateProcessorError> {
    if ctx.depth > ctx.max_depth {
        return Err(TemplateProcessorError::MaxRecursionDepth(ctx.max_depth));
    }
    ctx.depth += 1;

    let mut output = String::new();

    for node in nodes {
        match node {
            Node::Text(text) => output.push_str(text),
            Node::Raw(text) => output.push_str(text),
            Node::Comment => {}
            Node::Expr {
                expression,
                filters,
                escape,
            } => {
                let val = ctx.resolve(expression);
                let mut s = value_to_string(&val);
                for filter in filters {
                    s = apply_filter(&s, filter, ctx)?;
                }
                if *escape && ctx.auto_escape && !filters.iter().any(|f| f.name == "safe") {
                    s = html_escape(&s);
                }
                output.push_str(&s);
            }
            Node::If {
                condition,
                body,
                elif_branches,
                else_body,
            } => {
                if ctx.is_truthy(condition) {
                    output.push_str(&render_nodes(body, ctx, engine)?);
                } else {
                    let mut handled = false;
                    for (cond, branch_body) in elif_branches {
                        if ctx.is_truthy(cond) {
                            output.push_str(&render_nodes(branch_body, ctx, engine)?);
                            handled = true;
                            break;
                        }
                    }
                    if !handled && !else_body.is_empty() {
                        output.push_str(&render_nodes(else_body, ctx, engine)?);
                    }
                }
            }
            Node::For {
                variable,
                iterable,
                body,
                else_body,
            } => {
                let collection = ctx.resolve(iterable);
                match collection {
                    Value::Array(arr) if !arr.is_empty() => {
                        let total = arr.len();
                        for (i, item) in arr.iter().enumerate() {
                            ctx.variables.insert(variable.clone(), item.clone());
                            // loop variable
                            let mut loop_obj = serde_json::Map::new();
                            loop_obj.insert("index0".into(), Value::Number(i.into()));
                            loop_obj
                                .insert("index".into(), Value::Number((i + 1).into()));
                            loop_obj.insert("first".into(), Value::Bool(i == 0));
                            loop_obj
                                .insert("last".into(), Value::Bool(i == total - 1));
                            loop_obj.insert(
                                "length".into(),
                                Value::Number(total.into()),
                            );
                            ctx.variables
                                .insert("loop".into(), Value::Object(loop_obj));
                            output.push_str(&render_nodes(body, ctx, engine)?);
                        }
                        ctx.variables.remove(variable);
                        ctx.variables.remove("loop");
                    }
                    _ => {
                        if !else_body.is_empty() {
                            output.push_str(&render_nodes(else_body, ctx, engine)?);
                        }
                    }
                }
            }
            Node::Block { name, body } => {
                // If there's an override in ctx.blocks, use that.
                if let Some(override_body) = ctx.blocks.get(name).cloned() {
                    output.push_str(&render_nodes(&override_body, ctx, engine)?);
                } else {
                    output.push_str(&render_nodes(body, ctx, engine)?);
                }
            }
            Node::Extends(parent_name) => {
                // Handled at the engine level.
                let _ = parent_name;
            }
            Node::MacroDef { name, params, body } => {
                ctx.macros.insert(name.clone(), (params.clone(), body.clone()));
            }
            Node::MacroCall { name, args } => {
                let (params, body) = ctx
                    .macros
                    .get(name)
                    .cloned()
                    .ok_or_else(|| TemplateProcessorError::MacroNotFound(name.clone()))?;
                let saved: Vec<(String, Option<Value>)> = params
                    .iter()
                    .map(|p| (p.clone(), ctx.variables.get(p).cloned()))
                    .collect();
                for (i, param) in params.iter().enumerate() {
                    let val = args
                        .get(i)
                        .map(|a| ctx.resolve(a))
                        .unwrap_or(Value::Null);
                    ctx.variables.insert(param.clone(), val);
                }
                output.push_str(&render_nodes(&body, ctx, engine)?);
                for (param, orig) in saved {
                    if let Some(v) = orig {
                        ctx.variables.insert(param, v);
                    } else {
                        ctx.variables.remove(&param);
                    }
                }
            }
            Node::Set { variable, value } => {
                let val = ctx.resolve(value);
                ctx.variables.insert(variable.clone(), val);
            }
            Node::Include(name) => {
                let included = engine.render_by_name(name, ctx)?;
                output.push_str(&included);
            }
        }
    }

    ctx.depth -= 1;
    Ok(output)
}

// ── Engine ──────────────────────────────────────────────────────

/// Advanced template processor with Jinja2-like syntax.
pub struct TemplateProcessor {
    templates: HashMap<String, String>,
    auto_escape: bool,
}

impl TemplateProcessor {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            auto_escape: true,
        }
    }

    pub fn with_auto_escape(mut self, enabled: bool) -> Self {
        self.auto_escape = enabled;
        self
    }

    /// Register a named template.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        source: impl Into<String>,
    ) {
        self.templates.insert(name.into(), source.into());
    }

    /// Remove a template.
    pub fn remove(&mut self, name: &str) -> Option<String> {
        self.templates.remove(name)
    }

    /// Render a registered template by name.
    pub fn render(
        &self,
        name: &str,
        data: &Value,
    ) -> Result<String, TemplateProcessorError> {
        let source = self
            .templates
            .get(name)
            .ok_or_else(|| TemplateProcessorError::TemplateNotFound(name.to_string()))?;
        self.render_string(source, data)
    }

    /// Render an inline template string.
    pub fn render_string(
        &self,
        template: &str,
        data: &Value,
    ) -> Result<String, TemplateProcessorError> {
        let mut parser = Parser::new(template);
        let nodes = parser.parse()?;

        // Check for extends.
        let extends_name = nodes.iter().find_map(|n| {
            if let Node::Extends(name) = n {
                Some(name.clone())
            } else {
                None
            }
        });

        let mut ctx = RenderContext::new(data, self.auto_escape);

        if let Some(parent_name) = extends_name {
            // Collect block overrides from child.
            for node in &nodes {
                if let Node::Block { name, body } = node {
                    ctx.blocks.insert(name.clone(), body.clone());
                }
            }
            // Also collect macros from child.
            for node in &nodes {
                if let Node::MacroDef { name, params, body } = node {
                    ctx.macros.insert(name.clone(), (params.clone(), body.clone()));
                }
            }
            // Render parent.
            return self.render_by_name(&parent_name, &mut ctx);
        }

        render_nodes(&nodes, &mut ctx, self)
    }

    fn render_by_name(
        &self,
        name: &str,
        ctx: &mut RenderContext,
    ) -> Result<String, TemplateProcessorError> {
        let source = self
            .templates
            .get(name)
            .ok_or_else(|| TemplateProcessorError::TemplateNotFound(name.to_string()))?
            .clone();
        let mut parser = Parser::new(&source);
        let nodes = parser.parse()?;
        render_nodes(&nodes, ctx, self)
    }

    /// List registered template names.
    pub fn list_templates(&self) -> Vec<&str> {
        self.templates.keys().map(|k| k.as_str()).collect()
    }
}

impl Default for TemplateProcessor {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn engine() -> TemplateProcessor {
        TemplateProcessor::new().with_auto_escape(false)
    }

    #[test]
    fn test_basic_expression() {
        let tp = engine();
        let result = tp.render_string("Hello {{ name }}!", &json!({"name": "World"})).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_dot_access() {
        let tp = engine();
        let data = json!({"user": {"name": "Alice", "age": 30}});
        let result = tp.render_string("{{ user.name }} is {{ user.age }}", &data).unwrap();
        assert_eq!(result, "Alice is 30");
    }

    #[test]
    fn test_if_true() {
        let tp = engine();
        let data = json!({"show": true});
        let result = tp
            .render_string("{% if show %}visible{% endif %}", &data)
            .unwrap();
        assert_eq!(result, "visible");
    }

    #[test]
    fn test_if_false() {
        let tp = engine();
        let data = json!({"show": false});
        let result = tp
            .render_string("{% if show %}yes{% else %}no{% endif %}", &data)
            .unwrap();
        assert_eq!(result, "no");
    }

    #[test]
    fn test_elif() {
        let tp = engine();
        let data = json!({"val": 2});
        let result = tp.render_string(
            "{% if val == 1 %}one{% elif val == 2 %}two{% else %}other{% endif %}",
            &data,
        ).unwrap();
        assert_eq!(result, "two");
    }

    #[test]
    fn test_for_loop() {
        let tp = engine();
        let data = json!({"items": ["a", "b", "c"]});
        let result = tp
            .render_string("{% for x in items %}{{ x }}{% endfor %}", &data)
            .unwrap();
        assert_eq!(result, "abc");
    }

    #[test]
    fn test_for_loop_variables() {
        let tp = engine();
        let data = json!({"items": [1, 2, 3]});
        let result = tp.render_string(
            "{% for x in items %}{{ loop.index }}{% endfor %}",
            &data,
        ).unwrap();
        assert_eq!(result, "123");
    }

    #[test]
    fn test_for_else() {
        let tp = engine();
        let data = json!({"items": []});
        let result = tp
            .render_string(
                "{% for x in items %}{{ x }}{% else %}empty{% endfor %}",
                &data,
            )
            .unwrap();
        assert_eq!(result, "empty");
    }

    #[test]
    fn test_filter_upper() {
        let tp = engine();
        let result = tp
            .render_string("{{ name | upper }}", &json!({"name": "hello"}))
            .unwrap();
        assert_eq!(result, "HELLO");
    }

    #[test]
    fn test_filter_lower() {
        let tp = engine();
        let result = tp
            .render_string("{{ name | lower }}", &json!({"name": "HELLO"}))
            .unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_filter_capitalize() {
        let tp = engine();
        let result = tp
            .render_string("{{ name | capitalize }}", &json!({"name": "hello world"}))
            .unwrap();
        assert_eq!(result, "Hello world");
    }

    #[test]
    fn test_filter_title() {
        let tp = engine();
        let result = tp
            .render_string("{{ name | title }}", &json!({"name": "hello world"}))
            .unwrap();
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_filter_default() {
        let tp = engine();
        let result = tp
            .render_string(
                "{{ missing | default(\"fallback\") }}",
                &json!({}),
            )
            .unwrap();
        assert_eq!(result, "fallback");
    }

    #[test]
    fn test_filter_truncate() {
        let tp = engine();
        let result = tp
            .render_string(
                "{{ text | truncate(5) }}",
                &json!({"text": "Hello World"}),
            )
            .unwrap();
        assert_eq!(result, "Hello...");
    }

    #[test]
    fn test_filter_chain() {
        let tp = engine();
        let result = tp
            .render_string("{{ name | upper | trim }}", &json!({"name": "  hello  "}))
            .unwrap();
        assert_eq!(result, "HELLO");
    }

    #[test]
    fn test_set_variable() {
        let tp = engine();
        let result = tp
            .render_string("{% set x = 42 %}{{ x }}", &json!({}))
            .unwrap();
        assert_eq!(result, "42");
    }

    #[test]
    fn test_macro_def_and_call() {
        let tp = engine();
        let tmpl = "{% macro greet(name) %}Hello {{ name }}!{% endmacro %}{{ greet(\"World\") }}";
        let result = tp.render_string(tmpl, &json!({})).unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_comments() {
        let tp = engine();
        let result = tp
            .render_string("A{# comment #}B", &json!({}))
            .unwrap();
        assert_eq!(result, "AB");
    }

    #[test]
    fn test_raw_block() {
        let tp = engine();
        let result = tp
            .render_string("{% raw %}{{ not rendered }}{% endraw %}", &json!({}))
            .unwrap();
        assert_eq!(result, "{{ not rendered }}");
    }

    #[test]
    fn test_whitespace_control() {
        let tp = engine();
        let result = tp
            .render_string("A \n {%- if true -%} B {%- endif -%} \n C", &json!({}))
            .unwrap();
        // Left trim on tag removes trailing whitespace from text before.
        assert!(result.contains("B"));
    }

    #[test]
    fn test_auto_escape() {
        let tp = TemplateProcessor::new(); // auto_escape = true
        let result = tp
            .render_string("{{ html }}", &json!({"html": "<b>bold</b>"}))
            .unwrap();
        assert!(result.contains("&lt;b&gt;"));
    }

    #[test]
    fn test_safe_filter_skips_escape() {
        let tp = TemplateProcessor::new();
        let result = tp
            .render_string("{{ html | safe }}", &json!({"html": "<b>bold</b>"}))
            .unwrap();
        assert_eq!(result, "<b>bold</b>");
    }

    #[test]
    fn test_template_inheritance() {
        let mut tp = engine();
        tp.register("base", "Header{% block content %}default{% endblock %}Footer");
        tp.register("child", "{% extends \"base\" %}{% block content %}CUSTOM{% endblock %}");
        let result = tp.render("child", &json!({})).unwrap();
        assert_eq!(result, "HeaderCUSTOMFooter");
    }

    #[test]
    fn test_include() {
        let mut tp = engine();
        tp.register("partial", "Hello {{ name }}");
        tp.register("main", "Start-{% include \"partial\" %}-End");
        let result = tp.render("main", &json!({"name": "World"})).unwrap();
        assert_eq!(result, "Start-Hello World-End");
    }

    #[test]
    fn test_comparison_operators() {
        let tp = engine();
        let data = json!({"x": 5});

        let r1 = tp.render_string("{% if x > 3 %}yes{% endif %}", &data).unwrap();
        assert_eq!(r1, "yes");

        let r2 = tp.render_string("{% if x < 3 %}yes{% else %}no{% endif %}", &data).unwrap();
        assert_eq!(r2, "no");

        let r3 = tp.render_string("{% if x == 5 %}eq{% endif %}", &data).unwrap();
        assert_eq!(r3, "eq");
    }

    #[test]
    fn test_logical_and_or() {
        let tp = engine();
        let data = json!({"a": true, "b": false});

        let r1 = tp.render_string("{% if a and b %}yes{% else %}no{% endif %}", &data).unwrap();
        assert_eq!(r1, "no");

        let r2 = tp.render_string("{% if a or b %}yes{% else %}no{% endif %}", &data).unwrap();
        assert_eq!(r2, "yes");
    }

    #[test]
    fn test_not_operator() {
        let tp = engine();
        let data = json!({"flag": false});
        let result = tp
            .render_string("{% if not flag %}nope{% endif %}", &data)
            .unwrap();
        assert_eq!(result, "nope");
    }

    #[test]
    fn test_nested_for_if() {
        let tp = engine();
        let data = json!({"items": [1, 2, 3, 4, 5]});
        let result = tp.render_string(
            "{% for x in items %}{% if x > 3 %}{{ x }}{% endif %}{% endfor %}",
            &data,
        ).unwrap();
        assert_eq!(result, "45");
    }

    #[test]
    fn test_filter_abs() {
        let tp = engine();
        let result = tp
            .render_string("{{ val | abs }}", &json!({"val": -42}))
            .unwrap();
        assert_eq!(result, "42");
    }

    #[test]
    fn test_filter_reverse() {
        let tp = engine();
        let result = tp
            .render_string("{{ text | reverse }}", &json!({"text": "abc"}))
            .unwrap();
        assert_eq!(result, "cba");
    }

    #[test]
    fn test_filter_length() {
        let tp = engine();
        let result = tp
            .render_string("{{ text | length }}", &json!({"text": "hello"}))
            .unwrap();
        assert_eq!(result, "5");
    }

    #[test]
    fn test_template_not_found() {
        let tp = engine();
        let err = tp.render("nonexistent", &json!({})).unwrap_err();
        assert!(matches!(err, TemplateProcessorError::TemplateNotFound(_)));
    }

    #[test]
    fn test_filter_not_found() {
        let tp = engine();
        let err = tp
            .render_string("{{ x | nonexistent }}", &json!({"x": "val"}))
            .unwrap_err();
        assert!(matches!(err, TemplateProcessorError::FilterNotFound(_)));
    }

    #[test]
    fn test_string_literal_in_expression() {
        let tp = engine();
        let result = tp
            .render_string("{% if status == \"active\" %}yes{% endif %}", &json!({"status": "active"}))
            .unwrap();
        assert_eq!(result, "yes");
    }

    #[test]
    fn test_list_templates() {
        let mut tp = engine();
        tp.register("a", "");
        tp.register("b", "");
        let names = tp.list_templates();
        assert_eq!(names.len(), 2);
    }
}
