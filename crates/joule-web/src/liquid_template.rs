//! Liquid template engine — objects, tags, filters, for loops, assign/capture.
//!
//! Pure-Rust replacement for liquidjs and Shopify's Liquid. Supports objects
//! `{{ var }}`, tags `{% if %}{% endif %}`, filters `| upcase`, for loops
//! with `forloop` object, assign/capture, include, whitespace control,
//! and truthy/falsy semantics.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LiquidError {
    ParseError(String),
    RenderError(String),
    UnclosedTag { tag: String },
    UndefinedFilter { name: String },
    IncludeNotFound { name: String },
}

impl fmt::Display for LiquidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::RenderError(msg) => write!(f, "render error: {msg}"),
            Self::UnclosedTag { tag } => write!(f, "unclosed tag: {tag}"),
            Self::UndefinedFilter { name } => write!(f, "undefined filter: {name}"),
            Self::IncludeNotFound { name } => write!(f, "include not found: {name}"),
        }
    }
}

impl std::error::Error for LiquidError {}

// ── AST ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum LiquidNode {
    Text(String),
    /// `{{ expr | filter1 | filter2 }}` — output with filters.
    Output {
        expr: LiquidExpr,
        filters: Vec<FilterCall>,
    },
    /// `{% if expr %}...{% elsif expr %}...{% else %}...{% endif %}`
    If {
        branches: Vec<(LiquidExpr, Vec<LiquidNode>)>,
        else_body: Vec<LiquidNode>,
    },
    /// `{% unless expr %}...{% endunless %}`
    Unless {
        condition: LiquidExpr,
        body: Vec<LiquidNode>,
        else_body: Vec<LiquidNode>,
    },
    /// `{% for item in collection %}...{% endfor %}`
    For {
        var_name: String,
        collection: LiquidExpr,
        body: Vec<LiquidNode>,
        else_body: Vec<LiquidNode>,
        limit: Option<usize>,
        offset: Option<usize>,
        reversed: bool,
    },
    /// `{% assign var = expr %}`
    Assign {
        var_name: String,
        expr: LiquidExpr,
        filters: Vec<FilterCall>,
    },
    /// `{% capture var %}...{% endcapture %}`
    Capture {
        var_name: String,
        body: Vec<LiquidNode>,
    },
    /// `{% include 'partial' %}`
    Include { name: String },
    /// `{% case expr %}{% when val %}...{% endcase %}`
    Case {
        expr: LiquidExpr,
        whens: Vec<(Vec<LiquidExpr>, Vec<LiquidNode>)>,
        else_body: Vec<LiquidNode>,
    },
    /// `{% raw %}...{% endraw %}` — literal output.
    Raw(String),
    /// `{% comment %}...{% endcomment %}` — ignored.
    Comment,
    /// `{% cycle 'a', 'b', 'c' %}`
    Cycle { group: Option<String>, values: Vec<String> },
    /// `{% increment var %}` / `{% decrement var %}`
    Increment { var_name: String },
    Decrement { var_name: String },
    /// Break / Continue inside for loops.
    Break,
    Continue,
}

#[derive(Debug, Clone)]
enum LiquidExpr {
    Variable(Vec<String>),
    StringLiteral(String),
    NumberLiteral(f64),
    BoolLiteral(bool),
    Nil,
    /// `expr == expr`, `expr != expr`, etc.
    Comparison {
        left: Box<LiquidExpr>,
        op: CompOp,
        right: Box<LiquidExpr>,
    },
    /// `expr and expr`, `expr or expr`
    Logical {
        left: Box<LiquidExpr>,
        op: LogicOp,
        right: Box<LiquidExpr>,
    },
    /// `expr contains expr`
    Contains {
        left: Box<LiquidExpr>,
        right: Box<LiquidExpr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LogicOp {
    And,
    Or,
}

#[derive(Debug, Clone)]
struct FilterCall {
    name: String,
    args: Vec<LiquidExpr>,
}

// ── Parser ──────────────────────────────────────────────────────

fn parse_template(input: &str) -> Result<Vec<LiquidNode>, LiquidError> {
    let tokens = tokenize(input);
    parse_nodes(&tokens, &mut 0, &[])
}

#[derive(Debug, Clone)]
enum Token {
    Text(String),
    Output(String),
    Tag(String),
}

fn tokenize(input: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut rest = input;

    while !rest.is_empty() {
        // Look for the nearest {{ or {%.
        let output_pos = rest.find("{{");
        let tag_pos = rest.find("{%");

        let nearest = match (output_pos, tag_pos) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };

        match nearest {
            None => {
                tokens.push(Token::Text(rest.to_string()));
                break;
            }
            Some(pos) => {
                if pos > 0 {
                    tokens.push(Token::Text(rest[..pos].to_string()));
                }

                if rest[pos..].starts_with("{{") {
                    let start = pos + 2;
                    if let Some(end) = rest[start..].find("}}") {
                        let mut content = rest[start..start + end].to_string();
                        // Handle whitespace control.
                        let trim_left = content.starts_with('-');
                        let trim_right = content.ends_with('-');
                        if trim_left {
                            content = content[1..].to_string();
                            if let Some(Token::Text(t)) = tokens.last_mut() {
                                *t = t.trim_end().to_string();
                            }
                        }
                        if trim_right {
                            content = content[..content.len() - 1].to_string();
                        }
                        tokens.push(Token::Output(content.trim().to_string()));
                        rest = &rest[start + end + 2..];
                        if trim_right {
                            rest = rest.trim_start();
                        }
                    } else {
                        tokens.push(Token::Text(rest[pos..].to_string()));
                        break;
                    }
                } else {
                    // {%
                    let start = pos + 2;
                    if let Some(end) = rest[start..].find("%}") {
                        let mut content = rest[start..start + end].to_string();
                        let trim_left = content.starts_with('-');
                        let trim_right = content.ends_with('-');
                        if trim_left {
                            content = content[1..].to_string();
                            if let Some(Token::Text(t)) = tokens.last_mut() {
                                *t = t.trim_end().to_string();
                            }
                        }
                        if trim_right {
                            content = content[..content.len() - 1].to_string();
                        }
                        tokens.push(Token::Tag(content.trim().to_string()));
                        rest = &rest[start + end + 2..];
                        if trim_right {
                            rest = rest.trim_start();
                        }
                    } else {
                        tokens.push(Token::Text(rest[pos..].to_string()));
                        break;
                    }
                }
            }
        }
    }

    tokens
}

fn parse_nodes(
    tokens: &[Token],
    pos: &mut usize,
    end_tags: &[&str],
) -> Result<Vec<LiquidNode>, LiquidError> {
    let mut nodes = Vec::new();

    while *pos < tokens.len() {
        match &tokens[*pos] {
            Token::Text(t) => {
                nodes.push(LiquidNode::Text(t.clone()));
                *pos += 1;
            }
            Token::Output(content) => {
                let (expr, filters) = parse_output(content)?;
                nodes.push(LiquidNode::Output { expr, filters });
                *pos += 1;
            }
            Token::Tag(content) => {
                let parts: Vec<&str> = content.split_whitespace().collect();
                if parts.is_empty() {
                    *pos += 1;
                    continue;
                }

                let tag = parts[0];

                // Check if this is an end tag we're looking for.
                if end_tags.contains(&tag) {
                    return Ok(nodes);
                }

                match tag {
                    "if" => {
                        let expr = parse_expression(&parts[1..].join(" "))?;
                        *pos += 1;
                        let mut branches = vec![];
                        let body = parse_nodes(tokens, pos, &["elsif", "else", "endif"])?;
                        branches.push((expr, body));

                        while *pos < tokens.len() {
                            if let Token::Tag(c) = &tokens[*pos] {
                                let p: Vec<&str> = c.split_whitespace().collect();
                                if p[0] == "elsif" {
                                    let e = parse_expression(&p[1..].join(" "))?;
                                    *pos += 1;
                                    let b = parse_nodes(tokens, pos, &["elsif", "else", "endif"])?;
                                    branches.push((e, b));
                                } else if p[0] == "else" {
                                    *pos += 1;
                                    let else_body = parse_nodes(tokens, pos, &["endif"])?;
                                    *pos += 1; // skip endif
                                    nodes.push(LiquidNode::If {
                                        branches,
                                        else_body,
                                    });
                                    break;
                                } else if p[0] == "endif" {
                                    *pos += 1;
                                    nodes.push(LiquidNode::If {
                                        branches,
                                        else_body: Vec::new(),
                                    });
                                    break;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                    }
                    "unless" => {
                        let expr = parse_expression(&parts[1..].join(" "))?;
                        *pos += 1;
                        let body = parse_nodes(tokens, pos, &["else", "endunless"])?;
                        let mut else_body = Vec::new();
                        if *pos < tokens.len() {
                            if let Token::Tag(c) = &tokens[*pos] {
                                let p: Vec<&str> = c.split_whitespace().collect();
                                if p[0] == "else" {
                                    *pos += 1;
                                    else_body = parse_nodes(tokens, pos, &["endunless"])?;
                                }
                            }
                        }
                        *pos += 1; // skip endunless
                        nodes.push(LiquidNode::Unless {
                            condition: expr,
                            body,
                            else_body,
                        });
                    }
                    "for" => {
                        // {% for item in collection limit:N offset:N reversed %}
                        let var_name = parts.get(1).unwrap_or(&"item").to_string();
                        // parts[2] should be "in"
                        let col_str = parts.get(3).unwrap_or(&"");
                        let collection = parse_expression(col_str)?;

                        let mut limit = None;
                        let mut offset = None;
                        let mut reversed = false;
                        for part in &parts[4..] {
                            if part.starts_with("limit:") {
                                limit = part[6..].parse().ok();
                            } else if part.starts_with("offset:") {
                                offset = part[7..].parse().ok();
                            } else if *part == "reversed" {
                                reversed = true;
                            }
                        }

                        *pos += 1;
                        let body = parse_nodes(tokens, pos, &["else", "endfor"])?;
                        let mut else_body = Vec::new();
                        if *pos < tokens.len() {
                            if let Token::Tag(c) = &tokens[*pos] {
                                let p: Vec<&str> = c.split_whitespace().collect();
                                if p[0] == "else" {
                                    *pos += 1;
                                    else_body = parse_nodes(tokens, pos, &["endfor"])?;
                                }
                            }
                        }
                        *pos += 1; // skip endfor
                        nodes.push(LiquidNode::For {
                            var_name,
                            collection,
                            body,
                            else_body,
                            limit,
                            offset,
                            reversed,
                        });
                    }
                    "assign" => {
                        // {% assign var = expr | filter %}
                        let rest_str = parts[1..].join(" ");
                        if let Some(eq_pos) = rest_str.find('=') {
                            let var_name = rest_str[..eq_pos].trim().to_string();
                            let rhs = rest_str[eq_pos + 1..].trim();
                            let (expr, filters) = parse_output(rhs)?;
                            nodes.push(LiquidNode::Assign {
                                var_name,
                                expr,
                                filters,
                            });
                        }
                        *pos += 1;
                    }
                    "capture" => {
                        let var_name = parts.get(1).unwrap_or(&"").to_string();
                        *pos += 1;
                        let body = parse_nodes(tokens, pos, &["endcapture"])?;
                        *pos += 1; // skip endcapture
                        nodes.push(LiquidNode::Capture { var_name, body });
                    }
                    "include" => {
                        let name_raw = parts.get(1).unwrap_or(&"");
                        let name = name_raw.trim_matches('\'').trim_matches('"').to_string();
                        nodes.push(LiquidNode::Include { name });
                        *pos += 1;
                    }
                    "case" => {
                        let expr = parse_expression(&parts[1..].join(" "))?;
                        *pos += 1;
                        let mut whens = Vec::new();
                        let mut else_body = Vec::new();

                        loop {
                            if *pos >= tokens.len() {
                                break;
                            }
                            // Skip text nodes (whitespace) between when clauses.
                            if let Token::Text(_) = &tokens[*pos] {
                                *pos += 1;
                                continue;
                            }
                            if let Token::Tag(c) = &tokens[*pos] {
                                let p: Vec<&str> = c.split_whitespace().collect();
                                if p[0] == "when" {
                                    let vals_text = p[1..].join(" ");
                                    let vals: Vec<LiquidExpr> = vals_text
                                        .split(',')
                                        .map(|s| parse_expression(s.trim()))
                                        .collect::<Result<Vec<_>, _>>()?;
                                    *pos += 1;
                                    let body =
                                        parse_nodes(tokens, pos, &["when", "else", "endcase"])?;
                                    whens.push((vals, body));
                                } else if p[0] == "else" {
                                    *pos += 1;
                                    else_body = parse_nodes(tokens, pos, &["endcase"])?;
                                    *pos += 1; // skip endcase
                                    break;
                                } else if p[0] == "endcase" {
                                    *pos += 1;
                                    break;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        nodes.push(LiquidNode::Case {
                            expr,
                            whens,
                            else_body,
                        });
                    }
                    "raw" => {
                        *pos += 1;
                        let mut raw_text = String::new();
                        while *pos < tokens.len() {
                            match &tokens[*pos] {
                                Token::Tag(c) if c.trim() == "endraw" => {
                                    *pos += 1;
                                    break;
                                }
                                Token::Text(t) => raw_text.push_str(t),
                                Token::Output(o) => {
                                    raw_text.push_str("{{ ");
                                    raw_text.push_str(o);
                                    raw_text.push_str(" }}");
                                }
                                Token::Tag(t) => {
                                    raw_text.push_str("{% ");
                                    raw_text.push_str(t);
                                    raw_text.push_str(" %}");
                                }
                            }
                            *pos += 1;
                        }
                        nodes.push(LiquidNode::Raw(raw_text));
                    }
                    "comment" => {
                        *pos += 1;
                        // Skip until endcomment.
                        while *pos < tokens.len() {
                            if let Token::Tag(c) = &tokens[*pos] {
                                if c.trim() == "endcomment" {
                                    *pos += 1;
                                    break;
                                }
                            }
                            *pos += 1;
                        }
                        nodes.push(LiquidNode::Comment);
                    }
                    "cycle" => {
                        let rest_str = parts[1..].join(" ");
                        let values: Vec<String> = rest_str
                            .split(',')
                            .map(|s| {
                                s.trim()
                                    .trim_matches('\'')
                                    .trim_matches('"')
                                    .to_string()
                            })
                            .collect();
                        nodes.push(LiquidNode::Cycle {
                            group: None,
                            values,
                        });
                        *pos += 1;
                    }
                    "increment" => {
                        let var_name = parts.get(1).unwrap_or(&"").to_string();
                        nodes.push(LiquidNode::Increment { var_name });
                        *pos += 1;
                    }
                    "decrement" => {
                        let var_name = parts.get(1).unwrap_or(&"").to_string();
                        nodes.push(LiquidNode::Decrement { var_name });
                        *pos += 1;
                    }
                    "break" => {
                        nodes.push(LiquidNode::Break);
                        *pos += 1;
                    }
                    "continue" => {
                        nodes.push(LiquidNode::Continue);
                        *pos += 1;
                    }
                    _ => {
                        *pos += 1;
                    }
                }
            }
        }
    }

    Ok(nodes)
}

fn parse_output(input: &str) -> Result<(LiquidExpr, Vec<FilterCall>), LiquidError> {
    // Split on `|` but not inside strings.
    let segments = split_filters(input);
    if segments.is_empty() {
        return Ok((LiquidExpr::Nil, Vec::new()));
    }

    let expr = parse_expression(segments[0].trim())?;
    let mut filters = Vec::new();

    for segment in &segments[1..] {
        let trimmed = segment.trim();
        let (name, args) = parse_filter_call(trimmed)?;
        filters.push(FilterCall { name, args });
    }

    Ok((expr, filters))
}

fn split_filters(input: &str) -> Vec<String> {
    let mut segments = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut string_char = '"';

    for ch in input.chars() {
        if in_string {
            current.push(ch);
            if ch == string_char {
                in_string = false;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            in_string = true;
            string_char = ch;
            current.push(ch);
            continue;
        }
        if ch == '|' {
            segments.push(current.clone());
            current.clear();
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        segments.push(current);
    }
    segments
}

fn parse_filter_call(input: &str) -> Result<(String, Vec<LiquidExpr>), LiquidError> {
    let colon_pos = input.find(':');
    if let Some(cp) = colon_pos {
        let name = input[..cp].trim().to_string();
        let args_str = &input[cp + 1..];
        let arg_parts = split_filter_args(args_str);
        let args: Result<Vec<_>, _> = arg_parts
            .iter()
            .map(|a| parse_expression(a.trim()))
            .collect();
        Ok((name, args?))
    } else {
        Ok((input.trim().to_string(), Vec::new()))
    }
}

fn split_filter_args(input: &str) -> Vec<String> {
    let mut args = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut string_char = '"';

    for ch in input.chars() {
        if in_string {
            current.push(ch);
            if ch == string_char {
                in_string = false;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            in_string = true;
            string_char = ch;
            current.push(ch);
            continue;
        }
        if ch == ',' {
            args.push(current.clone());
            current.clear();
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        args.push(current);
    }
    args
}

fn parse_expression(input: &str) -> Result<LiquidExpr, LiquidError> {
    let s = input.trim();

    if s.is_empty() {
        return Ok(LiquidExpr::Nil);
    }

    // Check for logical operators (lowest precedence).
    // Split on " and " or " or " from left.
    if let Some(pos) = find_logic_op(s, " and ") {
        let left = parse_expression(&s[..pos])?;
        let right = parse_expression(&s[pos + 5..])?;
        return Ok(LiquidExpr::Logical {
            left: Box::new(left),
            op: LogicOp::And,
            right: Box::new(right),
        });
    }
    if let Some(pos) = find_logic_op(s, " or ") {
        let left = parse_expression(&s[..pos])?;
        let right = parse_expression(&s[pos + 4..])?;
        return Ok(LiquidExpr::Logical {
            left: Box::new(left),
            op: LogicOp::Or,
            right: Box::new(right),
        });
    }

    // Check for contains.
    if let Some(pos) = find_logic_op(s, " contains ") {
        let left = parse_expression(&s[..pos])?;
        let right = parse_expression(&s[pos + 10..])?;
        return Ok(LiquidExpr::Contains {
            left: Box::new(left),
            right: Box::new(right),
        });
    }

    // Check for comparison operators.
    for (op_str, op) in &[
        ("!=", CompOp::Ne),
        ("==", CompOp::Eq),
        ("<=", CompOp::Le),
        (">=", CompOp::Ge),
        ("<>", CompOp::Ne),
        ("<", CompOp::Lt),
        (">", CompOp::Gt),
    ] {
        if let Some(pos) = s.find(op_str) {
            let left = parse_expression(&s[..pos])?;
            let right = parse_expression(&s[pos + op_str.len()..])?;
            return Ok(LiquidExpr::Comparison {
                left: Box::new(left),
                op: *op,
                right: Box::new(right),
            });
        }
    }

    // Literals.
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return Ok(LiquidExpr::StringLiteral(s[1..s.len() - 1].to_string()));
    }
    if s == "true" {
        return Ok(LiquidExpr::BoolLiteral(true));
    }
    if s == "false" {
        return Ok(LiquidExpr::BoolLiteral(false));
    }
    if s == "nil" || s == "null" || s == "empty" {
        return Ok(LiquidExpr::Nil);
    }
    if let Ok(n) = s.parse::<f64>() {
        return Ok(LiquidExpr::NumberLiteral(n));
    }

    // Variable path.
    Ok(LiquidExpr::Variable(
        s.split('.').map(|p| p.to_string()).collect(),
    ))
}

fn find_logic_op(s: &str, op: &str) -> Option<usize> {
    let mut in_string = false;
    let mut string_char = '"';
    let bytes = s.as_bytes();
    let op_bytes = op.as_bytes();

    if s.len() < op.len() {
        return None;
    }

    for i in 0..=s.len() - op.len() {
        let ch = bytes[i] as char;
        if in_string {
            if ch == string_char {
                in_string = false;
            }
            continue;
        }
        if ch == '\'' || ch == '"' {
            in_string = true;
            string_char = ch;
            continue;
        }
        if &bytes[i..i + op.len()] == op_bytes {
            return Some(i);
        }
    }
    None
}

// ── Renderer ────────────────────────────────────────────────────

/// The Liquid template engine.
pub struct LiquidEngine {
    includes: HashMap<String, String>,
    custom_filters: HashMap<String, Box<dyn Fn(&Value, &[Value]) -> Value>>,
}

impl LiquidEngine {
    pub fn new() -> Self {
        Self {
            includes: HashMap::new(),
            custom_filters: HashMap::new(),
        }
    }

    /// Register an include template.
    pub fn register_include(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.includes.insert(name.into(), template.into());
    }

    /// Register a custom filter.
    pub fn register_filter(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&Value, &[Value]) -> Value + 'static,
    ) {
        self.custom_filters.insert(name.into(), Box::new(f));
    }

    /// Render a template string with the given data.
    pub fn render(&self, template: &str, data: &Value) -> Result<String, LiquidError> {
        let nodes = parse_template(template)?;
        let mut vars = match data {
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect::<HashMap<String, Value>>(),
            _ => HashMap::new(),
        };
        let mut counters: HashMap<String, i64> = HashMap::new();
        let mut cycle_counters: HashMap<String, usize> = HashMap::new();
        self.render_nodes(&nodes, &mut vars, &mut counters, &mut cycle_counters)
    }

    fn render_nodes(
        &self,
        nodes: &[LiquidNode],
        vars: &mut HashMap<String, Value>,
        counters: &mut HashMap<String, i64>,
        cycle_counters: &mut HashMap<String, usize>,
    ) -> Result<String, LiquidError> {
        let mut out = String::new();

        for node in nodes {
            match node {
                LiquidNode::Text(t) => out.push_str(t),
                LiquidNode::Output { expr, filters } => {
                    let mut val = self.resolve_expr(expr, vars);
                    for filter in filters {
                        val = self.apply_filter(&filter.name, &val, &filter.args, vars)?;
                    }
                    out.push_str(&value_to_output(&val));
                }
                LiquidNode::If {
                    branches,
                    else_body,
                } => {
                    let mut rendered = false;
                    for (expr, body) in branches {
                        let val = self.resolve_expr(expr, vars);
                        if is_truthy_liquid(&val) {
                            let s = self.render_nodes(body, vars, counters, cycle_counters)?;
                            out.push_str(&s);
                            rendered = true;
                            break;
                        }
                    }
                    if !rendered {
                        let s = self.render_nodes(else_body, vars, counters, cycle_counters)?;
                        out.push_str(&s);
                    }
                }
                LiquidNode::Unless {
                    condition,
                    body,
                    else_body,
                } => {
                    let val = self.resolve_expr(condition, vars);
                    if !is_truthy_liquid(&val) {
                        let s = self.render_nodes(body, vars, counters, cycle_counters)?;
                        out.push_str(&s);
                    } else {
                        let s = self.render_nodes(else_body, vars, counters, cycle_counters)?;
                        out.push_str(&s);
                    }
                }
                LiquidNode::For {
                    var_name,
                    collection,
                    body,
                    else_body,
                    limit,
                    offset,
                    reversed,
                } => {
                    let col_val = self.resolve_expr(collection, vars);
                    let items: Vec<Value> = match &col_val {
                        Value::Array(arr) => arr.clone(),
                        _ => Vec::new(),
                    };

                    if items.is_empty() {
                        let s = self.render_nodes(else_body, vars, counters, cycle_counters)?;
                        out.push_str(&s);
                        continue;
                    }

                    let start = offset.unwrap_or(0);
                    let end = limit.map_or(items.len(), |l| (start + l).min(items.len()));
                    let mut slice: Vec<Value> = if start < items.len() {
                        items[start..end].to_vec()
                    } else {
                        Vec::new()
                    };

                    if *reversed {
                        slice.reverse();
                    }

                    let len = slice.len();
                    let old_val = vars.remove(var_name.as_str());
                    let old_forloop = vars.remove("forloop");

                    for (i, item) in slice.iter().enumerate() {
                        vars.insert(var_name.clone(), item.clone());

                        let forloop = serde_json::json!({
                            "index": i + 1,
                            "index0": i,
                            "rindex": len - i,
                            "rindex0": len - i - 1,
                            "first": i == 0,
                            "last": i == len - 1,
                            "length": len
                        });
                        vars.insert("forloop".to_string(), forloop);

                        let s = self.render_nodes(body, vars, counters, cycle_counters)?;
                        out.push_str(&s);
                    }

                    vars.remove(var_name.as_str());
                    vars.remove("forloop");
                    if let Some(v) = old_val {
                        vars.insert(var_name.clone(), v);
                    }
                    if let Some(v) = old_forloop {
                        vars.insert("forloop".to_string(), v);
                    }
                }
                LiquidNode::Assign {
                    var_name,
                    expr,
                    filters,
                } => {
                    let mut val = self.resolve_expr(expr, vars);
                    for filter in filters {
                        val = self.apply_filter(&filter.name, &val, &filter.args, vars)?;
                    }
                    vars.insert(var_name.clone(), val);
                }
                LiquidNode::Capture { var_name, body } => {
                    let s = self.render_nodes(body, vars, counters, cycle_counters)?;
                    vars.insert(var_name.clone(), Value::String(s));
                }
                LiquidNode::Include { name } => {
                    let src = self
                        .includes
                        .get(name.as_str())
                        .ok_or_else(|| LiquidError::IncludeNotFound {
                            name: name.clone(),
                        })?
                        .clone();
                    let inc_nodes = parse_template(&src)?;
                    let s = self.render_nodes(&inc_nodes, vars, counters, cycle_counters)?;
                    out.push_str(&s);
                }
                LiquidNode::Case {
                    expr,
                    whens,
                    else_body,
                } => {
                    let val = self.resolve_expr(expr, vars);
                    let mut matched = false;
                    for (when_vals, body) in whens {
                        for wv in when_vals {
                            let resolved = self.resolve_expr(wv, vars);
                            if values_equal(&val, &resolved) {
                                let s = self.render_nodes(body, vars, counters, cycle_counters)?;
                                out.push_str(&s);
                                matched = true;
                                break;
                            }
                        }
                        if matched {
                            break;
                        }
                    }
                    if !matched {
                        let s = self.render_nodes(else_body, vars, counters, cycle_counters)?;
                        out.push_str(&s);
                    }
                }
                LiquidNode::Raw(text) => {
                    out.push_str(text);
                }
                LiquidNode::Comment => {}
                LiquidNode::Cycle { group: _, values } => {
                    let key = values.join(",");
                    let count = cycle_counters.entry(key).or_insert(0);
                    let idx = *count % values.len();
                    out.push_str(&values[idx]);
                    *count += 1;
                }
                LiquidNode::Increment { var_name } => {
                    let val = counters.entry(var_name.clone()).or_insert(0);
                    out.push_str(&val.to_string());
                    *val += 1;
                }
                LiquidNode::Decrement { var_name } => {
                    let val = counters.entry(var_name.clone()).or_insert(0);
                    *val -= 1;
                    out.push_str(&val.to_string());
                }
                LiquidNode::Break | LiquidNode::Continue => {
                    // These are handled by the for loop at a higher level;
                    // simplified here.
                }
            }
        }

        Ok(out)
    }

    fn resolve_expr(&self, expr: &LiquidExpr, vars: &HashMap<String, Value>) -> Value {
        match expr {
            LiquidExpr::Variable(parts) => {
                if parts.is_empty() {
                    return Value::Null;
                }
                let mut current = vars.get(&parts[0]).cloned().unwrap_or(Value::Null);
                for part in &parts[1..] {
                    current = current.get(part).cloned().unwrap_or(Value::Null);
                }
                current
            }
            LiquidExpr::StringLiteral(s) => Value::String(s.clone()),
            LiquidExpr::NumberLiteral(n) => {
                // Preserve integer representation for whole numbers so that
                // serde_json::Value::as_u64 / as_i64 works on filter args
                // and display omits the trailing ".0".
                let v = *n;
                if v.fract() == 0.0 {
                    if v >= 0.0 && v <= u64::MAX as f64 {
                        Value::Number(serde_json::Number::from(v as u64))
                    } else if v >= i64::MIN as f64 && v <= i64::MAX as f64 {
                        Value::Number(serde_json::Number::from(v as i64))
                    } else {
                        serde_json::Number::from_f64(v)
                            .map_or(Value::Null, Value::Number)
                    }
                } else {
                    serde_json::Number::from_f64(v)
                        .map_or(Value::Null, Value::Number)
                }
            }
            LiquidExpr::BoolLiteral(b) => Value::Bool(*b),
            LiquidExpr::Nil => Value::Null,
            LiquidExpr::Comparison { left, op, right } => {
                let l = self.resolve_expr(left, vars);
                let r = self.resolve_expr(right, vars);
                Value::Bool(compare_values(&l, *op, &r))
            }
            LiquidExpr::Logical { left, op, right } => {
                let l = self.resolve_expr(left, vars);
                let r = self.resolve_expr(right, vars);
                match op {
                    LogicOp::And => Value::Bool(is_truthy_liquid(&l) && is_truthy_liquid(&r)),
                    LogicOp::Or => Value::Bool(is_truthy_liquid(&l) || is_truthy_liquid(&r)),
                }
            }
            LiquidExpr::Contains { left, right } => {
                let l = self.resolve_expr(left, vars);
                let r = self.resolve_expr(right, vars);
                let result = match (&l, &r) {
                    (Value::String(s), Value::String(sub)) => s.contains(sub.as_str()),
                    (Value::Array(arr), _) => arr.contains(&r),
                    _ => false,
                };
                Value::Bool(result)
            }
        }
    }

    fn apply_filter(
        &self,
        name: &str,
        val: &Value,
        args: &[LiquidExpr],
        vars: &HashMap<String, Value>,
    ) -> Result<Value, LiquidError> {
        let resolved_args: Vec<Value> = args.iter().map(|a| self.resolve_expr(a, vars)).collect();

        // Check custom filters first.
        if let Some(f) = self.custom_filters.get(name) {
            return Ok(f(val, &resolved_args));
        }

        match name {
            "upcase" => Ok(Value::String(value_to_output(val).to_uppercase())),
            "downcase" => Ok(Value::String(value_to_output(val).to_lowercase())),
            "capitalize" => {
                let s = value_to_output(val);
                let cap = capitalize_first(&s);
                Ok(Value::String(cap))
            }
            "strip" | "trim" => Ok(Value::String(value_to_output(val).trim().to_string())),
            "lstrip" => Ok(Value::String(value_to_output(val).trim_start().to_string())),
            "rstrip" => Ok(Value::String(value_to_output(val).trim_end().to_string())),
            "size" => match val {
                Value::String(s) => Ok(Value::Number(s.len().into())),
                Value::Array(a) => Ok(Value::Number(a.len().into())),
                _ => Ok(Value::Number(0.into())),
            },
            "reverse" => match val {
                Value::Array(a) => {
                    let mut reversed = a.clone();
                    reversed.reverse();
                    Ok(Value::Array(reversed))
                }
                Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
                _ => Ok(val.clone()),
            },
            "first" => match val {
                Value::Array(a) => Ok(a.first().cloned().unwrap_or(Value::Null)),
                _ => Ok(Value::Null),
            },
            "last" => match val {
                Value::Array(a) => Ok(a.last().cloned().unwrap_or(Value::Null)),
                _ => Ok(Value::Null),
            },
            "join" => {
                let sep = resolved_args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or(", ");
                match val {
                    Value::Array(a) => {
                        let joined: Vec<String> = a.iter().map(|v| value_to_output(v)).collect();
                        Ok(Value::String(joined.join(sep)))
                    }
                    _ => Ok(val.clone()),
                }
            }
            "split" => {
                let sep = resolved_args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or(",");
                let s = value_to_output(val);
                let parts: Vec<Value> = s.split(sep).map(|p| Value::String(p.to_string())).collect();
                Ok(Value::Array(parts))
            }
            "append" => {
                let suffix = resolved_args
                    .first()
                    .map(|v| value_to_output(v))
                    .unwrap_or_default();
                let mut s = value_to_output(val);
                s.push_str(&suffix);
                Ok(Value::String(s))
            }
            "prepend" => {
                let prefix = resolved_args
                    .first()
                    .map(|v| value_to_output(v))
                    .unwrap_or_default();
                let s = value_to_output(val);
                Ok(Value::String(format!("{prefix}{s}")))
            }
            "replace" => {
                let from = resolved_args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let to = resolved_args.get(1).and_then(|v| v.as_str()).unwrap_or("");
                let s = value_to_output(val);
                Ok(Value::String(s.replace(from, to)))
            }
            "replace_first" => {
                let from = resolved_args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let to = resolved_args.get(1).and_then(|v| v.as_str()).unwrap_or("");
                let s = value_to_output(val);
                Ok(Value::String(s.replacen(from, to, 1)))
            }
            "remove" => {
                let target = resolved_args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let s = value_to_output(val);
                Ok(Value::String(s.replace(target, "")))
            }
            "truncate" => {
                let len = resolved_args
                    .first()
                    .and_then(|v| v.as_u64())
                    .unwrap_or(50) as usize;
                let ellipsis = resolved_args
                    .get(1)
                    .and_then(|v| v.as_str())
                    .unwrap_or("...");
                let s = value_to_output(val);
                if s.len() <= len {
                    Ok(Value::String(s))
                } else {
                    let truncated_len = len.saturating_sub(ellipsis.len());
                    let mut result = s[..truncated_len].to_string();
                    result.push_str(ellipsis);
                    Ok(Value::String(result))
                }
            }
            "truncatewords" => {
                let count = resolved_args
                    .first()
                    .and_then(|v| v.as_u64())
                    .unwrap_or(15) as usize;
                let s = value_to_output(val);
                let words: Vec<&str> = s.split_whitespace().collect();
                if words.len() <= count {
                    Ok(Value::String(s))
                } else {
                    let mut result = words[..count].join(" ");
                    result.push_str("...");
                    Ok(Value::String(result))
                }
            }
            "strip_html" => {
                let s = value_to_output(val);
                Ok(Value::String(strip_html_tags(&s)))
            }
            "escape" | "escape_once" => {
                let s = value_to_output(val);
                Ok(Value::String(html_escape_liquid(&s)))
            }
            "newline_to_br" => {
                let s = value_to_output(val);
                Ok(Value::String(s.replace('\n', "<br />\n")))
            }
            "strip_newlines" => {
                let s = value_to_output(val);
                Ok(Value::String(s.replace('\n', "").replace('\r', "")))
            }
            "abs" => {
                if let Some(n) = val.as_f64() {
                    let a = n.abs();
                    Ok(Value::Number(f64_to_number(a)))
                } else {
                    Ok(val.clone())
                }
            }
            "plus" => {
                let a = val.as_f64().unwrap_or(0.0);
                let b = resolved_args.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
                Ok(serde_json::Number::from_f64(a + b).map_or(Value::Null, Value::Number))
            }
            "minus" => {
                let a = val.as_f64().unwrap_or(0.0);
                let b = resolved_args.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
                Ok(serde_json::Number::from_f64(a - b).map_or(Value::Null, Value::Number))
            }
            "times" => {
                let a = val.as_f64().unwrap_or(0.0);
                let b = resolved_args.first().and_then(|v| v.as_f64()).unwrap_or(0.0);
                Ok(serde_json::Number::from_f64(a * b).map_or(Value::Null, Value::Number))
            }
            "divided_by" => {
                let a = val.as_f64().unwrap_or(0.0);
                let b = resolved_args.first().and_then(|v| v.as_f64()).unwrap_or(1.0);
                if b == 0.0 {
                    Ok(Value::Number(0.into()))
                } else {
                    Ok(serde_json::Number::from_f64((a / b).floor())
                        .map_or(Value::Null, Value::Number))
                }
            }
            "modulo" => {
                let a = val.as_f64().unwrap_or(0.0);
                let b = resolved_args.first().and_then(|v| v.as_f64()).unwrap_or(1.0);
                if b == 0.0 {
                    Ok(Value::Number(0.into()))
                } else {
                    Ok(serde_json::Number::from_f64(a % b).map_or(Value::Null, Value::Number))
                }
            }
            "ceil" => {
                let n = val.as_f64().unwrap_or(0.0);
                Ok(serde_json::Number::from_f64(n.ceil()).map_or(Value::Null, Value::Number))
            }
            "floor" => {
                let n = val.as_f64().unwrap_or(0.0);
                Ok(serde_json::Number::from_f64(n.floor()).map_or(Value::Null, Value::Number))
            }
            "round" => {
                let n = val.as_f64().unwrap_or(0.0);
                let precision = resolved_args
                    .first()
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as i32;
                let factor = 10f64.powi(precision);
                let rounded = (n * factor).round() / factor;
                Ok(serde_json::Number::from_f64(rounded).map_or(Value::Null, Value::Number))
            }
            "default" => {
                if is_truthy_liquid(val) || (val.is_string() && !val.as_str().unwrap_or("").is_empty()) {
                    Ok(val.clone())
                } else {
                    Ok(resolved_args.first().cloned().unwrap_or(Value::Null))
                }
            }
            "sort" => match val {
                Value::Array(a) => {
                    let mut sorted = a.clone();
                    sorted.sort_by(|a, b| {
                        let sa = value_to_output(a);
                        let sb = value_to_output(b);
                        sa.cmp(&sb)
                    });
                    Ok(Value::Array(sorted))
                }
                _ => Ok(val.clone()),
            },
            "uniq" | "unique" => match val {
                Value::Array(a) => {
                    let mut seen = Vec::new();
                    let mut unique = Vec::new();
                    for item in a {
                        let s = value_to_output(item);
                        if !seen.contains(&s) {
                            seen.push(s);
                            unique.push(item.clone());
                        }
                    }
                    Ok(Value::Array(unique))
                }
                _ => Ok(val.clone()),
            },
            "compact" => match val {
                Value::Array(a) => {
                    let compacted: Vec<Value> =
                        a.iter().filter(|v| !v.is_null()).cloned().collect();
                    Ok(Value::Array(compacted))
                }
                _ => Ok(val.clone()),
            },
            "map" => match val {
                Value::Array(a) => {
                    let key = resolved_args
                        .first()
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    let mapped: Vec<Value> = a
                        .iter()
                        .map(|item| item.get(key).cloned().unwrap_or(Value::Null))
                        .collect();
                    Ok(Value::Array(mapped))
                }
                _ => Ok(val.clone()),
            },
            "concat" => match val {
                Value::Array(a) => {
                    let mut result = a.clone();
                    if let Some(Value::Array(b)) = resolved_args.first() {
                        result.extend(b.iter().cloned());
                    }
                    Ok(Value::Array(result))
                }
                _ => Ok(val.clone()),
            },
            "slice" => {
                let s = value_to_output(val);
                let start = resolved_args
                    .first()
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let len = resolved_args
                    .get(1)
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;

                let chars: Vec<char> = s.chars().collect();
                let total = chars.len() as i64;
                let actual_start = if start < 0 {
                    (total + start).max(0) as usize
                } else {
                    start as usize
                };
                let end = (actual_start + len).min(chars.len());
                if actual_start < chars.len() {
                    Ok(Value::String(chars[actual_start..end].iter().collect()))
                } else {
                    Ok(Value::String(String::new()))
                }
            }
            _ => {
                // Unknown filter — pass through.
                Ok(val.clone())
            }
        }
    }
}

impl Default for LiquidEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ─────────────────────────────────────────────────────

/// Convert an f64 to a serde_json::Number, using integer representation when
/// the value is a whole number so that `.as_u64()` / `.as_i64()` work and
/// display omits the trailing ".0".
fn f64_to_number(v: f64) -> serde_json::Number {
    if v.fract() == 0.0 {
        if v >= 0.0 && v <= u64::MAX as f64 {
            serde_json::Number::from(v as u64)
        } else if v >= i64::MIN as f64 && v <= i64::MAX as f64 {
            serde_json::Number::from(v as i64)
        } else {
            serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0))
        }
    } else {
        serde_json::Number::from_f64(v).unwrap_or_else(|| serde_json::Number::from(0))
    }
}

fn value_to_output(val: &Value) -> String {
    match val {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => {
            // If the number is a whole float (e.g. 10.0), display as integer.
            if let Some(f) = n.as_f64() {
                if f.fract() == 0.0 && n.is_f64() {
                    return format!("{}", f as i64);
                }
            }
            n.to_string()
        }
        Value::String(s) => s.clone(),
        _ => serde_json::to_string(val).unwrap_or_default(),
    }
}

fn is_truthy_liquid(val: &Value) -> bool {
    // In Liquid, only nil and false are falsy; empty string is truthy,
    // 0 is truthy, empty array is truthy.
    match val {
        Value::Null => false,
        Value::Bool(b) => *b,
        _ => true,
    }
}

fn compare_values(left: &Value, op: CompOp, right: &Value) -> bool {
    match op {
        CompOp::Eq => values_equal(left, right),
        CompOp::Ne => !values_equal(left, right),
        CompOp::Lt => compare_numeric(left, right).map_or(false, |c| c < 0),
        CompOp::Le => compare_numeric(left, right).map_or(false, |c| c <= 0),
        CompOp::Gt => compare_numeric(left, right).map_or(false, |c| c > 0),
        CompOp::Ge => compare_numeric(left, right).map_or(false, |c| c >= 0),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(sa), Value::String(sb)) => sa == sb,
        (Value::Number(na), Value::Number(nb)) => {
            na.as_f64().unwrap_or(0.0) == nb.as_f64().unwrap_or(0.0)
        }
        (Value::Bool(ba), Value::Bool(bb)) => ba == bb,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

fn compare_numeric(a: &Value, b: &Value) -> Option<i8> {
    let an = a.as_f64()?;
    let bn = b.as_f64()?;
    if an < bn {
        Some(-1)
    } else if an > bn {
        Some(1)
    } else {
        Some(0)
    }
}

fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => {
            let upper: String = first.to_uppercase().collect();
            format!("{upper}{}", chars.as_str())
        }
    }
}

fn strip_html_tags(s: &str) -> String {
    let mut result = String::new();
    let mut in_tag = false;
    for ch in s.chars() {
        if ch == '<' {
            in_tag = true;
        } else if ch == '>' {
            in_tag = false;
        } else if !in_tag {
            result.push(ch);
        }
    }
    result
}

fn html_escape_liquid(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            _ => out.push(ch),
        }
    }
    out
}

// ── Convenience ─────────────────────────────────────────────────

/// Render a template with data using a default engine.
pub fn render(template: &str, data: &Value) -> Result<String, LiquidError> {
    let engine = LiquidEngine::new();
    engine.render(template, data)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_output() {
        let result = render("Hello, {{ name }}!", &json!({"name": "World"})).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_filter_upcase() {
        let result = render("{{ name | upcase }}", &json!({"name": "alice"})).unwrap();
        assert_eq!(result, "ALICE");
    }

    #[test]
    fn test_filter_downcase() {
        let result = render("{{ name | downcase }}", &json!({"name": "HELLO"})).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_filter_append() {
        let result =
            render("{{ name | append: ' World' }}", &json!({"name": "Hello"})).unwrap();
        assert_eq!(result, "Hello World");
    }

    #[test]
    fn test_filter_capitalize() {
        let result = render("{{ s | capitalize }}", &json!({"s": "hello"})).unwrap();
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_if_true() {
        let tpl = "{% if show %}yes{% endif %}";
        let result = render(tpl, &json!({"show": true})).unwrap();
        assert_eq!(result, "yes");
    }

    #[test]
    fn test_if_false() {
        let tpl = "{% if show %}yes{% endif %}";
        let result = render(tpl, &json!({"show": false})).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_if_else() {
        let tpl = "{% if show %}yes{% else %}no{% endif %}";
        let result = render(tpl, &json!({"show": false})).unwrap();
        assert_eq!(result, "no");
    }

    #[test]
    fn test_if_elsif() {
        let tpl = "{% if x == 1 %}one{% elsif x == 2 %}two{% else %}other{% endif %}";
        let result = render(tpl, &json!({"x": 2})).unwrap();
        assert_eq!(result, "two");
    }

    #[test]
    fn test_unless() {
        let tpl = "{% unless hidden %}visible{% endunless %}";
        let result = render(tpl, &json!({"hidden": false})).unwrap();
        assert_eq!(result, "visible");
    }

    #[test]
    fn test_for_loop() {
        let tpl = "{% for item in items %}{{ item }} {% endfor %}";
        let result = render(tpl, &json!({"items": ["a", "b", "c"]})).unwrap();
        assert_eq!(result, "a b c ");
    }

    #[test]
    fn test_for_loop_with_forloop_index() {
        let tpl = "{% for item in items %}{{ forloop.index }}{% endfor %}";
        let result = render(tpl, &json!({"items": ["a", "b", "c"]})).unwrap();
        assert_eq!(result, "123");
    }

    #[test]
    fn test_for_loop_first_last() {
        let tpl = "{% for item in items %}{% if forloop.first %}[{% endif %}{{ item }}{% if forloop.last %}]{% endif %}{% endfor %}";
        let result = render(tpl, &json!({"items": ["a", "b", "c"]})).unwrap();
        assert_eq!(result, "[abc]");
    }

    #[test]
    fn test_for_empty() {
        let tpl = "{% for item in items %}{{ item }}{% else %}empty{% endfor %}";
        let result = render(tpl, &json!({"items": []})).unwrap();
        assert_eq!(result, "empty");
    }

    #[test]
    fn test_assign() {
        let tpl = "{% assign greeting = 'hello' %}{{ greeting }}";
        let result = render(tpl, &json!({})).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_capture() {
        let tpl = "{% capture full_name %}{{ first }} {{ last }}{% endcapture %}{{ full_name }}";
        let result = render(tpl, &json!({"first": "John", "last": "Doe"})).unwrap();
        assert_eq!(result, "John Doe");
    }

    #[test]
    fn test_include() {
        let mut engine = LiquidEngine::new();
        engine.register_include("greeting", "Hello, {{ name }}!");
        let result = engine
            .render(
                "{% include 'greeting' %}",
                &json!({"name": "World"}),
            )
            .unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_whitespace_control() {
        let tpl = "A {%- if true -%} B {%- endif -%} C";
        let result = render(tpl, &json!({})).unwrap();
        assert_eq!(result, "ABC");
    }

    #[test]
    fn test_filter_size() {
        let result = render("{{ items | size }}", &json!({"items": [1, 2, 3]})).unwrap();
        assert_eq!(result, "3");
    }

    #[test]
    fn test_filter_join() {
        let result =
            render("{{ items | join: ', ' }}", &json!({"items": ["a", "b"]})).unwrap();
        assert_eq!(result, "a, b");
    }

    #[test]
    fn test_filter_split() {
        let result =
            render("{{ s | split: ',' | size }}", &json!({"s": "a,b,c"})).unwrap();
        assert_eq!(result, "3");
    }

    #[test]
    fn test_filter_replace() {
        let result = render(
            "{{ s | replace: 'world', 'earth' }}",
            &json!({"s": "hello world"}),
        )
        .unwrap();
        assert_eq!(result, "hello earth");
    }

    #[test]
    fn test_nested_dot_path() {
        let result =
            render("{{ a.b.c }}", &json!({"a": {"b": {"c": "deep"}}})).unwrap();
        assert_eq!(result, "deep");
    }

    #[test]
    fn test_comparison_operators() {
        let tpl = "{% if x > 5 %}big{% else %}small{% endif %}";
        let r1 = render(tpl, &json!({"x": 10})).unwrap();
        assert_eq!(r1, "big");
        let r2 = render(tpl, &json!({"x": 3})).unwrap();
        assert_eq!(r2, "small");
    }

    #[test]
    fn test_logical_and() {
        let tpl = "{% if a and b %}both{% endif %}";
        let result = render(tpl, &json!({"a": true, "b": true})).unwrap();
        assert_eq!(result, "both");
    }

    #[test]
    fn test_filter_truncate() {
        let result =
            render("{{ s | truncate: 10 }}", &json!({"s": "This is a long string"})).unwrap();
        assert_eq!(result, "This is...");
    }

    #[test]
    fn test_filter_strip_html() {
        let result =
            render("{{ s | strip_html }}", &json!({"s": "<p>hello</p>"})).unwrap();
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_case() {
        let tpl = "{% case color %}{% when 'red' %}R{% when 'blue' %}B{% else %}?{% endcase %}";
        let r1 = render(tpl, &json!({"color": "red"})).unwrap();
        assert_eq!(r1, "R");
        let r2 = render(tpl, &json!({"color": "green"})).unwrap();
        assert_eq!(r2, "?");
    }

    #[test]
    fn test_increment_decrement() {
        let tpl = "{% increment x %}{% increment x %}{% increment x %}";
        let result = render(tpl, &json!({})).unwrap();
        assert_eq!(result, "012");
    }

    #[test]
    fn test_comment_ignored() {
        let tpl = "a{% comment %}hidden{% endcomment %}b";
        let result = render(tpl, &json!({})).unwrap();
        assert_eq!(result, "ab");
    }

    #[test]
    fn test_raw_output() {
        let tpl = "{% raw %}{{ not_rendered }}{% endraw %}";
        let result = render(tpl, &json!({})).unwrap();
        assert_eq!(result, "{{ not_rendered }}");
    }

    #[test]
    fn test_filter_default() {
        let tpl = "{{ missing | default: 'fallback' }}";
        let result = render(tpl, &json!({})).unwrap();
        assert_eq!(result, "fallback");
    }

    #[test]
    fn test_for_reversed() {
        let tpl = "{% for item in items reversed %}{{ item }}{% endfor %}";
        let result = render(tpl, &json!({"items": [1, 2, 3]})).unwrap();
        assert_eq!(result, "321");
    }

    #[test]
    fn test_filter_chaining() {
        let result = render(
            "{{ name | upcase | append: '!' }}",
            &json!({"name": "hello"}),
        )
        .unwrap();
        assert_eq!(result, "HELLO!");
    }

    #[test]
    fn test_contains_operator() {
        let tpl = "{% if msg contains 'hello' %}found{% endif %}";
        let result = render(tpl, &json!({"msg": "say hello world"})).unwrap();
        assert_eq!(result, "found");
    }

    #[test]
    fn test_null_is_falsy() {
        let tpl = "{% if val %}yes{% else %}no{% endif %}";
        let result = render(tpl, &json!({"val": null})).unwrap();
        assert_eq!(result, "no");
    }

    #[test]
    fn test_custom_filter() {
        let mut engine = LiquidEngine::new();
        engine.register_filter("double", |val, _args| {
            let n = val.as_f64().unwrap_or(0.0);
            serde_json::Number::from_f64(n * 2.0)
                .map_or(Value::Null, Value::Number)
        });
        let result = engine.render("{{ n | double }}", &json!({"n": 5})).unwrap();
        assert_eq!(result, "10");
    }

    #[test]
    fn test_filter_abs() {
        let result = render("{{ n | abs }}", &json!({"n": -5})).unwrap();
        assert_eq!(result, "5");
    }

    #[test]
    fn test_empty_template() {
        let result = render("", &json!({})).unwrap();
        assert_eq!(result, "");
    }
}
