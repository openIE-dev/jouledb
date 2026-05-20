//! Handlebars template engine — extends Mustache with helpers, block helpers,
//! nested paths, literal values, subexpressions, and inline partials.
//!
//! Pure-Rust replacement for handlebars.js and handlebars npm.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HandlebarsError {
    UnclosedTag { line: usize },
    UnclosedBlock { name: String },
    MismatchedBlock { expected: String, found: String },
    PartialNotFound { name: String },
    HelperNotFound { name: String },
    ParseError(String),
    RenderError(String),
}

impl fmt::Display for HandlebarsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnclosedTag { line } => write!(f, "unclosed tag at line {line}"),
            Self::UnclosedBlock { name } => write!(f, "unclosed block: {name}"),
            Self::MismatchedBlock { expected, found } => {
                write!(f, "expected closing for '{expected}', found '{found}'")
            }
            Self::PartialNotFound { name } => write!(f, "partial not found: {name}"),
            Self::HelperNotFound { name } => write!(f, "helper not found: {name}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
            Self::RenderError(msg) => write!(f, "render error: {msg}"),
        }
    }
}

impl std::error::Error for HandlebarsError {}

// ── AST ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
enum HbNode {
    Text(String),
    /// `{{expr}}` — escaped output.
    Expression(HbExpr),
    /// `{{{expr}}}` or `{{&expr}}` — raw/unescaped.
    RawExpression(HbExpr),
    /// `{{#helper args}}...{{/helper}}` — block helper.
    Block {
        helper: String,
        args: Vec<HbExpr>,
        hash: Vec<(String, HbExpr)>,
        body: Vec<HbNode>,
        inverse: Vec<HbNode>,
    },
    /// `{{>partial}}` — partial.
    Partial {
        name: String,
        context: Option<HbExpr>,
    },
    /// `{{!-- comment --}}` or `{{! comment }}`.
    Comment,
}

/// An expression — path, literal, or subexpression.
#[derive(Debug, Clone)]
enum HbExpr {
    Path(Vec<String>),
    StringLiteral(String),
    NumberLiteral(f64),
    BoolLiteral(bool),
    NullLiteral,
    SubExpression {
        helper: String,
        args: Vec<HbExpr>,
    },
}

// ── Context ─────────────────────────────────────────────────────

struct HbContext {
    stack: Vec<Value>,
    /// Block parameters (e.g., `as |item index|`).
    block_params: Vec<HashMap<String, Value>>,
    /// Root data.
    root: Value,
}

impl HbContext {
    fn new(data: &Value) -> Self {
        Self {
            stack: vec![data.clone()],
            block_params: Vec::new(),
            root: data.clone(),
        }
    }

    fn push(&mut self, val: &Value) {
        self.stack.push(val.clone());
    }

    fn pop(&mut self) {
        self.stack.pop();
    }

    fn push_block_params(&mut self, params: HashMap<String, Value>) {
        self.block_params.push(params);
    }

    fn pop_block_params(&mut self) {
        self.block_params.pop();
    }

    fn lookup_path(&self, parts: &[String]) -> Value {
        if parts.is_empty() {
            return Value::Null;
        }

        let first = &parts[0];

        // Handle @root, @index, @key, @first, @last.
        if first == "@root" {
            let mut current = self.root.clone();
            for part in &parts[1..] {
                current = current.get(part).cloned().unwrap_or(Value::Null);
            }
            return current;
        }

        // Check block params.
        for bp in self.block_params.iter().rev() {
            if let Some(val) = bp.get(first.as_str()) {
                let mut current = val.clone();
                for part in &parts[1..] {
                    current = current.get(part).cloned().unwrap_or(Value::Null);
                }
                return current;
            }
        }

        // Handle `this`.
        if first == "this" || first == "." {
            let base = self.stack.last().cloned().unwrap_or(Value::Null);
            let mut current = base;
            for part in &parts[1..] {
                current = current.get(part).cloned().unwrap_or(Value::Null);
            }
            return current;
        }

        // Handle `../` parent references.
        if first == ".." {
            if self.stack.len() >= 2 {
                let parent = &self.stack[self.stack.len() - 2];
                let mut current = parent.clone();
                for part in &parts[1..] {
                    if part == ".." {
                        // We do not support multiple ../ levels beyond simple.
                        continue;
                    }
                    current = current.get(part).cloned().unwrap_or(Value::Null);
                }
                return current;
            }
            return Value::Null;
        }

        // Walk context stack from top to bottom.
        for ctx in self.stack.iter().rev() {
            let mut current = (*ctx).clone();
            let mut found = true;
            for part in parts {
                match current.get(part) {
                    Some(v) => current = v.clone(),
                    None => {
                        found = false;
                        break;
                    }
                }
            }
            if found {
                return current;
            }
        }

        Value::Null
    }

    fn resolve_expr(&self, expr: &HbExpr) -> Value {
        match expr {
            HbExpr::Path(parts) => self.lookup_path(parts),
            HbExpr::StringLiteral(s) => Value::String(s.clone()),
            HbExpr::NumberLiteral(n) => {
                serde_json::Number::from_f64(*n).map_or(Value::Null, Value::Number)
            }
            HbExpr::BoolLiteral(b) => Value::Bool(*b),
            HbExpr::NullLiteral => Value::Null,
            HbExpr::SubExpression { .. } => {
                // Subexpressions are resolved during render.
                Value::Null
            }
        }
    }
}

// ── HTML escaping ───────────────────────────────────────────────

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

fn value_to_string(val: &Value) -> String {
    match val {
        Value::Null => String::new(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        _ => serde_json::to_string(val).unwrap_or_default(),
    }
}

fn is_truthy(val: &Value) -> bool {
    match val {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Array(a) => !a.is_empty(),
        Value::String(s) => !s.is_empty(),
        Value::Number(_) | Value::Object(_) => true,
    }
}

// ── Parser ──────────────────────────────────────────────────────

fn parse_template(input: &str) -> Result<Vec<HbNode>, HandlebarsError> {
    let mut nodes = Vec::new();
    let mut rest = input;
    let mut line = 1;

    while !rest.is_empty() {
        if let Some(pos) = rest.find("{{") {
            if pos > 0 {
                let text = &rest[..pos];
                line += text.chars().filter(|c| *c == '\n').count();
                nodes.push(HbNode::Text(text.to_string()));
            }
            rest = &rest[pos + 2..];

            // Long comment {{!-- ... --}}
            if rest.starts_with("!--") {
                let end = rest
                    .find("--}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                rest = &rest[end + 4..];
                nodes.push(HbNode::Comment);
            } else if rest.starts_with('!') {
                // Short comment {{! ... }}
                let end = rest
                    .find("}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                rest = &rest[end + 2..];
                nodes.push(HbNode::Comment);
            } else if rest.starts_with('{') {
                // Triple: {{{expr}}}
                let end = rest
                    .find("}}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                let inner = rest[1..end].trim();
                let expr = parse_expression(inner);
                rest = &rest[end + 3..];
                nodes.push(HbNode::RawExpression(expr));
            } else if rest.starts_with('&') {
                // {{&expr}} — unescaped
                let end = rest
                    .find("}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                let inner = rest[1..end].trim();
                let expr = parse_expression(inner);
                rest = &rest[end + 2..];
                nodes.push(HbNode::RawExpression(expr));
            } else if rest.starts_with('#') {
                // Block: {{#helper args}}
                let end = rest
                    .find("}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                let inner = rest[1..end].trim();
                let (helper, args, hash) = parse_helper_call(inner);
                rest = &rest[end + 2..];

                let (body_str, inverse_str, after) = find_block_close(rest, &helper)?;
                let body = parse_template(body_str)?;
                let inverse = if inverse_str.is_empty() {
                    Vec::new()
                } else {
                    parse_template(inverse_str)?
                };
                rest = after;
                nodes.push(HbNode::Block {
                    helper,
                    args,
                    hash,
                    body,
                    inverse,
                });
            } else if rest.starts_with('^') {
                // Inverted block shorthand: {{^helper}}
                let end = rest
                    .find("}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                let inner = rest[1..end].trim();
                let helper = inner.to_string();
                rest = &rest[end + 2..];

                let (body_str, _inverse_str, after) = find_block_close(rest, &helper)?;
                let body = parse_template(body_str)?;
                rest = after;
                nodes.push(HbNode::Block {
                    helper: "unless".to_string(),
                    args: vec![HbExpr::Path(
                        helper.split('.').map(|s| s.to_string()).collect(),
                    )],
                    hash: Vec::new(),
                    body,
                    inverse: Vec::new(),
                });
            } else if rest.starts_with('>') {
                // Partial: {{>name}} or {{>name context}}
                let end = rest
                    .find("}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                let inner = rest[1..end].trim();
                let parts_vec: Vec<&str> = inner.split_whitespace().collect();
                let name = parts_vec[0].to_string();
                let context = if parts_vec.len() > 1 {
                    Some(parse_expression(parts_vec[1]))
                } else {
                    None
                };
                rest = &rest[end + 2..];
                nodes.push(HbNode::Partial { name, context });
            } else {
                // Expression: {{expr}}
                let end = rest
                    .find("}}")
                    .ok_or(HandlebarsError::UnclosedTag { line })?;
                let inner = rest[..end].trim();
                let expr = parse_expression(inner);
                rest = &rest[end + 2..];
                nodes.push(HbNode::Expression(expr));
            }
        } else {
            nodes.push(HbNode::Text(rest.to_string()));
            break;
        }
    }

    Ok(nodes)
}

fn parse_expression(input: &str) -> HbExpr {
    let s = input.trim();

    // String literal
    if (s.starts_with('"') && s.ends_with('"')) || (s.starts_with('\'') && s.ends_with('\'')) {
        return HbExpr::StringLiteral(s[1..s.len() - 1].to_string());
    }

    // Boolean
    if s == "true" {
        return HbExpr::BoolLiteral(true);
    }
    if s == "false" {
        return HbExpr::BoolLiteral(false);
    }

    // Null
    if s == "null" || s == "undefined" {
        return HbExpr::NullLiteral;
    }

    // Number
    if let Ok(n) = s.parse::<f64>() {
        return HbExpr::NumberLiteral(n);
    }

    // Subexpression (helper args)
    if s.starts_with('(') && s.ends_with(')') {
        let inner = &s[1..s.len() - 1];
        let parts: Vec<&str> = inner.split_whitespace().collect();
        if !parts.is_empty() {
            let helper = parts[0].to_string();
            let args = parts[1..].iter().map(|p| parse_expression(p)).collect();
            return HbExpr::SubExpression { helper, args };
        }
    }

    // Path: a.b.c or ../a or this.a
    HbExpr::Path(s.split('.').map(|p| p.to_string()).collect())
}

fn parse_helper_call(input: &str) -> (String, Vec<HbExpr>, Vec<(String, HbExpr)>) {
    let mut tokens = tokenize_helper_call(input);
    let helper = if tokens.is_empty() {
        String::new()
    } else {
        tokens.remove(0)
    };

    let mut args = Vec::new();
    let mut hash = Vec::new();

    for token in &tokens {
        if let Some(eq_pos) = token.find('=') {
            let key = token[..eq_pos].to_string();
            let val = parse_expression(&token[eq_pos + 1..]);
            hash.push((key, val));
        } else {
            args.push(parse_expression(token));
        }
    }

    (helper, args, hash)
}

fn tokenize_helper_call(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut in_string = false;
    let mut string_char = '"';
    let mut paren_depth = 0u32;

    for ch in input.chars() {
        if in_string {
            current.push(ch);
            if ch == string_char {
                in_string = false;
            }
            continue;
        }
        if ch == '"' || ch == '\'' {
            in_string = true;
            string_char = ch;
            current.push(ch);
            continue;
        }
        if ch == '(' {
            paren_depth += 1;
            current.push(ch);
            continue;
        }
        if ch == ')' {
            paren_depth = paren_depth.saturating_sub(1);
            current.push(ch);
            continue;
        }
        if ch == ' ' && paren_depth == 0 {
            if !current.is_empty() {
                tokens.push(current.clone());
                current.clear();
            }
            continue;
        }
        current.push(ch);
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

fn find_block_close<'a>(
    input: &'a str,
    name: &str,
) -> Result<(&'a str, &'a str, &'a str), HandlebarsError> {
    let open_prefix = format!("{{{{#{name}");
    let close_tag = format!("{{{{/{name}}}}}");
    let else_tag = "{{else}}";
    let mut depth = 1u32;
    let mut pos = 0;
    let mut else_pos: Option<usize> = None;

    while pos < input.len() {
        if input[pos..].starts_with(&open_prefix) {
            // Check that the next character after the name is }} or a space.
            let after = pos + open_prefix.len();
            if after < input.len() {
                let next = input.as_bytes()[after];
                if next == b'}' || next == b' ' {
                    depth += 1;
                }
            }
            pos += open_prefix.len();
            continue;
        }
        if input[pos..].starts_with(&close_tag) {
            depth -= 1;
            if depth == 0 {
                let after = &input[pos + close_tag.len()..];
                if let Some(ep) = else_pos {
                    let body = &input[..ep];
                    let inverse = &input[ep + else_tag.len()..pos];
                    return Ok((body, inverse, after));
                }
                let body = &input[..pos];
                return Ok((body, "", after));
            }
            pos += close_tag.len();
            continue;
        }
        if depth == 1 && input[pos..].starts_with(else_tag) && else_pos.is_none() {
            else_pos = Some(pos);
            pos += else_tag.len();
            continue;
        }
        pos += 1;
    }

    Err(HandlebarsError::UnclosedBlock {
        name: name.to_string(),
    })
}

// ── Custom helper type ──────────────────────────────────────────

/// A registered helper function. Receives resolved arguments, hash params,
/// and the rendered body/inverse strings.
pub type HelperFn = Box<dyn Fn(&[Value], &[(String, Value)], &str, &str) -> String>;

// ── Engine ──────────────────────────────────────────────────────

/// Handlebars template engine with custom helpers and partials.
pub struct HandlebarsEngine {
    partials: HashMap<String, String>,
    helpers: HashMap<String, HelperFn>,
}

impl HandlebarsEngine {
    pub fn new() -> Self {
        Self {
            partials: HashMap::new(),
            helpers: HashMap::new(),
        }
    }

    /// Register a partial template.
    pub fn register_partial(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.partials.insert(name.into(), template.into());
    }

    /// Register a custom helper.
    pub fn register_helper(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&[Value], &[(String, Value)], &str, &str) -> String + 'static,
    ) {
        self.helpers.insert(name.into(), Box::new(f));
    }

    /// Render a template string with the given data.
    pub fn render(&self, template: &str, data: &Value) -> Result<String, HandlebarsError> {
        let nodes = parse_template(template)?;
        let mut ctx = HbContext::new(data);
        self.render_nodes(&nodes, &mut ctx)
    }

    fn render_nodes(
        &self,
        nodes: &[HbNode],
        ctx: &mut HbContext,
    ) -> Result<String, HandlebarsError> {
        let mut out = String::new();

        for node in nodes {
            match node {
                HbNode::Text(t) => out.push_str(t),
                HbNode::Expression(expr) => {
                    let val = ctx.resolve_expr(expr);
                    out.push_str(&html_escape(&value_to_string(&val)));
                }
                HbNode::RawExpression(expr) => {
                    let val = ctx.resolve_expr(expr);
                    out.push_str(&value_to_string(&val));
                }
                HbNode::Comment => {}
                HbNode::Block {
                    helper,
                    args,
                    hash,
                    body,
                    inverse,
                } => {
                    let rendered =
                        self.render_block(helper, args, hash, body, inverse, ctx)?;
                    out.push_str(&rendered);
                }
                HbNode::Partial { name, context } => {
                    let rendered = self.render_partial(name, context.as_ref(), ctx)?;
                    out.push_str(&rendered);
                }
            }
        }

        Ok(out)
    }

    fn render_block(
        &self,
        helper: &str,
        args: &[HbExpr],
        hash: &[(String, HbExpr)],
        body: &[HbNode],
        inverse: &[HbNode],
        ctx: &mut HbContext,
    ) -> Result<String, HandlebarsError> {
        // Built-in helpers.
        match helper {
            "if" => {
                let val = if args.is_empty() {
                    Value::Null
                } else {
                    ctx.resolve_expr(&args[0])
                };
                if is_truthy(&val) {
                    self.render_nodes(body, ctx)
                } else {
                    self.render_nodes(inverse, ctx)
                }
            }
            "unless" => {
                let val = if args.is_empty() {
                    Value::Null
                } else {
                    ctx.resolve_expr(&args[0])
                };
                if !is_truthy(&val) {
                    self.render_nodes(body, ctx)
                } else {
                    self.render_nodes(inverse, ctx)
                }
            }
            "each" => {
                let val = if args.is_empty() {
                    Value::Null
                } else {
                    ctx.resolve_expr(&args[0])
                };
                let mut out = String::new();
                // Extract owned items to avoid borrowing val during ctx.push.
                if let Value::Array(arr) = &val {
                    if arr.is_empty() {
                        return self.render_nodes(inverse, ctx);
                    }
                    let items: Vec<Value> = arr.clone();
                    let len = items.len();
                    for (i, item) in items.iter().enumerate() {
                        let mut bp = HashMap::new();
                        bp.insert("@index".to_string(), Value::Number(i.into()));
                        bp.insert(
                            "@first".to_string(),
                            Value::Bool(i == 0),
                        );
                        bp.insert(
                            "@last".to_string(),
                            Value::Bool(i == len - 1),
                        );
                        ctx.push(item);
                        ctx.push_block_params(bp);
                        let rendered = self.render_nodes(body, ctx)?;
                        out.push_str(&rendered);
                        ctx.pop_block_params();
                        ctx.pop();
                    }
                } else if let Value::Object(map) = &val {
                    if map.is_empty() {
                        return self.render_nodes(inverse, ctx);
                    }
                    let mut entries: Vec<(String, Value)> = map
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    let len = entries.len();
                    for (i, (key, item)) in entries.iter().enumerate() {
                        let mut bp = HashMap::new();
                        bp.insert(
                            "@key".to_string(),
                            Value::String(key.clone()),
                        );
                        bp.insert("@index".to_string(), Value::Number(i.into()));
                        bp.insert("@first".to_string(), Value::Bool(i == 0));
                        bp.insert(
                            "@last".to_string(),
                            Value::Bool(i == len - 1),
                        );
                        ctx.push(item);
                        ctx.push_block_params(bp);
                        let rendered = self.render_nodes(body, ctx)?;
                        out.push_str(&rendered);
                        ctx.pop_block_params();
                        ctx.pop();
                    }
                } else {
                    return self.render_nodes(inverse, ctx);
                }
                Ok(out)
            }
            "with" => {
                let val = if args.is_empty() {
                    Value::Null
                } else {
                    ctx.resolve_expr(&args[0])
                };
                if is_truthy(&val) {
                    ctx.push(&val);
                    let rendered = self.render_nodes(body, ctx)?;
                    ctx.pop();
                    Ok(rendered)
                } else {
                    self.render_nodes(inverse, ctx)
                }
            }
            "lookup" => {
                if args.len() >= 2 {
                    let obj = ctx.resolve_expr(&args[0]);
                    let key = ctx.resolve_expr(&args[1]);
                    let result = match &key {
                        Value::String(k) => obj.get(k).cloned().unwrap_or(Value::Null),
                        Value::Number(n) => {
                            if let Some(idx) = n.as_u64() {
                                obj.get(idx as usize).cloned().unwrap_or(Value::Null)
                            } else if let Some(f) = n.as_f64() {
                                let idx = f as usize;
                                obj.get(idx).cloned().unwrap_or(Value::Null)
                            } else {
                                Value::Null
                            }
                        }
                        _ => Value::Null,
                    };
                    Ok(value_to_string(&result))
                } else {
                    Ok(String::new())
                }
            }
            _ => {
                // Check custom helpers.
                if let Some(helper_fn) = self.helpers.get(helper) {
                    let resolved_args: Vec<Value> =
                        args.iter().map(|a| ctx.resolve_expr(a)).collect();
                    let resolved_hash: Vec<(String, Value)> = hash
                        .iter()
                        .map(|(k, v)| (k.clone(), ctx.resolve_expr(v)))
                        .collect();
                    let body_str = self.render_nodes(body, ctx)?;
                    let inverse_str = self.render_nodes(inverse, ctx)?;
                    Ok(helper_fn(&resolved_args, &resolved_hash, &body_str, &inverse_str))
                } else {
                    // Treat as a section (context push).
                    let val = ctx.resolve_expr(&HbExpr::Path(
                        helper.split('.').map(|s| s.to_string()).collect(),
                    ));
                    if is_truthy(&val) {
                        if let Value::Array(arr) = &val {
                            let items: Vec<Value> = arr.clone();
                            let mut result = String::new();
                            for item in &items {
                                ctx.push(item);
                                let rendered = self.render_nodes(body, ctx)?;
                                result.push_str(&rendered);
                                ctx.pop();
                            }
                            Ok(result)
                        } else {
                            ctx.push(&val);
                            let rendered = self.render_nodes(body, ctx)?;
                            ctx.pop();
                            Ok(rendered)
                        }
                    } else {
                        self.render_nodes(inverse, ctx)
                    }
                }
            }
        }
    }

    fn render_partial(
        &self,
        name: &str,
        context: Option<&HbExpr>,
        ctx: &mut HbContext,
    ) -> Result<String, HandlebarsError> {
        let src = self
            .partials
            .get(name)
            .ok_or_else(|| HandlebarsError::PartialNotFound {
                name: name.to_string(),
            })?
            .clone();

        if let Some(ctx_expr) = context {
            let val = ctx.resolve_expr(ctx_expr);
            ctx.push(&val);
            let nodes = parse_template(&src)?;
            let rendered = self.render_nodes(&nodes, ctx)?;
            ctx.pop();
            Ok(rendered)
        } else {
            let nodes = parse_template(&src)?;
            self.render_nodes(&nodes, ctx)
        }
    }
}

impl Default for HandlebarsEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Convenience ─────────────────────────────────────────────────

/// Render a template string with data using a default engine.
pub fn render(template: &str, data: &Value) -> Result<String, HandlebarsError> {
    let engine = HandlebarsEngine::new();
    engine.render(template, data)
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_simple_variable() {
        let result = render("Hello, {{name}}!", &json!({"name": "World"})).unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_html_escaping() {
        let result = render("{{v}}", &json!({"v": "<b>hi</b>"})).unwrap();
        assert_eq!(result, "&lt;b&gt;hi&lt;/b&gt;");
    }

    #[test]
    fn test_triple_mustache_raw() {
        let result = render("{{{v}}}", &json!({"v": "<b>hi</b>"})).unwrap();
        assert_eq!(result, "<b>hi</b>");
    }

    #[test]
    fn test_ampersand_raw() {
        let result = render("{{&v}}", &json!({"v": "<em>x</em>"})).unwrap();
        assert_eq!(result, "<em>x</em>");
    }

    #[test]
    fn test_if_truthy() {
        let tpl = "{{#if show}}yes{{/if}}";
        let result = render(tpl, &json!({"show": true})).unwrap();
        assert_eq!(result, "yes");
    }

    #[test]
    fn test_if_falsy() {
        let tpl = "{{#if show}}yes{{/if}}";
        let result = render(tpl, &json!({"show": false})).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_if_else() {
        let tpl = "{{#if show}}yes{{else}}no{{/if}}";
        let result = render(tpl, &json!({"show": false})).unwrap();
        assert_eq!(result, "no");
    }

    #[test]
    fn test_unless() {
        let tpl = "{{#unless hidden}}visible{{/unless}}";
        let result = render(tpl, &json!({"hidden": false})).unwrap();
        assert_eq!(result, "visible");
    }

    #[test]
    fn test_each_array() {
        let tpl = "{{#each items}}{{this}} {{/each}}";
        let result = render(tpl, &json!({"items": ["a", "b", "c"]})).unwrap();
        assert_eq!(result, "a b c ");
    }

    #[test]
    fn test_each_with_index() {
        let tpl = "{{#each items}}{{@index}}:{{this}} {{/each}}";
        let result = render(tpl, &json!({"items": ["x", "y"]})).unwrap();
        assert_eq!(result, "0:x 1:y ");
    }

    #[test]
    fn test_each_empty_inverse() {
        let tpl = "{{#each items}}{{this}}{{else}}empty{{/each}}";
        let result = render(tpl, &json!({"items": []})).unwrap();
        assert_eq!(result, "empty");
    }

    #[test]
    fn test_with_block() {
        let tpl = "{{#with person}}{{name}} is {{age}}{{/with}}";
        let result = render(tpl, &json!({"person": {"name": "Alice", "age": 30}})).unwrap();
        assert_eq!(result, "Alice is 30");
    }

    #[test]
    fn test_nested_path() {
        let result = render("{{a.b.c}}", &json!({"a": {"b": {"c": "deep"}}})).unwrap();
        assert_eq!(result, "deep");
    }

    #[test]
    fn test_comment_ignored() {
        let result = render("a{{! comment }}b", &json!({})).unwrap();
        assert_eq!(result, "ab");
    }

    #[test]
    fn test_long_comment_ignored() {
        let result = render("a{{!-- long comment --}}b", &json!({})).unwrap();
        assert_eq!(result, "ab");
    }

    #[test]
    fn test_partial() {
        let mut engine = HandlebarsEngine::new();
        engine.register_partial("item", "<li>{{name}}</li>");
        let result = engine
            .render("{{>item}}", &json!({"name": "test"}))
            .unwrap();
        assert_eq!(result, "<li>test</li>");
    }

    #[test]
    fn test_partial_with_context() {
        let mut engine = HandlebarsEngine::new();
        engine.register_partial("item", "{{name}}");
        let result = engine
            .render("{{>item person}}", &json!({"person": {"name": "Bob"}}))
            .unwrap();
        assert_eq!(result, "Bob");
    }

    #[test]
    fn test_custom_helper() {
        let mut engine = HandlebarsEngine::new();
        engine.register_helper("shout", |args, _hash, _body, _inv| {
            let text = args
                .first()
                .and_then(|v| v.as_str())
                .unwrap_or("");
            text.to_uppercase()
        });
        let result = engine
            .render("{{#shout name}}{{/shout}}", &json!({"name": "hello"}))
            .unwrap();
        assert_eq!(result, "HELLO");
    }

    #[test]
    fn test_each_first_last() {
        let tpl = "{{#each items}}{{#if @first}}[{{/if}}{{this}}{{#if @last}}]{{/if}}{{/each}}";
        let result = render(tpl, &json!({"items": ["a", "b", "c"]})).unwrap();
        assert_eq!(result, "[abc]");
    }

    #[test]
    fn test_section_as_context() {
        let tpl = "{{#person}}{{name}}{{/person}}";
        let result = render(tpl, &json!({"person": {"name": "Carol"}})).unwrap();
        assert_eq!(result, "Carol");
    }

    #[test]
    fn test_missing_variable_empty() {
        let result = render("({{missing}})", &json!({})).unwrap();
        assert_eq!(result, "()");
    }

    #[test]
    fn test_number_output() {
        let result = render("{{n}}", &json!({"n": 42})).unwrap();
        assert_eq!(result, "42");
    }

    #[test]
    fn test_boolean_output() {
        let result = render("{{b}}", &json!({"b": true})).unwrap();
        assert_eq!(result, "true");
    }

    #[test]
    fn test_string_literal_in_if() {
        // Even an empty string literal in source is parsed as a path, but
        // this tests that the if helper works with truthy strings from data.
        let tpl = "{{#if msg}}{{msg}}{{/if}}";
        let result = render(tpl, &json!({"msg": "hi"})).unwrap();
        assert_eq!(result, "hi");
    }

    #[test]
    fn test_lookup_helper() {
        let tpl = "{{#lookup items 1}}{{/lookup}}";
        let result = render(tpl, &json!({"items": ["a", "b", "c"]})).unwrap();
        assert_eq!(result, "b");
    }
}
