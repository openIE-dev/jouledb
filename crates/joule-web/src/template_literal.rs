//! Template string engine with interpolation, filters, conditionals, and loops.
//!
//! Supports `${variable}` interpolation, nested property access `${user.name}`,
//! filters/pipes `${name|upper}`, conditionals `${if cond}...${endif}`,
//! loops `${for item in list}...${endfor}`, default values `${name:default}`.

use std::collections::HashMap;
use std::fmt;

// ── Value ────────────────────────────────────────────────────────

/// A template context value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Str(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<Value>),
    Map(HashMap<String, Value>),
    Null,
}

impl Value {
    pub fn as_str(&self) -> String {
        match self {
            Value::Str(s) => s.clone(),
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => String::new(),
            Value::List(items) => {
                let parts: Vec<String> = items.iter().map(|v| v.as_str()).collect();
                parts.join(", ")
            }
            Value::Map(_) => "[object]".to_string(),
        }
    }

    pub fn is_truthy(&self) -> bool {
        match self {
            Value::Bool(b) => *b,
            Value::Str(s) => !s.is_empty(),
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Null => false,
            Value::List(l) => !l.is_empty(),
            Value::Map(m) => !m.is_empty(),
        }
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ── Context ──────────────────────────────────────────────────────

/// Template rendering context.
#[derive(Debug, Clone)]
pub struct Context {
    values: HashMap<String, Value>,
}

impl Context {
    pub fn new() -> Self {
        Self { values: HashMap::new() }
    }

    pub fn set(&mut self, key: &str, value: Value) {
        self.values.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&Value> {
        self.values.get(key)
    }

    /// Resolve a dotted path like "user.name".
    pub fn resolve(&self, path: &str) -> Value {
        let parts: Vec<&str> = path.split('.').collect();
        if parts.is_empty() { return Value::Null; }

        let mut current = match self.values.get(parts[0]) {
            Some(v) => v.clone(),
            None => return Value::Null,
        };

        for part in &parts[1..] {
            current = match &current {
                Value::Map(m) => match m.get(*part) {
                    Some(v) => v.clone(),
                    None => return Value::Null,
                },
                _ => return Value::Null,
            };
        }
        current
    }
}

// ── Filters ──────────────────────────────────────────────────────

fn apply_filter(value: &str, filter: &str) -> String {
    match filter.trim() {
        "upper" => value.to_uppercase(),
        "lower" => value.to_lowercase(),
        "trim" => value.trim().to_string(),
        "capitalize" => {
            let mut chars = value.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => {
                    let upper: String = c.to_uppercase().collect();
                    format!("{}{}", upper, chars.as_str())
                }
            }
        }
        "length" | "len" => value.len().to_string(),
        "reverse" => value.chars().rev().collect(),
        _ => value.to_string(),
    }
}

// ── Template Engine ──────────────────────────────────────────────

/// A compiled template.
pub struct Template {
    source: String,
}

#[derive(Debug)]
pub enum TemplateError {
    UnmatchedIf,
    UnmatchedFor,
    UnmatchedEndif,
    UnmatchedEndfor,
    ParseError(String),
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TemplateError::UnmatchedIf => write!(f, "unmatched ${{if}}"),
            TemplateError::UnmatchedFor => write!(f, "unmatched ${{for}}"),
            TemplateError::UnmatchedEndif => write!(f, "unmatched ${{endif}}"),
            TemplateError::UnmatchedEndfor => write!(f, "unmatched ${{endfor}}"),
            TemplateError::ParseError(msg) => write!(f, "parse error: {}", msg),
        }
    }
}

impl Template {
    /// Create a template from a source string.
    pub fn new(source: &str) -> Self {
        Self { source: source.to_string() }
    }

    /// Render the template with the given context.
    pub fn render(&self, ctx: &Context) -> Result<String, TemplateError> {
        let tokens = tokenize(&self.source);
        render_tokens(&tokens, ctx)
    }
}

#[derive(Debug, Clone)]
enum TplToken {
    Text(String),
    Expr(String),       // ${expr}
    If(String),         // ${if cond}
    EndIf,              // ${endif}
    For(String, String),// ${for item in list}
    EndFor,             // ${endfor}
}

fn tokenize(source: &str) -> Vec<TplToken> {
    let mut tokens = vec![];
    let mut remaining = source;

    while let Some(start) = remaining.find("${") {
        if start > 0 {
            tokens.push(TplToken::Text(remaining[..start].to_string()));
        }
        remaining = &remaining[start + 2..];
        if let Some(end) = remaining.find('}') {
            let expr = remaining[..end].trim().to_string();
            remaining = &remaining[end + 1..];

            if expr.starts_with("if ") {
                tokens.push(TplToken::If(expr[3..].trim().to_string()));
            } else if expr == "endif" {
                tokens.push(TplToken::EndIf);
            } else if expr.starts_with("for ") {
                // Parse "item in list"
                let body = expr[4..].trim();
                if let Some(in_pos) = body.find(" in ") {
                    let var = body[..in_pos].trim().to_string();
                    let list = body[in_pos + 4..].trim().to_string();
                    tokens.push(TplToken::For(var, list));
                } else {
                    tokens.push(TplToken::Text(format!("${{{}}}",  expr)));
                }
            } else if expr == "endfor" {
                tokens.push(TplToken::EndFor);
            } else {
                tokens.push(TplToken::Expr(expr));
            }
        } else {
            // No closing brace, treat as text
            tokens.push(TplToken::Text(format!("${{{}", remaining)));
            break;
        }
    }
    if !remaining.is_empty() {
        tokens.push(TplToken::Text(remaining.to_string()));
    }
    tokens
}

fn render_tokens(tokens: &[TplToken], ctx: &Context) -> Result<String, TemplateError> {
    let mut output = String::new();
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i] {
            TplToken::Text(s) => { output.push_str(s); i += 1; }
            TplToken::Expr(expr) => {
                output.push_str(&eval_expr(expr, ctx));
                i += 1;
            }
            TplToken::If(cond) => {
                // Find matching endif
                let (body, end_idx) = find_block_end(tokens, i + 1, "if")?;
                let val = ctx.resolve(cond);
                if val.is_truthy() {
                    output.push_str(&render_tokens(&body, ctx)?);
                }
                i = end_idx + 1;
            }
            TplToken::For(var, list_name) => {
                let (body, end_idx) = find_block_end(tokens, i + 1, "for")?;
                let list_val = ctx.resolve(list_name);
                if let Value::List(items) = list_val {
                    for item in &items {
                        let mut inner_ctx = ctx.clone();
                        inner_ctx.set(var, item.clone());
                        output.push_str(&render_tokens(&body, &inner_ctx)?);
                    }
                }
                i = end_idx + 1;
            }
            TplToken::EndIf => { return Err(TemplateError::UnmatchedEndif); }
            TplToken::EndFor => { return Err(TemplateError::UnmatchedEndfor); }
        }
    }
    Ok(output)
}

fn find_block_end(tokens: &[TplToken], start: usize, kind: &str)
    -> Result<(Vec<TplToken>, usize), TemplateError>
{
    let mut depth = 1;
    let mut i = start;
    while i < tokens.len() {
        match &tokens[i] {
            TplToken::If(_) if kind == "if" => depth += 1,
            TplToken::EndIf if kind == "if" => {
                depth -= 1;
                if depth == 0 { return Ok((tokens[start..i].to_vec(), i)); }
            }
            TplToken::For(_, _) if kind == "for" => depth += 1,
            TplToken::EndFor if kind == "for" => {
                depth -= 1;
                if depth == 0 { return Ok((tokens[start..i].to_vec(), i)); }
            }
            _ => {}
        }
        i += 1;
    }
    match kind {
        "if" => Err(TemplateError::UnmatchedIf),
        _ => Err(TemplateError::UnmatchedFor),
    }
}

fn eval_expr(expr: &str, ctx: &Context) -> String {
    // Check for default value: "name:default_value"
    let (expr_part, default) = if let Some(colon_pos) = expr.rfind(':') {
        // Only treat as default if the part before colon looks like a var name
        let before = &expr[..colon_pos];
        // Make sure it's not a filter expression
        if !before.contains('|') {
            (before.trim(), Some(expr[colon_pos + 1..].trim()))
        } else {
            (expr, None)
        }
    } else {
        (expr, None)
    };

    // Check for filters: "name|filter1|filter2"
    let parts: Vec<&str> = expr_part.split('|').collect();
    let var_name = parts[0].trim();
    let filters = &parts[1..];

    let val = ctx.resolve(var_name);
    let mut result = match &val {
        Value::Null => {
            match default {
                Some(d) => return d.to_string(),
                None => String::new(),
            }
        }
        v => v.as_str(),
    };

    for filter in filters {
        result = apply_filter(&result, filter);
    }
    result
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx_with(pairs: &[(&str, Value)]) -> Context {
        let mut ctx = Context::new();
        for (k, v) in pairs { ctx.set(k, v.clone()); }
        ctx
    }

    #[test]
    fn test_simple_interpolation() {
        let tpl = Template::new("Hello, ${name}!");
        let ctx = ctx_with(&[("name", Value::Str("World".into()))]);
        assert_eq!(tpl.render(&ctx).unwrap(), "Hello, World!");
    }

    #[test]
    fn test_nested_property() {
        let mut user = HashMap::new();
        user.insert("name".to_string(), Value::Str("Alice".into()));
        let ctx = ctx_with(&[("user", Value::Map(user))]);
        let tpl = Template::new("Hi ${user.name}");
        assert_eq!(tpl.render(&ctx).unwrap(), "Hi Alice");
    }

    #[test]
    fn test_filter_upper() {
        let ctx = ctx_with(&[("name", Value::Str("alice".into()))]);
        let tpl = Template::new("${name|upper}");
        assert_eq!(tpl.render(&ctx).unwrap(), "ALICE");
    }

    #[test]
    fn test_filter_lower() {
        let ctx = ctx_with(&[("name", Value::Str("ALICE".into()))]);
        let tpl = Template::new("${name|lower}");
        assert_eq!(tpl.render(&ctx).unwrap(), "alice");
    }

    #[test]
    fn test_filter_chain() {
        let ctx = ctx_with(&[("name", Value::Str("  alice  ".into()))]);
        let tpl = Template::new("${name|trim|upper}");
        assert_eq!(tpl.render(&ctx).unwrap(), "ALICE");
    }

    #[test]
    fn test_default_value() {
        let ctx = Context::new();
        let tpl = Template::new("Hello, ${name:stranger}!");
        assert_eq!(tpl.render(&ctx).unwrap(), "Hello, stranger!");
    }

    #[test]
    fn test_conditional_true() {
        let ctx = ctx_with(&[("show", Value::Bool(true))]);
        let tpl = Template::new("${if show}visible${endif}");
        assert_eq!(tpl.render(&ctx).unwrap(), "visible");
    }

    #[test]
    fn test_conditional_false() {
        let ctx = ctx_with(&[("show", Value::Bool(false))]);
        let tpl = Template::new("${if show}hidden${endif}");
        assert_eq!(tpl.render(&ctx).unwrap(), "");
    }

    #[test]
    fn test_loop() {
        let ctx = ctx_with(&[("items", Value::List(vec![
            Value::Str("a".into()),
            Value::Str("b".into()),
            Value::Str("c".into()),
        ]))]);
        let tpl = Template::new("${for item in items}[${item}]${endfor}");
        assert_eq!(tpl.render(&ctx).unwrap(), "[a][b][c]");
    }

    #[test]
    fn test_plain_text() {
        let tpl = Template::new("no variables here");
        let ctx = Context::new();
        assert_eq!(tpl.render(&ctx).unwrap(), "no variables here");
    }

    #[test]
    fn test_int_interpolation() {
        let ctx = ctx_with(&[("count", Value::Int(42))]);
        let tpl = Template::new("Count: ${count}");
        assert_eq!(tpl.render(&ctx).unwrap(), "Count: 42");
    }

    #[test]
    fn test_missing_var() {
        let ctx = Context::new();
        let tpl = Template::new("${missing}");
        assert_eq!(tpl.render(&ctx).unwrap(), "");
    }

    #[test]
    fn test_filter_capitalize() {
        let ctx = ctx_with(&[("name", Value::Str("alice".into()))]);
        let tpl = Template::new("${name|capitalize}");
        assert_eq!(tpl.render(&ctx).unwrap(), "Alice");
    }
}
