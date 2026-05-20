//! Template engine with Mustache/Handlebars-subset syntax.
//!
//! Replaces Handlebars, Mustache, EJS with a pure-Rust compile-and-render
//! pipeline that parses templates into an AST and evaluates them against
//! `serde_json::Value` data.

use serde_json::Value;
use std::collections::HashMap;
use thiserror::Error;

// ── Errors ──

#[derive(Debug, Error)]
pub enum TemplateError {
    #[error("template not found: {0}")]
    NotFound(String),
    #[error("parse error: {0}")]
    Parse(String),
    #[error("render error: {0}")]
    Render(String),
}

// ── AST ──

#[derive(Debug, Clone)]
enum TemplateNode {
    Text(String),
    Expr(String),         // {{variable}}
    RawExpr(String),      // {{{variable}}}
    If {
        condition: String,
        body: Vec<TemplateNode>,
        else_body: Vec<TemplateNode>,
    },
    Unless {
        condition: String,
        body: Vec<TemplateNode>,
    },
    Each {
        expr: String,
        body: Vec<TemplateNode>,
    },
    Partial(String),
    Helper {
        name: String,
        args: Vec<String>,
    },
    Comment,
}

// ── Engine ──

/// Template engine with partials and helpers.
pub struct TemplateEngine {
    templates: HashMap<String, String>,
    helpers: HashMap<String, Box<dyn Fn(&[Value]) -> String>>,
    partials: HashMap<String, String>,
}

impl TemplateEngine {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
            helpers: HashMap::new(),
            partials: HashMap::new(),
        }
    }

    /// Register a named template.
    pub fn register(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.templates.insert(name.into(), template.into());
    }

    /// Register a partial template.
    pub fn register_partial(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.partials.insert(name.into(), template.into());
    }

    /// Register a helper function.
    pub fn register_helper(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&[Value]) -> String + 'static,
    ) {
        self.helpers.insert(name.into(), Box::new(f));
    }

    /// Render a registered template by name.
    pub fn render(&self, name: &str, data: &Value) -> Result<String, TemplateError> {
        let src = self
            .templates
            .get(name)
            .ok_or_else(|| TemplateError::NotFound(name.to_string()))?;
        self.render_string(src, data)
    }

    /// Render an inline template string.
    pub fn render_string(&self, template: &str, data: &Value) -> Result<String, TemplateError> {
        let nodes = parse(template)?;
        self.render_nodes(&nodes, data)
    }

    fn render_nodes(&self, nodes: &[TemplateNode], data: &Value) -> Result<String, TemplateError> {
        let mut out = String::new();
        for node in nodes {
            match node {
                TemplateNode::Text(t) => out.push_str(t),
                TemplateNode::Comment => {}
                TemplateNode::Expr(path) => {
                    let val = resolve(data, path);
                    out.push_str(&html_escape(&value_to_string(&val)));
                }
                TemplateNode::RawExpr(path) => {
                    let val = resolve(data, path);
                    out.push_str(&value_to_string(&val));
                }
                TemplateNode::If { condition, body, else_body } => {
                    if is_truthy(&resolve(data, condition)) {
                        out.push_str(&self.render_nodes(body, data)?);
                    } else {
                        out.push_str(&self.render_nodes(else_body, data)?);
                    }
                }
                TemplateNode::Unless { condition, body } => {
                    if !is_truthy(&resolve(data, condition)) {
                        out.push_str(&self.render_nodes(body, data)?);
                    }
                }
                TemplateNode::Each { expr, body } => {
                    let val = resolve(data, expr);
                    if let Value::Array(arr) = &val {
                        for (i, item) in arr.iter().enumerate() {
                            // Build a context that merges the item with @index and this
                            let mut ctx = match item {
                                Value::Object(m) => Value::Object(m.clone()),
                                _ => Value::Object(serde_json::Map::new()),
                            };
                            if let Value::Object(ref mut m) = ctx {
                                m.insert("@index".to_string(), Value::Number(i.into()));
                                m.insert("this".to_string(), item.clone());
                            }
                            out.push_str(&self.render_nodes(body, &ctx)?);
                        }
                    } else if let Value::Object(map) = &val {
                        for (k, v) in map {
                            let mut ctx = serde_json::Map::new();
                            ctx.insert("@key".to_string(), Value::String(k.clone()));
                            ctx.insert("this".to_string(), v.clone());
                            out.push_str(
                                &self.render_nodes(body, &Value::Object(ctx))?,
                            );
                        }
                    }
                }
                TemplateNode::Partial(name) => {
                    if let Some(src) = self.partials.get(name.as_str()) {
                        let nodes = parse(src)?;
                        out.push_str(&self.render_nodes(&nodes, data)?);
                    }
                }
                TemplateNode::Helper { name, args } => {
                    if let Some(f) = self.helpers.get(name.as_str()) {
                        let vals: Vec<Value> = args
                            .iter()
                            .map(|a| {
                                // Try as JSON literal first, then resolve path
                                if let Ok(v) = serde_json::from_str::<Value>(a) {
                                    v
                                } else {
                                    resolve(data, a)
                                }
                            })
                            .collect();
                        out.push_str(&f(&vals));
                    }
                }
            }
        }
        Ok(out)
    }
}

impl Default for TemplateEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Parser ──

fn parse(src: &str) -> Result<Vec<TemplateNode>, TemplateError> {
    let mut nodes = Vec::new();
    let mut rest = src;

    while !rest.is_empty() {
        // Look for the next `{{` (or `{{{`)
        if let Some(pos) = rest.find("{{") {
            if pos > 0 {
                nodes.push(TemplateNode::Text(rest[..pos].to_string()));
            }
            rest = &rest[pos..];

            // Triple-stache raw?
            if rest.starts_with("{{{") {
                let end = rest.find("}}}")
                    .ok_or_else(|| TemplateError::Parse("unclosed {{{".into()))?;
                let inner = rest[3..end].trim();
                nodes.push(TemplateNode::RawExpr(inner.to_string()));
                rest = &rest[end + 3..];
            } else {
                // Double-stache
                let end = rest.find("}}")
                    .ok_or_else(|| TemplateError::Parse("unclosed {{".into()))?;
                let inner = rest[2..end].trim();
                rest = &rest[end + 2..];

                if let Some(stripped) = inner.strip_prefix('!') {
                    let _ = stripped; // comment — just consume
                    nodes.push(TemplateNode::Comment);
                } else if let Some(stripped) = inner.strip_prefix("#if ") {
                    let cond = stripped.trim().to_string();
                    let (body, else_body, remaining) = parse_block(rest, "if")?;
                    rest = remaining;
                    nodes.push(TemplateNode::If {
                        condition: cond,
                        body,
                        else_body,
                    });
                } else if let Some(stripped) = inner.strip_prefix("#unless ") {
                    let cond = stripped.trim().to_string();
                    let (body, _else, remaining) = parse_block(rest, "unless")?;
                    rest = remaining;
                    nodes.push(TemplateNode::Unless {
                        condition: cond,
                        body,
                    });
                } else if let Some(stripped) = inner.strip_prefix("#each ") {
                    let expr = stripped.trim().to_string();
                    let (body, _else, remaining) = parse_block(rest, "each")?;
                    rest = remaining;
                    nodes.push(TemplateNode::Each { expr, body });
                } else if let Some(stripped) = inner.strip_prefix("> ") {
                    nodes.push(TemplateNode::Partial(stripped.trim().to_string()));
                } else {
                    // Could be helper (multiple words) or simple expression
                    let parts: Vec<&str> = inner.split_whitespace().collect();
                    if parts.len() > 1 && !parts[0].contains('.') {
                        nodes.push(TemplateNode::Helper {
                            name: parts[0].to_string(),
                            args: parts[1..].iter().map(|s| s.to_string()).collect(),
                        });
                    } else {
                        nodes.push(TemplateNode::Expr(inner.to_string()));
                    }
                }
            }
        } else {
            // No more tags
            nodes.push(TemplateNode::Text(rest.to_string()));
            break;
        }
    }

    Ok(nodes)
}

/// Parse until `{{/tag}}`, handling `{{else}}`.
fn parse_block<'a>(
    src: &'a str,
    tag: &str,
) -> Result<(Vec<TemplateNode>, Vec<TemplateNode>, &'a str), TemplateError> {
    let close = format!("{{{{/{tag}}}}}");
    let close_pos = src.find(&close).ok_or_else(|| {
        TemplateError::Parse(format!("unclosed {{{{#{tag}}}}}"))
    })?;

    let block = &src[..close_pos];
    let remaining = &src[close_pos + close.len()..];

    // Split on {{else}} if present
    if let Some(else_pos) = block.find("{{else}}") {
        let body_src = &block[..else_pos];
        let else_src = &block[else_pos + 8..];
        Ok((parse(body_src)?, parse(else_src)?, remaining))
    } else {
        Ok((parse(block)?, Vec::new(), remaining))
    }
}

// ── Helpers ──

fn resolve(data: &Value, path: &str) -> Value {
    if path == "this" || path == "." {
        // If data is an object with a "this" key (set by #each), return that.
        if let Value::Object(m) = data {
            if let Some(v) = m.get("this") {
                return v.clone();
            }
        }
        return data.clone();
    }
    let mut current = data;
    for segment in path.split('.') {
        match current {
            Value::Object(map) => {
                if let Some(v) = map.get(segment) {
                    current = v;
                } else {
                    return Value::Null;
                }
            }
            _ => return Value::Null,
        }
    }
    current.clone()
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Null => String::new(),
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

fn is_truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().unwrap_or(0.0) != 0.0,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(_) => true,
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn simple_variable() {
        let engine = TemplateEngine::new();
        let out = engine.render_string("Hello, {{name}}!", &json!({"name": "world"})).unwrap();
        assert_eq!(out, "Hello, world!");
    }

    #[test]
    fn nested_path() {
        let engine = TemplateEngine::new();
        let data = json!({"person": {"name": "Alice"}});
        let out = engine.render_string("Hi {{person.name}}", &data).unwrap();
        assert_eq!(out, "Hi Alice");
    }

    #[test]
    fn html_escaping() {
        let engine = TemplateEngine::new();
        let data = json!({"x": "<b>bold</b>"});
        let out = engine.render_string("{{x}}", &data).unwrap();
        assert_eq!(out, "&lt;b&gt;bold&lt;/b&gt;");
    }

    #[test]
    fn raw_output() {
        let engine = TemplateEngine::new();
        let data = json!({"x": "<b>bold</b>"});
        let out = engine.render_string("{{{x}}}", &data).unwrap();
        assert_eq!(out, "<b>bold</b>");
    }

    #[test]
    fn if_else() {
        let engine = TemplateEngine::new();
        let t = "{{#if show}}yes{{else}}no{{/if}}";
        assert_eq!(
            engine.render_string(t, &json!({"show": true})).unwrap(),
            "yes"
        );
        assert_eq!(
            engine.render_string(t, &json!({"show": false})).unwrap(),
            "no"
        );
    }

    #[test]
    fn each_with_array() {
        let engine = TemplateEngine::new();
        let data = json!({"items": ["a", "b", "c"]});
        let out = engine.render_string("{{#each items}}{{this}},{{/each}}", &data).unwrap();
        assert_eq!(out, "a,b,c,");
    }

    #[test]
    fn each_with_index() {
        let engine = TemplateEngine::new();
        let data = json!({"items": ["x", "y"]});
        let out = engine
            .render_string("{{#each items}}{{@index}}:{{this}} {{/each}}", &data)
            .unwrap();
        assert_eq!(out, "0:x 1:y ");
    }

    #[test]
    fn unless_block() {
        let engine = TemplateEngine::new();
        let t = "{{#unless hidden}}visible{{/unless}}";
        assert_eq!(
            engine.render_string(t, &json!({"hidden": false})).unwrap(),
            "visible"
        );
        assert_eq!(
            engine.render_string(t, &json!({"hidden": true})).unwrap(),
            ""
        );
    }

    #[test]
    fn partial_inclusion() {
        let mut engine = TemplateEngine::new();
        engine.register_partial("greeting", "Hello, {{name}}!");
        engine.register("page", "{{> greeting}}");
        let out = engine.render("page", &json!({"name": "Bob"})).unwrap();
        assert_eq!(out, "Hello, Bob!");
    }

    #[test]
    fn helper_function() {
        let mut engine = TemplateEngine::new();
        engine.register_helper("upper", |args: &[Value]| {
            args.first()
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_uppercase()
        });
        let data = json!({"name": "alice"});
        let out = engine.render_string("{{upper name}}", &data).unwrap();
        assert_eq!(out, "ALICE");
    }

    #[test]
    fn comment_stripped() {
        let engine = TemplateEngine::new();
        let out = engine
            .render_string("before{{! this is a comment }}after", &json!({}))
            .unwrap();
        assert_eq!(out, "beforeafter");
    }

    #[test]
    fn missing_variable_empty_string() {
        let engine = TemplateEngine::new();
        let out = engine.render_string("Hello, {{name}}!", &json!({})).unwrap();
        assert_eq!(out, "Hello, !");
    }

    #[test]
    fn nested_if_each() {
        let engine = TemplateEngine::new();
        let data = json!({"show": true, "items": [1, 2]});
        let out = engine
            .render_string(
                "{{#if show}}{{#each items}}{{this}},{{/each}}{{/if}}",
                &data,
            )
            .unwrap();
        assert_eq!(out, "1,2,");
    }

    #[test]
    fn template_not_found() {
        let engine = TemplateEngine::new();
        assert!(engine.render("missing", &json!({})).is_err());
    }
}
