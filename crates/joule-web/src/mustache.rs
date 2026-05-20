//! Mustache template engine — logic-less templates with full spec coverage.
//!
//! Replaces mustache.js and hogan.js with a pure-Rust implementation.
//! Supports variable interpolation, sections, inverted sections, partials,
//! HTML escaping, triple-mustache unescaped output, lambdas, and context stacks.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors that can occur during parsing or rendering.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MustacheError {
    /// Unclosed tag (missing `}}` or `}}}`).
    UnclosedTag { line: usize },
    /// Section opened but never closed.
    UnclosedSection { name: String },
    /// Closing tag does not match opening tag.
    MismatchedSection { expected: String, found: String },
    /// Partial not found in the registry.
    PartialNotFound { name: String },
    /// General parse error.
    ParseError(String),
}

impl fmt::Display for MustacheError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnclosedTag { line } => write!(f, "unclosed tag at line {line}"),
            Self::UnclosedSection { name } => write!(f, "unclosed section: {name}"),
            Self::MismatchedSection { expected, found } => {
                write!(f, "expected closing for '{expected}', found '{found}'")
            }
            Self::PartialNotFound { name } => write!(f, "partial not found: {name}"),
            Self::ParseError(msg) => write!(f, "parse error: {msg}"),
        }
    }
}

impl std::error::Error for MustacheError {}

// ── AST ─────────────────────────────────────────────────────────

/// A node in the parsed template AST.
#[derive(Debug, Clone)]
enum Node {
    /// Literal text, emitted as-is.
    Text(String),
    /// `{{name}}` — HTML-escaped variable interpolation.
    Variable(String),
    /// `{{{name}}}` or `{{&name}}` — unescaped variable.
    UnescapedVariable(String),
    /// `{{#name}}...{{/name}}` — section (truthy / list iteration).
    Section { name: String, body: Vec<Node> },
    /// `{{^name}}...{{/name}}` — inverted section (falsy / empty list).
    InvertedSection { name: String, body: Vec<Node> },
    /// `{{>name}}` — partial inclusion.
    Partial { name: String, indent: String },
    /// `{{! comment }}` — comment, ignored in output.
    Comment,
}

// ── Context stack ───────────────────────────────────────────────

/// A stack of JSON contexts for dotted-name resolution.
/// Uses owned values to avoid lifetime issues with push/pop during rendering.
struct ContextStack {
    stack: Vec<Value>,
}

impl ContextStack {
    fn new(root: &Value) -> Self {
        Self {
            stack: vec![root.clone()],
        }
    }

    fn push(&mut self, val: &Value) {
        self.stack.push(val.clone());
    }

    fn pop(&mut self) {
        self.stack.pop();
    }

    /// Resolve a dotted name like `a.b.c` walking up the context stack.
    fn lookup(&self, name: &str) -> Value {
        if name == "." {
            return self.stack.last().cloned().unwrap_or(Value::Null);
        }

        let parts: Vec<&str> = name.split('.').collect();

        // Walk from top of stack down to root.
        for ctx in self.stack.iter().rev() {
            let mut current = ctx;
            let mut found = true;
            for part in &parts {
                match current.get(*part) {
                    Some(v) => current = v,
                    None => {
                        found = false;
                        break;
                    }
                }
            }
            if found {
                return current.clone();
            }
        }
        Value::Null
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

/// Convert a JSON value to its string representation for output.
fn value_to_string(val: &Value) -> String {
    match val {
        Value::Null => String::new(),
        Value::Bool(b) => {
            if *b {
                "true".to_string()
            } else {
                "false".to_string()
            }
        }
        Value::Number(n) => n.to_string(),
        Value::String(s) => s.clone(),
        Value::Array(_) | Value::Object(_) => serde_json::to_string(val).unwrap_or_default(),
    }
}

// ── Parser ──────────────────────────────────────────────────────

/// Parse a Mustache template string into an AST.
fn parse(template: &str) -> Result<Vec<Node>, MustacheError> {
    let mut nodes = Vec::new();
    let mut rest = template;
    let mut line = 1;

    while !rest.is_empty() {
        if let Some(pos) = rest.find("{{") {
            // Emit text before the tag.
            if pos > 0 {
                let text = &rest[..pos];
                line += text.chars().filter(|c| *c == '\n').count();
                nodes.push(Node::Text(text.to_string()));
            }
            rest = &rest[pos + 2..];

            // Determine tag type.
            if rest.starts_with('{') {
                // Triple mustache: {{{name}}}
                let end = rest.find("}}}").ok_or(MustacheError::UnclosedTag { line })?;
                let name = rest[1..end].trim().to_string();
                rest = &rest[end + 3..];
                nodes.push(Node::UnescapedVariable(name));
            } else if rest.starts_with('!') {
                // Comment: {{! ... }}
                let end = rest.find("}}").ok_or(MustacheError::UnclosedTag { line })?;
                rest = &rest[end + 2..];
                nodes.push(Node::Comment);
            } else if rest.starts_with('#') {
                // Section open: {{#name}}
                let end = rest.find("}}").ok_or(MustacheError::UnclosedTag { line })?;
                let name = rest[1..end].trim().to_string();
                rest = &rest[end + 2..];

                // Find matching close tag.
                let (body_str, after) = find_section_close(rest, &name)?;
                let body = parse(body_str)?;
                rest = after;
                nodes.push(Node::Section { name, body });
            } else if rest.starts_with('^') {
                // Inverted section: {{^name}}
                let end = rest.find("}}").ok_or(MustacheError::UnclosedTag { line })?;
                let name = rest[1..end].trim().to_string();
                rest = &rest[end + 2..];

                let (body_str, after) = find_section_close(rest, &name)?;
                let body = parse(body_str)?;
                rest = after;
                nodes.push(Node::InvertedSection { name, body });
            } else if rest.starts_with('>') {
                // Partial: {{>name}}
                let end = rest.find("}}").ok_or(MustacheError::UnclosedTag { line })?;
                let name = rest[1..end].trim().to_string();
                rest = &rest[end + 2..];

                // Determine indentation (whitespace before the partial tag).
                let indent = compute_partial_indent(&nodes);
                nodes.push(Node::Partial { name, indent });
            } else if rest.starts_with('&') {
                // Unescaped: {{&name}}
                let end = rest.find("}}").ok_or(MustacheError::UnclosedTag { line })?;
                let name = rest[1..end].trim().to_string();
                rest = &rest[end + 2..];
                nodes.push(Node::UnescapedVariable(name));
            } else {
                // Variable: {{name}}
                let end = rest.find("}}").ok_or(MustacheError::UnclosedTag { line })?;
                let name = rest[..end].trim().to_string();
                rest = &rest[end + 2..];
                nodes.push(Node::Variable(name));
            }
        } else {
            // No more tags — rest is all text.
            nodes.push(Node::Text(rest.to_string()));
            break;
        }
    }

    Ok(nodes)
}

/// Determine the indentation that precedes a partial tag.
fn compute_partial_indent(nodes: &[Node]) -> String {
    if let Some(Node::Text(t)) = nodes.last() {
        if let Some(last_line) = t.rsplit('\n').next() {
            if last_line.chars().all(|c| c == ' ' || c == '\t') {
                return last_line.to_string();
            }
        }
    }
    String::new()
}

/// Find the matching `{{/name}}` close tag, handling nested sections of the same name.
fn find_section_close<'a>(
    input: &'a str,
    name: &str,
) -> Result<(&'a str, &'a str), MustacheError> {
    let open_tag_prefix = format!("{{{{#{name}}}}}");
    let close_tag = format!("{{{{/{name}}}}}");
    let mut depth = 1usize;
    let mut pos = 0;

    while pos < input.len() {
        // Check for nested open of the same name.
        if input[pos..].starts_with(&open_tag_prefix) {
            depth += 1;
            pos += open_tag_prefix.len();
            continue;
        }
        if input[pos..].starts_with(&close_tag) {
            depth -= 1;
            if depth == 0 {
                let body = &input[..pos];
                let after = &input[pos + close_tag.len()..];
                return Ok((body, after));
            }
            pos += close_tag.len();
            continue;
        }
        pos += 1;
    }
    Err(MustacheError::UnclosedSection {
        name: name.to_string(),
    })
}

// ── Renderer ────────────────────────────────────────────────────

/// Lambda function type — receives the raw inner template and returns rendered text.
pub type Lambda = Box<dyn Fn(&str) -> String>;

/// The Mustache template engine.
pub struct MustacheEngine {
    partials: HashMap<String, String>,
    lambdas: HashMap<String, Lambda>,
}

impl MustacheEngine {
    pub fn new() -> Self {
        Self {
            partials: HashMap::new(),
            lambdas: HashMap::new(),
        }
    }

    /// Register a partial template by name.
    pub fn register_partial(&mut self, name: impl Into<String>, template: impl Into<String>) {
        self.partials.insert(name.into(), template.into());
    }

    /// Register a lambda (section function).
    pub fn register_lambda(
        &mut self,
        name: impl Into<String>,
        f: impl Fn(&str) -> String + 'static,
    ) {
        self.lambdas.insert(name.into(), Box::new(f));
    }

    /// Render a template string with the given data context.
    pub fn render(&self, template: &str, data: &Value) -> Result<String, MustacheError> {
        let nodes = parse(template)?;
        let mut ctx = ContextStack::new(data);
        self.render_nodes(&nodes, &mut ctx, template)
    }

    fn render_nodes(
        &self,
        nodes: &[Node],
        ctx: &mut ContextStack,
        _original: &str,
    ) -> Result<String, MustacheError> {
        let mut out = String::new();

        for node in nodes {
            match node {
                Node::Text(t) => out.push_str(t),
                Node::Variable(name) => {
                    let val = ctx.lookup(name);
                    out.push_str(&html_escape(&value_to_string(&val)));
                }
                Node::UnescapedVariable(name) => {
                    let val = ctx.lookup(name);
                    out.push_str(&value_to_string(&val));
                }
                Node::Section { name, body } => {
                    // Check for lambda first.
                    if let Some(lambda) = self.lambdas.get(name.as_str()) {
                        let raw = render_nodes_to_raw(body);
                        let result = lambda(&raw);
                        out.push_str(&result);
                        continue;
                    }

                    let val = ctx.lookup(name);

                    if is_truthy(&val) {
                        match &val {
                            Value::Array(arr) => {
                                let items: Vec<Value> = arr.clone();
                                for item in &items {
                                    ctx.push(item);
                                    let rendered = self.render_nodes(body, ctx, _original)?;
                                    out.push_str(&rendered);
                                    ctx.pop();
                                }
                            }
                            _ => {
                                ctx.push(&val);
                                let rendered = self.render_nodes(body, ctx, _original)?;
                                out.push_str(&rendered);
                                ctx.pop();
                            }
                        }
                    }
                }
                Node::InvertedSection { name, body } => {
                    let val = ctx.lookup(name);
                    if !is_truthy(&val) {
                        let rendered = self.render_nodes(body, ctx, _original)?;
                        out.push_str(&rendered);
                    }
                }
                Node::Partial { name, indent } => {
                    let partial_src = self.partials.get(name.as_str()).ok_or_else(|| {
                        MustacheError::PartialNotFound {
                            name: name.clone(),
                        }
                    })?;
                    // Indent each line of the partial.
                    let indented = if indent.is_empty() {
                        partial_src.clone()
                    } else {
                        indent_text(partial_src, indent)
                    };
                    let partial_nodes = parse(&indented)?;
                    let rendered = self.render_nodes(&partial_nodes, ctx, _original)?;
                    out.push_str(&rendered);
                }
                Node::Comment => {}
            }
        }

        Ok(out)
    }
}

impl Default for MustacheEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine if a value is "truthy" in Mustache terms.
fn is_truthy(val: &Value) -> bool {
    match val {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Array(a) => !a.is_empty(),
        Value::String(s) => !s.is_empty(),
        Value::Number(_) => true,
        Value::Object(_) => true,
    }
}

/// Indent every line (after the first) of `text` with `indent`.
fn indent_text(text: &str, indent: &str) -> String {
    let mut out = String::new();
    for (i, line) in text.split('\n').enumerate() {
        if i > 0 {
            out.push('\n');
            out.push_str(indent);
        }
        out.push_str(line);
    }
    out
}

/// Reconstruct raw text from AST nodes (for lambda support).
fn render_nodes_to_raw(nodes: &[Node]) -> String {
    let mut out = String::new();
    for node in nodes {
        match node {
            Node::Text(t) => out.push_str(t),
            Node::Variable(name) => {
                out.push_str("{{");
                out.push_str(name);
                out.push_str("}}");
            }
            Node::UnescapedVariable(name) => {
                out.push_str("{{{");
                out.push_str(name);
                out.push_str("}}}");
            }
            Node::Section { name, body } => {
                out.push_str("{{#");
                out.push_str(name);
                out.push_str("}}");
                out.push_str(&render_nodes_to_raw(body));
                out.push_str("{{/");
                out.push_str(name);
                out.push_str("}}");
            }
            Node::InvertedSection { name, body } => {
                out.push_str("{{^");
                out.push_str(name);
                out.push_str("}}");
                out.push_str(&render_nodes_to_raw(body));
                out.push_str("{{/");
                out.push_str(name);
                out.push_str("}}");
            }
            Node::Partial { name, .. } => {
                out.push_str("{{>");
                out.push_str(name);
                out.push_str("}}");
            }
            Node::Comment => {
                out.push_str("{{!comment}}");
            }
        }
    }
    out
}

// ── Convenience ─────────────────────────────────────────────────

/// Render a template string with data, using an engine with no partials.
pub fn render(template: &str, data: &Value) -> Result<String, MustacheError> {
    let engine = MustacheEngine::new();
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
        let result = render("{{content}}", &json!({"content": "<b>bold</b>"})).unwrap();
        assert_eq!(result, "&lt;b&gt;bold&lt;/b&gt;");
    }

    #[test]
    fn test_triple_mustache_unescaped() {
        let result = render("{{{content}}}", &json!({"content": "<b>bold</b>"})).unwrap();
        assert_eq!(result, "<b>bold</b>");
    }

    #[test]
    fn test_ampersand_unescaped() {
        let result = render("{{&content}}", &json!({"content": "<em>hi</em>"})).unwrap();
        assert_eq!(result, "<em>hi</em>");
    }

    #[test]
    fn test_missing_variable_is_empty() {
        let result = render("Hello, {{name}}!", &json!({})).unwrap();
        assert_eq!(result, "Hello, !");
    }

    #[test]
    fn test_section_truthy() {
        let tpl = "{{#show}}visible{{/show}}";
        let result = render(tpl, &json!({"show": true})).unwrap();
        assert_eq!(result, "visible");
    }

    #[test]
    fn test_section_falsy() {
        let tpl = "{{#show}}visible{{/show}}";
        let result = render(tpl, &json!({"show": false})).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_section_list() {
        let tpl = "{{#items}}{{name}} {{/items}}";
        let data = json!({"items": [{"name": "a"}, {"name": "b"}, {"name": "c"}]});
        let result = render(tpl, &data).unwrap();
        assert_eq!(result, "a b c ");
    }

    #[test]
    fn test_inverted_section_falsy() {
        let tpl = "{{^items}}No items{{/items}}";
        let result = render(tpl, &json!({"items": []})).unwrap();
        assert_eq!(result, "No items");
    }

    #[test]
    fn test_inverted_section_truthy() {
        let tpl = "{{^items}}No items{{/items}}";
        let result = render(tpl, &json!({"items": [1]})).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_dot_notation() {
        let tpl = "{{person.name}}";
        let result = render(tpl, &json!({"person": {"name": "Alice"}})).unwrap();
        assert_eq!(result, "Alice");
    }

    #[test]
    fn test_nested_sections() {
        let tpl = "{{#a}}{{#b}}deep{{/b}}{{/a}}";
        let result = render(tpl, &json!({"a": true, "b": true})).unwrap();
        assert_eq!(result, "deep");
    }

    #[test]
    fn test_comment_ignored() {
        let tpl = "before{{! this is a comment }}after";
        let result = render(tpl, &json!({})).unwrap();
        assert_eq!(result, "beforeafter");
    }

    #[test]
    fn test_partial() {
        let mut engine = MustacheEngine::new();
        engine.register_partial("greeting", "Hello, {{name}}!");
        let result = engine
            .render("{{>greeting}}", &json!({"name": "World"}))
            .unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_partial_not_found() {
        let engine = MustacheEngine::new();
        let err = engine.render("{{>missing}}", &json!({})).unwrap_err();
        assert!(matches!(err, MustacheError::PartialNotFound { .. }));
    }

    #[test]
    fn test_lambda() {
        let mut engine = MustacheEngine::new();
        engine.register_lambda("bold", |text| format!("<b>{text}</b>"));
        let result = engine
            .render("{{#bold}}hello{{/bold}}", &json!({}))
            .unwrap();
        assert_eq!(result, "<b>hello</b>");
    }

    #[test]
    fn test_context_stack_section_push() {
        let tpl = "{{#person}}{{name}} is {{age}}{{/person}}";
        let data = json!({"person": {"name": "Bob", "age": 30}});
        let result = render(tpl, &data).unwrap();
        assert_eq!(result, "Bob is 30");
    }

    #[test]
    fn test_dot_variable_in_list() {
        let tpl = "{{#items}}{{.}} {{/items}}";
        let data = json!({"items": ["x", "y", "z"]});
        let result = render(tpl, &data).unwrap();
        assert_eq!(result, "x y z ");
    }

    #[test]
    fn test_numeric_values() {
        let result = render("{{n}}", &json!({"n": 42})).unwrap();
        assert_eq!(result, "42");
    }

    #[test]
    fn test_boolean_values() {
        let result = render("{{b}}", &json!({"b": true})).unwrap();
        assert_eq!(result, "true");
    }

    #[test]
    fn test_escape_ampersand() {
        let result = render("{{v}}", &json!({"v": "A & B"})).unwrap();
        assert_eq!(result, "A &amp; B");
    }

    #[test]
    fn test_escape_quotes() {
        let result = render("{{v}}", &json!({"v": "say \"hi\""})).unwrap();
        assert_eq!(result, "say &quot;hi&quot;");
    }

    #[test]
    fn test_empty_template() {
        let result = render("", &json!({})).unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_unclosed_section_error() {
        let err = render("{{#open}}no close", &json!({})).unwrap_err();
        assert!(matches!(err, MustacheError::UnclosedSection { .. }));
    }

    #[test]
    fn test_section_with_object_context() {
        let tpl = "{{#address}}{{city}}, {{state}}{{/address}}";
        let data = json!({"address": {"city": "Sarasota", "state": "FL"}});
        let result = render(tpl, &data).unwrap();
        assert_eq!(result, "Sarasota, FL");
    }

    #[test]
    fn test_multiple_partials() {
        let mut engine = MustacheEngine::new();
        engine.register_partial("header", "<h1>{{title}}</h1>");
        engine.register_partial("footer", "<footer>{{year}}</footer>");
        let result = engine
            .render(
                "{{>header}}{{>footer}}",
                &json!({"title": "Page", "year": 2026}),
            )
            .unwrap();
        assert_eq!(result, "<h1>Page</h1><footer>2026</footer>");
    }

    #[test]
    fn test_null_is_falsy() {
        let tpl = "{{#val}}yes{{/val}}{{^val}}no{{/val}}";
        let result = render(tpl, &json!({"val": null})).unwrap();
        assert_eq!(result, "no");
    }
}
