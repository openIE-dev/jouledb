//! Email template engine — variable substitution, conditionals, loops, inline CSS.
//!
//! Replaces Handlebars / Mustache / MJML with a pure-Rust template engine
//! purpose-built for transactional and marketing email rendering.

use serde_json::Value;
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Template errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// Unbalanced block tag.
    UnbalancedTag { tag: String },
    /// Undefined variable referenced in template.
    UndefinedVariable { name: String },
    /// Mismatched block close.
    MismatchedClose { expected: String, found: String },
    /// Invalid syntax.
    InvalidSyntax(String),
}

impl std::fmt::Display for TemplateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnbalancedTag { tag } => write!(f, "unbalanced tag: {tag}"),
            Self::UndefinedVariable { name } => write!(f, "undefined variable: {name}"),
            Self::MismatchedClose { expected, found } => {
                write!(f, "expected closing for '{expected}', found '{found}'")
            }
            Self::InvalidSyntax(msg) => write!(f, "invalid syntax: {msg}"),
        }
    }
}

impl std::error::Error for TemplateError {}

// ── Inline CSS ──────────────────────────────────────────────────

/// A set of CSS rules to be inlined as style attributes.
#[derive(Debug, Clone, Default)]
pub struct InlineCss {
    /// Selector -> property:value pairs.
    rules: Vec<(String, Vec<(String, String)>)>,
}

impl InlineCss {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule: selector maps to CSS declarations.
    pub fn add_rule(&mut self, selector: &str, declarations: &[(&str, &str)]) {
        let decls = declarations
            .iter()
            .map(|(p, v)| (p.to_string(), v.to_string()))
            .collect();
        self.rules.push((selector.to_string(), decls));
    }

    /// Apply inline CSS to HTML by injecting style attributes on matching tags.
    /// Simple tag-name matching only (no class/id selectors — keeps it dependency-free).
    pub fn apply(&self, html: &str) -> String {
        let mut result = html.to_string();
        for (selector, decls) in &self.rules {
            let style_str: String = decls
                .iter()
                .map(|(p, v)| format!("{p}: {v}"))
                .collect::<Vec<_>>()
                .join("; ");
            // Match <selector or <selector  (tag name match)
            let open = format!("<{selector}");
            let replacement = format!("<{selector} style=\"{style_str}\"");
            result = result.replace(&open, &replacement);
        }
        result
    }
}

// ── EmailTemplate ───────────────────────────────────────────────

/// An email template with subject and body variants.
#[derive(Debug, Clone)]
pub struct EmailTemplate {
    pub subject: String,
    pub html_body: String,
    pub text_body: String,
    pub preheader: Option<String>,
    pub inline_css: Option<InlineCss>,
}

impl EmailTemplate {
    pub fn new(subject: &str, html_body: &str, text_body: &str) -> Self {
        Self {
            subject: subject.to_string(),
            html_body: html_body.to_string(),
            text_body: text_body.to_string(),
            preheader: None,
            inline_css: None,
        }
    }

    /// Set preheader text (preview text in email clients).
    pub fn with_preheader(mut self, preheader: &str) -> Self {
        self.preheader = Some(preheader.to_string());
        self
    }

    /// Set inline CSS rules.
    pub fn with_inline_css(mut self, css: InlineCss) -> Self {
        self.inline_css = Some(css);
        self
    }

    /// Validate the template: check balanced tags and undefined variables.
    pub fn validate(&self, vars: &HashMap<String, Value>) -> Result<(), Vec<TemplateError>> {
        let mut errors = Vec::new();
        // Check all three template sources
        for source in [&self.subject, &self.html_body, &self.text_body] {
            if let Err(mut errs) = validate_template(source, vars) {
                errors.append(&mut errs);
            }
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Render the template with the given variables.
    pub fn render(&self, vars: &HashMap<String, Value>) -> Result<RenderedEmail, TemplateError> {
        let subject = render_template(&self.subject, vars)?;
        let mut html = render_template(&self.html_body, vars)?;
        let text = render_template(&self.text_body, vars)?;

        // Apply inline CSS if present.
        if let Some(css) = &self.inline_css {
            html = css.apply(&html);
        }

        // Inject preheader if present.
        if let Some(preheader) = &self.preheader {
            let rendered_preheader = render_template(preheader, vars)?;
            let preheader_html = format!(
                "<span style=\"display:none;max-height:0;overflow:hidden\">{rendered_preheader}</span>"
            );
            // Insert after <body> if present, otherwise prepend.
            if let Some(pos) = html.find("<body>") {
                html.insert_str(pos + 6, &preheader_html);
            } else {
                html = format!("{preheader_html}{html}");
            }
        }

        Ok(RenderedEmail {
            subject,
            html_body: html,
            text_body: text,
        })
    }
}

/// A rendered email ready to send.
#[derive(Debug, Clone)]
pub struct RenderedEmail {
    pub subject: String,
    pub html_body: String,
    pub text_body: String,
}

// ── Template Engine ─────────────────────────────────────────────

/// Render a template string with variable substitution, conditionals, and loops.
fn render_template(
    template: &str,
    vars: &HashMap<String, Value>,
) -> Result<String, TemplateError> {
    let mut output = String::with_capacity(template.len());
    let mut pos = 0;
    let bytes = template.as_bytes();

    while pos < bytes.len() {
        if pos + 1 < bytes.len() && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
            // Find closing }}
            let start = pos + 2;
            let end = find_closing_braces(template, start)?;
            let tag_content = template[start..end].trim();

            if let Some(stripped) = tag_content.strip_prefix("#if ") {
                let condition = stripped.trim();
                let block_end = find_block_end(template, end + 2, "if")?;
                let inner = &template[end + 2..block_end];

                // Check for {{else}}
                let (if_block, else_block) = split_else(inner);

                let truthy = is_truthy(condition, vars);
                let block = if truthy { if_block } else { else_block };
                let rendered = render_template(block, vars)?;
                output.push_str(&rendered);

                // Skip past {{/if}}
                let after_close = find_closing_braces(template, block_end + 2 + 3)? + 2; // +3 for "/if"
                pos = after_close;
            } else if let Some(stripped) = tag_content.strip_prefix("#each ") {
                let list_name = stripped.trim();
                let block_end = find_block_end(template, end + 2, "each")?;
                let inner = &template[end + 2..block_end];

                if let Some(Value::Array(items)) = vars.get(list_name) {
                    for item in items {
                        // Create a child scope with {{this}} and item fields.
                        let mut child_vars = vars.clone();
                        child_vars.insert("this".to_string(), item.clone());
                        if let Value::Object(map) = item {
                            for (k, v) in map {
                                child_vars.insert(k.clone(), v.clone());
                            }
                        }
                        let rendered = render_template(inner, &child_vars)?;
                        output.push_str(&rendered);
                    }
                }

                // Skip past {{/each}}
                let after_close =
                    find_closing_braces(template, block_end + 2 + 4)? + 2; // +4 for "/each" (minus the '/')
                pos = after_close;
            } else if tag_content.starts_with('/') {
                // Stray closing tag — should have been consumed.
                return Err(TemplateError::InvalidSyntax(format!(
                    "unexpected closing tag: {tag_content}"
                )));
            } else {
                // Variable substitution.
                let val = resolve_var(tag_content, vars);
                output.push_str(&val);
                pos = end + 2;
            }
        } else {
            output.push(bytes[pos] as char);
            pos += 1;
        }
    }

    Ok(output)
}

/// Find the closing `}}` starting from `start`.
fn find_closing_braces(template: &str, start: usize) -> Result<usize, TemplateError> {
    let bytes = template.as_bytes();
    let mut i = start;
    while i + 1 < bytes.len() {
        if bytes[i] == b'}' && bytes[i + 1] == b'}' {
            return Ok(i);
        }
        i += 1;
    }
    Err(TemplateError::InvalidSyntax(
        "unclosed {{ tag".to_string(),
    ))
}

/// Find the position of the matching `{{/tag}}` block end.
fn find_block_end(template: &str, start: usize, tag: &str) -> Result<usize, TemplateError> {
    let open_pattern = format!("{{{{#{tag}");
    let close_pattern = format!("{{{{/{tag}}}}}");
    let mut depth = 1u32;
    let mut pos = start;

    while pos < template.len() {
        if template[pos..].starts_with(&close_pattern) {
            depth -= 1;
            if depth == 0 {
                return Ok(pos);
            }
            pos += close_pattern.len();
        } else if template[pos..].starts_with(&open_pattern) {
            depth += 1;
            pos += open_pattern.len();
        } else {
            pos += 1;
        }
    }

    Err(TemplateError::UnbalancedTag {
        tag: tag.to_string(),
    })
}

/// Split a block at `{{else}}`, returning (if_block, else_block).
fn split_else(block: &str) -> (&str, &str) {
    if let Some(pos) = block.find("{{else}}") {
        (&block[..pos], &block[pos + 8..])
    } else {
        (block, "")
    }
}

/// Check if a condition is truthy in the variable context.
fn is_truthy(condition: &str, vars: &HashMap<String, Value>) -> bool {
    match vars.get(condition) {
        Some(Value::Bool(b)) => *b,
        Some(Value::Null) => false,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().is_some_and(|v| v != 0.0),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(_)) => true,
        None => false,
    }
}

/// Resolve a variable name to its string representation.
fn resolve_var(name: &str, vars: &HashMap<String, Value>) -> String {
    match vars.get(name) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        Some(Value::Null) => String::new(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

/// Validate template for balanced tags and undefined variables.
fn validate_template(
    template: &str,
    vars: &HashMap<String, Value>,
) -> Result<(), Vec<TemplateError>> {
    let mut errors = Vec::new();
    let mut stack: Vec<String> = Vec::new();
    let mut pos = 0;
    let bytes = template.as_bytes();

    while pos < bytes.len() {
        if pos + 1 < bytes.len() && bytes[pos] == b'{' && bytes[pos + 1] == b'{' {
            let start = pos + 2;
            match find_closing_braces(template, start) {
                Ok(end) => {
                    let tag_content = template[start..end].trim();
                    if let Some(stripped) = tag_content.strip_prefix("#if ") {
                        stack.push(format!("if:{}", stripped.trim()));
                    } else if let Some(stripped) = tag_content.strip_prefix("#each ") {
                        let list_name = stripped.trim();
                        if !vars.contains_key(list_name) {
                            errors.push(TemplateError::UndefinedVariable {
                                name: list_name.to_string(),
                            });
                        }
                        stack.push(format!("each:{list_name}"));
                    } else if let Some(stripped) = tag_content.strip_prefix('/') {
                        let close_tag = stripped.trim();
                        if let Some(top) = stack.pop() {
                            let expected_type =
                                top.split(':').next().unwrap_or("");
                            if expected_type != close_tag {
                                errors.push(TemplateError::MismatchedClose {
                                    expected: expected_type.to_string(),
                                    found: close_tag.to_string(),
                                });
                            }
                        } else {
                            errors.push(TemplateError::UnbalancedTag {
                                tag: close_tag.to_string(),
                            });
                        }
                    } else if tag_content != "else" && tag_content != "this" {
                        if !vars.contains_key(tag_content) {
                            errors.push(TemplateError::UndefinedVariable {
                                name: tag_content.to_string(),
                            });
                        }
                    }
                    pos = end + 2;
                }
                Err(e) => {
                    errors.push(e);
                    break;
                }
            }
        } else {
            pos += 1;
        }
    }

    for remaining in &stack {
        let tag = remaining.split(':').next().unwrap_or(remaining);
        errors.push(TemplateError::UnbalancedTag {
            tag: tag.to_string(),
        });
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn vars(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    #[test]
    fn test_simple_variable_substitution() {
        let tpl = EmailTemplate::new("Hello {{name}}", "<p>{{name}}</p>", "Hi {{name}}");
        let v = vars(&[("name", json!("Alice"))]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.subject, "Hello Alice");
        assert_eq!(rendered.html_body, "<p>Alice</p>");
        assert_eq!(rendered.text_body, "Hi Alice");
    }

    #[test]
    fn test_multiple_variables() {
        let tpl = EmailTemplate::new("{{greeting}} {{name}}", "", "");
        let v = vars(&[("greeting", json!("Hi")), ("name", json!("Bob"))]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.subject, "Hi Bob");
    }

    #[test]
    fn test_conditional_true() {
        let tpl = EmailTemplate::new("", "{{#if premium}}VIP{{/if}}", "");
        let v = vars(&[("premium", json!(true))]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.html_body, "VIP");
    }

    #[test]
    fn test_conditional_false() {
        let tpl = EmailTemplate::new("", "{{#if premium}}VIP{{/if}}", "");
        let v = vars(&[("premium", json!(false))]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.html_body, "");
    }

    #[test]
    fn test_conditional_else() {
        let tpl = EmailTemplate::new("", "{{#if premium}}VIP{{else}}Free{{/if}}", "");
        let v = vars(&[("premium", json!(false))]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.html_body, "Free");
    }

    #[test]
    fn test_each_loop() {
        let tpl = EmailTemplate::new("", "{{#each items}}{{name}},{{/each}}", "");
        let v = vars(&[(
            "items",
            json!([{"name": "A"}, {"name": "B"}, {"name": "C"}]),
        )]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.html_body, "A,B,C,");
    }

    #[test]
    fn test_each_with_this() {
        let tpl = EmailTemplate::new("", "{{#each colors}}{{this}} {{/each}}", "");
        let v = vars(&[("colors", json!(["red", "green", "blue"]))]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.html_body, "red green blue ");
    }

    #[test]
    fn test_inline_css() {
        let mut css = InlineCss::new();
        css.add_rule("p", &[("color", "red"), ("font-size", "14px")]);
        let tpl = EmailTemplate::new("", "<p>Hello</p>", "").with_inline_css(css);
        let v = HashMap::new();
        let rendered = tpl.render(&v).unwrap();
        assert!(rendered.html_body.contains("style=\"color: red; font-size: 14px\""));
    }

    #[test]
    fn test_preheader() {
        let tpl = EmailTemplate::new("", "<body><p>Hi</p></body>", "")
            .with_preheader("Preview text");
        let v = HashMap::new();
        let rendered = tpl.render(&v).unwrap();
        assert!(rendered.html_body.contains("display:none"));
        assert!(rendered.html_body.contains("Preview text"));
    }

    #[test]
    fn test_validate_balanced_tags() {
        let tpl = EmailTemplate::new("", "{{#if x}}hello", "");
        let v = vars(&[("x", json!(true))]);
        let result = tpl.validate(&v);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_undefined_variable() {
        let tpl = EmailTemplate::new("{{missing}}", "", "");
        let v = HashMap::new();
        let result = tpl.validate(&v);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, TemplateError::UndefinedVariable { name } if name == "missing")));
    }

    #[test]
    fn test_number_variable() {
        let tpl = EmailTemplate::new("Count: {{count}}", "", "");
        let v = vars(&[("count", json!(42))]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.subject, "Count: 42");
    }

    #[test]
    fn test_nested_conditional_in_loop() {
        let tpl = EmailTemplate::new(
            "",
            "{{#each users}}{{#if active}}{{name}}{{/if}}{{/each}}",
            "",
        );
        let v = vars(&[(
            "users",
            json!([
                {"name": "Alice", "active": true},
                {"name": "Bob", "active": false},
                {"name": "Carol", "active": true}
            ]),
        )]);
        let rendered = tpl.render(&v).unwrap();
        assert_eq!(rendered.html_body, "AliceCarol");
    }

    #[test]
    fn test_empty_template() {
        let tpl = EmailTemplate::new("", "", "");
        let rendered = tpl.render(&HashMap::new()).unwrap();
        assert_eq!(rendered.subject, "");
        assert_eq!(rendered.html_body, "");
    }
}
