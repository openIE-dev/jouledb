//! Syntax transformation model — replaces Babel / SWC transforms in pure Rust.
//!
//! Operates on source text via pattern matching and string replacement.
//! Each transform is a `TransformRule` that can be chained into a pipeline.

use std::fmt;

// ── TransformRule trait ─────────────────────────────────────────

/// A single syntax transformation that rewrites source code.
pub trait TransformRule: fmt::Debug {
    /// Human-readable name of this transform.
    fn name(&self) -> &str;

    /// Apply the transform to `input`, returning transformed source.
    fn apply(&self, input: &str) -> String;
}

// ── Transform Pipeline ──────────────────────────────────────────

/// Chains multiple transforms in sequence.
#[derive(Debug, Default)]
pub struct TransformPipeline {
    rules: Vec<Box<dyn TransformRule>>,
}

impl TransformPipeline {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_rule(&mut self, rule: Box<dyn TransformRule>) {
        self.rules.push(rule);
    }

    pub fn transform(&self, input: &str) -> String {
        let mut src = input.to_string();
        for rule in &self.rules {
            src = rule.apply(&src);
        }
        src
    }

    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── Arrow Function Transform ────────────────────────────────────

/// Converts `(x) => expr` to `function(x) { return expr; }` (model-level).
#[derive(Debug)]
pub struct ArrowFunctionTransform;

impl TransformRule for ArrowFunctionTransform {
    fn name(&self) -> &str {
        "arrow-function"
    }

    fn apply(&self, input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let mut chars: Vec<char> = input.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            // Look for => pattern
            if i + 1 < len && chars[i] == '=' && chars[i + 1] == '>' {
                // Find the parameter list before =>
                // Walk backwards to find ( ... ) or a single identifier
                let before = &out;
                if let Some(params) = extract_arrow_params(before) {
                    // Remove the params from output
                    let trim_len = out.len() - params.raw_len;
                    out.truncate(trim_len);

                    // Skip =>
                    i += 2;
                    // Skip whitespace
                    while i < len && chars[i].is_ascii_whitespace() {
                        i += 1;
                    }

                    // Check if body is a block { ... } or an expression
                    if i < len && chars[i] == '{' {
                        out.push_str(&format!("function({}) ", params.text));
                        // Copy the block as-is
                        let mut depth = 0;
                        while i < len {
                            if chars[i] == '{' {
                                depth += 1;
                            } else if chars[i] == '}' {
                                depth -= 1;
                            }
                            out.push(chars[i]);
                            i += 1;
                            if depth == 0 {
                                break;
                            }
                        }
                    } else {
                        // Expression body — collect until ; or , or ) or end
                        let mut expr = String::new();
                        let mut depth = 0i32;
                        while i < len {
                            match chars[i] {
                                '(' | '[' => {
                                    depth += 1;
                                    expr.push(chars[i]);
                                }
                                ')' | ']' => {
                                    if depth <= 0 {
                                        break;
                                    }
                                    depth -= 1;
                                    expr.push(chars[i]);
                                }
                                ';' | ',' if depth == 0 => break,
                                _ => expr.push(chars[i]),
                            }
                            i += 1;
                        }
                        out.push_str(&format!(
                            "function({}) {{ return {}; }}",
                            params.text,
                            expr.trim()
                        ));
                    }
                    continue;
                }
            }
            out.push(chars[i]);
            i += 1;
        }

        let _ = chars; // suppress unused warning on older compilers
        out
    }
}

struct ArrowParams {
    text: String,
    raw_len: usize,
}

fn extract_arrow_params(before: &str) -> Option<ArrowParams> {
    let trimmed = before.trim_end();
    if trimmed.ends_with(')') {
        // Find matching (
        let chars: Vec<char> = trimmed.chars().collect();
        let mut depth = 0;
        let mut start = None;
        for j in (0..chars.len()).rev() {
            if chars[j] == ')' {
                depth += 1;
            } else if chars[j] == '(' {
                depth -= 1;
                if depth == 0 {
                    start = Some(j);
                    break;
                }
            }
        }
        if let Some(s) = start {
            let params_str: String = chars[s + 1..chars.len() - 1].iter().collect();
            // raw_len = from start of ( to end of string
            let raw_len = before.len() - before[..].rfind('(').unwrap_or(0);
            // Actually: we need the number of bytes from the ( to the end of `before`
            let paren_pos = before.rfind('(').unwrap_or(0);
            let raw_len = before.len() - paren_pos;
            // But also include any whitespace before the (
            let prefix = &before[..paren_pos];
            let space_count = prefix.len() - prefix.trim_end().len();
            let raw_len = raw_len + space_count;
            return Some(ArrowParams {
                text: params_str.trim().to_string(),
                raw_len,
            });
        }
    }

    // Single identifier: `x =>`
    let trimmed = before.trim_end();
    let last_token_start = trimmed
        .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '$')
        .map(|p| p + 1)
        .unwrap_or(0);
    let ident = &trimmed[last_token_start..];
    if !ident.is_empty() && ident.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '$')
    {
        let raw_len = before.len() - last_token_start;
        return Some(ArrowParams {
            text: ident.to_string(),
            raw_len,
        });
    }

    None
}

// ── Template Literal Transform ──────────────────────────────────

/// Converts `` `hello ${name}` `` to `"hello " + name`.
#[derive(Debug)]
pub struct TemplateLiteralTransform;

impl TransformRule for TemplateLiteralTransform {
    fn name(&self) -> &str {
        "template-literal"
    }

    fn apply(&self, input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let chars: Vec<char> = input.chars().collect();
        let len = chars.len();
        let mut i = 0;

        while i < len {
            if chars[i] == '`' {
                i += 1;
                let mut parts: Vec<String> = Vec::new();
                let mut current = String::new();

                while i < len && chars[i] != '`' {
                    if i + 1 < len && chars[i] == '$' && chars[i + 1] == '{' {
                        if !current.is_empty() {
                            parts.push(format!("\"{}\"", current));
                            current.clear();
                        }
                        i += 2;
                        let mut depth = 1;
                        let mut expr = String::new();
                        while i < len && depth > 0 {
                            if chars[i] == '{' {
                                depth += 1;
                            } else if chars[i] == '}' {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            }
                            expr.push(chars[i]);
                            i += 1;
                        }
                        if i < len {
                            i += 1; // skip }
                        }
                        parts.push(expr);
                    } else if chars[i] == '\\' && i + 1 < len {
                        current.push(chars[i + 1]);
                        i += 2;
                    } else {
                        current.push(chars[i]);
                        i += 1;
                    }
                }
                if i < len {
                    i += 1; // skip closing `
                }
                if !current.is_empty() {
                    parts.push(format!("\"{}\"", current));
                }
                if parts.is_empty() {
                    out.push_str("\"\"");
                } else {
                    out.push_str(&parts.join(" + "));
                }
                continue;
            }
            out.push(chars[i]);
            i += 1;
        }
        out
    }
}

// ── Optional Chaining Transform ─────────────────────────────────

/// Converts `a?.b` to `a != null ? a.b : undefined`.
#[derive(Debug)]
pub struct OptionalChainingTransform;

impl TransformRule for OptionalChainingTransform {
    fn name(&self) -> &str {
        "optional-chaining"
    }

    fn apply(&self, input: &str) -> String {
        // Simple pattern: identifier?.property
        let mut result = input.to_string();
        while let Some(pos) = result.find("?.") {
            // Extract the identifier before ?.
            let before = &result[..pos];
            let ident_start = before
                .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '$' && c != '.')
                .map(|p| p + 1)
                .unwrap_or(0);
            let ident = &result[ident_start..pos];

            // Extract the property after ?.
            let after = &result[pos + 2..];
            let prop_end = after
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '$')
                .unwrap_or(after.len());
            let prop = &after[..prop_end];

            if ident.is_empty() || prop.is_empty() {
                break;
            }

            let replacement = format!(
                "{ident} != null ? {ident}.{prop} : undefined"
            );
            result = format!(
                "{}{}{}",
                &result[..ident_start],
                replacement,
                &result[pos + 2 + prop_end..]
            );
        }
        result
    }
}

// ── Nullish Coalescing Transform ────────────────────────────────

/// Converts `a ?? b` to `a != null ? a : b`.
#[derive(Debug)]
pub struct NullishCoalescingTransform;

impl TransformRule for NullishCoalescingTransform {
    fn name(&self) -> &str {
        "nullish-coalescing"
    }

    fn apply(&self, input: &str) -> String {
        let mut result = input.to_string();
        while let Some(pos) = result.find("??") {
            let before = result[..pos].trim_end();
            let after = result[pos + 2..].trim_start();

            // Extract left operand (last identifier/expression)
            let left_start = before
                .rfind(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '$' && c != '.')
                .map(|p| p + 1)
                .unwrap_or(0);
            let left = &before[left_start..];

            // Extract right operand (next identifier/expression/literal)
            let right_end = after
                .find(|c: char| !c.is_ascii_alphanumeric() && c != '_' && c != '$' && c != '.' && c != '"' && c != '\'')
                .unwrap_or(after.len());
            let right = &after[..right_end];

            if left.is_empty() || right.is_empty() {
                break;
            }

            let replacement = format!("{left} != null ? {left} : {right}");
            let prefix = &before[..left_start];
            let suffix = &after[right_end..];
            result = format!("{prefix}{replacement}{suffix}");
        }
        result
    }
}

// ── Destructuring Transform ─────────────────────────────────────

/// Models destructuring: `const {a, b} = obj;` -> `const a = obj.a; const b = obj.b;`
#[derive(Debug)]
pub struct DestructuringTransform;

impl TransformRule for DestructuringTransform {
    fn name(&self) -> &str {
        "destructuring"
    }

    fn apply(&self, input: &str) -> String {
        let mut out = String::new();
        for line in input.lines() {
            let trimmed = line.trim();
            if let Some(transformed) = self.try_transform_object(trimmed) {
                out.push_str(&transformed);
                out.push('\n');
            } else {
                out.push_str(line);
                out.push('\n');
            }
        }
        // Remove trailing newline if input didn't have one
        if !input.ends_with('\n') && out.ends_with('\n') {
            out.pop();
        }
        out
    }
}

impl DestructuringTransform {
    fn try_transform_object(&self, line: &str) -> Option<String> {
        // Match: const { a, b } = expr;
        // or:    let { a, b } = expr;
        // or:    var { a, b } = expr;
        let keyword = if line.starts_with("const ") {
            "const"
        } else if line.starts_with("let ") {
            "let"
        } else if line.starts_with("var ") {
            "var"
        } else {
            return None;
        };

        let rest = line[keyword.len()..].trim_start();
        if !rest.starts_with('{') {
            return None;
        }

        let close = rest.find('}')?;
        let props_str = &rest[1..close];
        let after_brace = rest[close + 1..].trim_start();

        if !after_brace.starts_with('=') {
            return None;
        }
        let rhs = after_brace[1..].trim_start().trim_end_matches(';').trim();

        let props: Vec<&str> = props_str.split(',').map(|p| p.trim()).filter(|p| !p.is_empty()).collect();

        let mut lines = Vec::new();
        for prop in &props {
            // Support renaming: `a: localA`
            if let Some(colon_pos) = prop.find(':') {
                let key = prop[..colon_pos].trim();
                let local = prop[colon_pos + 1..].trim();
                lines.push(format!("{keyword} {local} = {rhs}.{key};"));
            } else {
                lines.push(format!("{keyword} {prop} = {rhs}.{prop};"));
            }
        }

        Some(lines.join("\n"))
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arrow_simple_expression() {
        let t = ArrowFunctionTransform;
        let out = t.apply("(x) => x + 1");
        assert!(out.contains("function"));
        assert!(out.contains("return"));
    }

    #[test]
    fn arrow_block_body() {
        let t = ArrowFunctionTransform;
        let out = t.apply("(x) => { return x; }");
        assert!(out.contains("function(x)"));
        assert!(out.contains("return x;"));
    }

    #[test]
    fn template_literal_basic() {
        let t = TemplateLiteralTransform;
        let out = t.apply("`hello ${name}`");
        assert!(out.contains("\"hello \""));
        assert!(out.contains("name"));
        assert!(out.contains("+"));
    }

    #[test]
    fn template_literal_no_expr() {
        let t = TemplateLiteralTransform;
        let out = t.apply("`plain text`");
        assert!(out.contains("\"plain text\""));
    }

    #[test]
    fn optional_chaining_basic() {
        let t = OptionalChainingTransform;
        let out = t.apply("obj?.prop");
        assert!(out.contains("obj != null ? obj.prop : undefined"));
    }

    #[test]
    fn nullish_coalescing_basic() {
        let t = NullishCoalescingTransform;
        let out = t.apply("val ?? fallback");
        assert!(out.contains("val != null ? val : fallback"));
    }

    #[test]
    fn destructuring_basic() {
        let t = DestructuringTransform;
        let out = t.apply("const { a, b } = obj;");
        assert!(out.contains("const a = obj.a;"));
        assert!(out.contains("const b = obj.b;"));
    }

    #[test]
    fn destructuring_rename() {
        let t = DestructuringTransform;
        let out = t.apply("const { x: localX } = obj;");
        assert!(out.contains("const localX = obj.x;"));
    }

    #[test]
    fn pipeline_chains_transforms() {
        let mut pipeline = TransformPipeline::new();
        pipeline.add_rule(Box::new(TemplateLiteralTransform));
        pipeline.add_rule(Box::new(NullishCoalescingTransform));
        assert_eq!(pipeline.rule_count(), 2);
        let out = pipeline.transform("`hello`");
        assert!(out.contains("\"hello\""));
    }

    #[test]
    fn transform_names() {
        assert_eq!(ArrowFunctionTransform.name(), "arrow-function");
        assert_eq!(TemplateLiteralTransform.name(), "template-literal");
        assert_eq!(OptionalChainingTransform.name(), "optional-chaining");
        assert_eq!(NullishCoalescingTransform.name(), "nullish-coalescing");
        assert_eq!(DestructuringTransform.name(), "destructuring");
    }

    #[test]
    fn template_multiple_expressions() {
        let t = TemplateLiteralTransform;
        let out = t.apply("`${a} and ${b}`");
        assert!(out.contains("a"));
        assert!(out.contains("b"));
        assert!(out.contains("\" and \""));
    }

    #[test]
    fn destructuring_let_var() {
        let t = DestructuringTransform;
        let out = t.apply("let { x } = src;");
        assert!(out.contains("let x = src.x;"));
    }
}
