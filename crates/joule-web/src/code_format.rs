//! Code formatter framework.
//!
//! Provides composable formatting rules applied sequentially: indentation
//! normalization, trailing whitespace removal, line length enforcement,
//! bracket alignment, and import sorting. Replaces Prettier/rustfmt concepts.

// ── Format config ───────────────────────────────────────────────

/// Configuration for the code formatter.
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Number of spaces per indent level.
    pub indent_size: usize,
    /// Maximum line length before wrapping.
    pub max_line_length: usize,
    /// Use tabs instead of spaces.
    pub use_tabs: bool,
    /// Insert a final newline at end of file.
    pub insert_final_newline: bool,
    /// Trim trailing whitespace.
    pub trim_trailing_whitespace: bool,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            indent_size: 4,
            max_line_length: 100,
            use_tabs: false,
            insert_final_newline: true,
            trim_trailing_whitespace: true,
        }
    }
}

// ── Format rule trait ───────────────────────────────────────────

/// A formatting rule that transforms source text.
pub trait FormatRule {
    /// Apply this rule to the text, returning the formatted result.
    fn apply(&self, text: &str, config: &FormatConfig) -> String;

    /// Name of this rule for diagnostics.
    fn name(&self) -> &str;
}

// ── IndentationNormalizer ───────────────────────────────────────

/// Normalizes indentation: converts tabs to spaces (or vice versa)
/// and ensures consistent indent width.
pub struct IndentationNormalizer;

impl IndentationNormalizer {
    /// Count the indentation level of a line.
    fn indent_level(line: &str, config: &FormatConfig) -> usize {
        let mut spaces = 0usize;
        for ch in line.chars() {
            match ch {
                ' ' => spaces += 1,
                '\t' => spaces += config.indent_size,
                _ => break,
            }
        }
        spaces / config.indent_size
    }
}

impl FormatRule for IndentationNormalizer {
    fn apply(&self, text: &str, config: &FormatConfig) -> String {
        let mut result = Vec::new();
        for line in text.lines() {
            let level = Self::indent_level(line, config);
            let content = line.trim_start();
            if content.is_empty() {
                result.push(String::new());
            } else {
                let indent = if config.use_tabs {
                    "\t".repeat(level)
                } else {
                    " ".repeat(level * config.indent_size)
                };
                result.push(format!("{indent}{content}"));
            }
        }
        result.join("\n")
    }

    fn name(&self) -> &str {
        "IndentationNormalizer"
    }
}

// ── TrailingWhitespaceRemover ───────────────────────────────────

/// Removes trailing whitespace from all lines.
pub struct TrailingWhitespaceRemover;

impl FormatRule for TrailingWhitespaceRemover {
    fn apply(&self, text: &str, _config: &FormatConfig) -> String {
        text.lines()
            .map(|line| line.trim_end())
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn name(&self) -> &str {
        "TrailingWhitespaceRemover"
    }
}

// ── LineLengthEnforcer ──────────────────────────────────────────

/// Breaks lines that exceed the maximum width at reasonable points.
pub struct LineLengthEnforcer;

impl LineLengthEnforcer {
    /// Find a good break point near the max length.
    fn find_break_point(line: &str, max: usize) -> Option<usize> {
        if line.len() <= max {
            return None;
        }

        // Try to break at a space near the max length
        let search_start = max.saturating_sub(20);
        let search_end = max.min(line.len());

        // Look backward from max for a space
        for i in (search_start..search_end).rev() {
            if line.as_bytes().get(i) == Some(&b' ') {
                return Some(i);
            }
        }

        // Look for comma or other punctuation
        for i in (search_start..search_end).rev() {
            let c = line.as_bytes().get(i).copied().unwrap_or(0);
            if c == b',' || c == b';' {
                return Some(i + 1);
            }
        }

        // Forced break at max
        Some(max)
    }
}

impl FormatRule for LineLengthEnforcer {
    fn apply(&self, text: &str, config: &FormatConfig) -> String {
        let mut result = Vec::new();

        for line in text.lines() {
            if line.len() <= config.max_line_length {
                result.push(line.to_string());
                continue;
            }

            // Determine indent of continuation lines
            let indent_len = line.len() - line.trim_start().len();
            let continuation_indent = format!(
                "{}{}",
                &line[..indent_len],
                if config.use_tabs {
                    "\t".to_string()
                } else {
                    " ".repeat(config.indent_size)
                }
            );

            let mut remaining = line.to_string();
            let mut first = true;

            while remaining.len() > config.max_line_length {
                if let Some(bp) = Self::find_break_point(&remaining, config.max_line_length) {
                    let (head, tail) = remaining.split_at(bp);
                    result.push(head.to_string());
                    remaining = format!("{continuation_indent}{}", tail.trim_start());
                    first = false;
                } else {
                    break;
                }
            }
            let _ = first;
            result.push(remaining);
        }

        result.join("\n")
    }

    fn name(&self) -> &str {
        "LineLengthEnforcer"
    }
}

// ── BracketAligner ──────────────────────────────────────────────

/// Ensures closing brackets are aligned with their opening line's indentation.
pub struct BracketAligner;

impl FormatRule for BracketAligner {
    fn apply(&self, text: &str, config: &FormatConfig) -> String {
        let mut result = Vec::new();
        let mut indent_stack: Vec<usize> = vec![0];

        for line in text.lines() {
            let trimmed = line.trim();

            // Closing bracket: pop indent level
            if trimmed.starts_with('}')
                || trimmed.starts_with(')')
                || trimmed.starts_with(']')
            {
                indent_stack.pop();
            }

            let level = *indent_stack.last().unwrap_or(&0);
            let indent = if config.use_tabs {
                "\t".repeat(level)
            } else {
                " ".repeat(level * config.indent_size)
            };

            if trimmed.is_empty() {
                result.push(String::new());
            } else {
                result.push(format!("{indent}{trimmed}"));
            }

            // Opening bracket: push indent level
            if trimmed.ends_with('{')
                || trimmed.ends_with('(')
                || trimmed.ends_with('[')
            {
                indent_stack.push(level + 1);
            }
        }

        result.join("\n")
    }

    fn name(&self) -> &str {
        "BracketAligner"
    }
}

// ── SortImports ─────────────────────────────────────────────────

/// Sorts import/use statements alphabetically, grouped by prefix.
pub struct SortImports {
    /// Keywords that identify import lines (e.g., "use", "import").
    pub import_keywords: Vec<String>,
}

impl SortImports {
    pub fn new(keywords: &[&str]) -> Self {
        Self {
            import_keywords: keywords.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Default Rust imports.
    pub fn rust() -> Self {
        Self::new(&["use"])
    }

    /// Default JS/TS imports.
    pub fn javascript() -> Self {
        Self::new(&["import"])
    }

    fn is_import_line(&self, line: &str) -> bool {
        let trimmed = line.trim();
        self.import_keywords
            .iter()
            .any(|kw| trimmed.starts_with(kw.as_str()))
    }

    /// Extract the import prefix group (e.g., "std" from "use std::collections").
    fn import_group(line: &str) -> String {
        let trimmed = line.trim();
        // After the keyword, get the first path segment
        if let Some(rest) = trimmed.split_whitespace().nth(1) {
            let path = rest.trim_end_matches(';');
            if let Some(group) = path.split("::").next() {
                return group.to_string();
            }
            return path.to_string();
        }
        String::new()
    }
}

impl FormatRule for SortImports {
    fn apply(&self, text: &str, _config: &FormatConfig) -> String {
        let lines: Vec<&str> = text.lines().collect();
        let mut result: Vec<String> = Vec::new();
        let mut i = 0;

        while i < lines.len() {
            if self.is_import_line(lines[i]) {
                // Collect the block of consecutive imports
                let start = i;
                while i < lines.len()
                    && (self.is_import_line(lines[i]) || lines[i].trim().is_empty())
                {
                    i += 1;
                }

                let import_lines: Vec<&str> = lines[start..i]
                    .iter()
                    .copied()
                    .filter(|l| self.is_import_line(l))
                    .collect();

                // Group by prefix
                let mut groups: std::collections::BTreeMap<String, Vec<String>> =
                    std::collections::BTreeMap::new();
                for line in &import_lines {
                    let group = Self::import_group(line);
                    groups.entry(group).or_default().push(line.to_string());
                }

                // Sort within each group and emit
                let mut first_group = true;
                for (_group, mut members) in groups {
                    if !first_group {
                        result.push(String::new());
                    }
                    members.sort();
                    result.extend(members);
                    first_group = false;
                }
            } else {
                result.push(lines[i].to_string());
                i += 1;
            }
        }

        result.join("\n")
    }

    fn name(&self) -> &str {
        "SortImports"
    }
}

// ── FinalNewlineInserter ────────────────────────────────────────

/// Ensures the file ends with exactly one newline.
pub struct FinalNewlineInserter;

impl FormatRule for FinalNewlineInserter {
    fn apply(&self, text: &str, config: &FormatConfig) -> String {
        if config.insert_final_newline {
            let trimmed = text.trim_end_matches('\n');
            format!("{trimmed}\n")
        } else {
            text.trim_end_matches('\n').to_string()
        }
    }

    fn name(&self) -> &str {
        "FinalNewlineInserter"
    }
}

// ── Formatter ───────────────────────────────────────────────────

/// A formatter that applies rules sequentially.
pub struct Formatter {
    rules: Vec<Box<dyn FormatRule>>,
    pub config: FormatConfig,
}

impl Formatter {
    /// Create a new formatter with the given config.
    pub fn new(config: FormatConfig) -> Self {
        Self {
            rules: Vec::new(),
            config,
        }
    }

    /// Create a formatter with default rules.
    pub fn with_defaults(config: FormatConfig) -> Self {
        let mut f = Self::new(config);
        f.add_rule(Box::new(TrailingWhitespaceRemover));
        f.add_rule(Box::new(IndentationNormalizer));
        f.add_rule(Box::new(FinalNewlineInserter));
        f
    }

    /// Add a formatting rule.
    pub fn add_rule(&mut self, rule: Box<dyn FormatRule>) {
        self.rules.push(rule);
    }

    /// Format the given text by applying all rules in order.
    pub fn format(&self, text: &str) -> String {
        let mut result = text.to_string();
        for rule in &self.rules {
            result = rule.apply(&result, &self.config);
        }
        result
    }

    /// Number of rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trailing_whitespace_removal() {
        let rule = TrailingWhitespaceRemover;
        let config = FormatConfig::default();
        let result = rule.apply("hello   \nworld  \n  ", &config);
        assert_eq!(result, "hello\nworld\n");
    }

    #[test]
    fn indentation_normalizer_tabs_to_spaces() {
        let rule = IndentationNormalizer;
        let config = FormatConfig {
            indent_size: 4,
            use_tabs: false,
            ..Default::default()
        };
        let result = rule.apply("\thello\n\t\tworld", &config);
        assert_eq!(result, "    hello\n        world");
    }

    #[test]
    fn indentation_normalizer_spaces_to_tabs() {
        let rule = IndentationNormalizer;
        let config = FormatConfig {
            indent_size: 4,
            use_tabs: true,
            ..Default::default()
        };
        let result = rule.apply("    hello\n        world", &config);
        assert_eq!(result, "\thello\n\t\tworld");
    }

    #[test]
    fn line_length_enforcer() {
        let rule = LineLengthEnforcer;
        let config = FormatConfig {
            max_line_length: 20,
            ..Default::default()
        };
        let input = "this is a very long line that should be broken up";
        let result = rule.apply(input, &config);
        for line in result.lines() {
            assert!(
                line.len() <= 30,
                "line too long ({} chars): {line}",
                line.len()
            );
        }
    }

    #[test]
    fn bracket_aligner() {
        let rule = BracketAligner;
        let config = FormatConfig::default();
        let input = "fn main() {\nlet x = 1;\n}";
        let result = rule.apply(input, &config);
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines[0], "fn main() {");
        assert!(lines[1].starts_with("    "));
        assert_eq!(lines[2], "}");
    }

    #[test]
    fn sort_imports_rust() {
        let rule = SortImports::rust();
        let config = FormatConfig::default();
        let input =
            "use std::io;\nuse std::collections::HashMap;\nuse crate::foo;\n\nfn main() {}";
        let result = rule.apply(input, &config);
        let lines: Vec<&str> = result.lines().collect();
        let crate_idx = lines.iter().position(|l| l.contains("crate::foo")).unwrap();
        let std_idx = lines
            .iter()
            .position(|l| l.contains("std::collections"))
            .unwrap();
        assert!(crate_idx < std_idx);
    }

    #[test]
    fn sort_imports_alphabetical_within_group() {
        let rule = SortImports::rust();
        let config = FormatConfig::default();
        let input = "use std::io;\nuse std::collections;\nuse std::fmt;";
        let result = rule.apply(input, &config);
        let lines: Vec<&str> = result.lines().collect();
        assert!(lines[0].contains("collections"));
        assert!(lines[1].contains("fmt"));
        assert!(lines[2].contains("io"));
    }

    #[test]
    fn final_newline_inserted() {
        let rule = FinalNewlineInserter;
        let config = FormatConfig {
            insert_final_newline: true,
            ..Default::default()
        };
        assert_eq!(rule.apply("hello", &config), "hello\n");
        assert_eq!(rule.apply("hello\n\n\n", &config), "hello\n");
    }

    #[test]
    fn final_newline_not_inserted() {
        let rule = FinalNewlineInserter;
        let config = FormatConfig {
            insert_final_newline: false,
            ..Default::default()
        };
        assert_eq!(rule.apply("hello\n", &config), "hello");
    }

    #[test]
    fn formatter_applies_rules_sequentially() {
        let config = FormatConfig::default();
        let formatter = Formatter::with_defaults(config);
        let input = "  hello   \n\tworld  ";
        let result = formatter.format(input);
        assert!(result.ends_with('\n'));
        for line in result.trim_end().lines() {
            assert_eq!(line, line.trim_end(), "trailing whitespace found");
        }
    }

    #[test]
    fn formatter_rule_count() {
        let formatter = Formatter::with_defaults(FormatConfig::default());
        assert_eq!(formatter.rule_count(), 3);
    }

    #[test]
    fn bracket_aligner_nested() {
        let rule = BracketAligner;
        let config = FormatConfig {
            indent_size: 2,
            ..Default::default()
        };
        let input = "fn foo() {\nif true {\nx();\n}\n}";
        let result = rule.apply(input, &config);
        let lines: Vec<&str> = result.lines().collect();
        assert_eq!(lines[0], "fn foo() {");
        assert_eq!(lines[1], "  if true {");
        assert_eq!(lines[2], "    x();");
        assert_eq!(lines[3], "  }");
        assert_eq!(lines[4], "}");
    }

    #[test]
    fn empty_input() {
        let formatter = Formatter::with_defaults(FormatConfig::default());
        let result = formatter.format("");
        assert_eq!(result, "\n");
    }

    #[test]
    fn sort_imports_preserves_non_imports() {
        let rule = SortImports::rust();
        let config = FormatConfig::default();
        let input = "// comment\nuse std::io;\n\nfn main() {}";
        let result = rule.apply(input, &config);
        assert!(result.contains("// comment"));
        assert!(result.contains("fn main()"));
    }
}
