//! Lint rule engine.
//!
//! Provides rule definitions (pattern + message + severity), rule categories,
//! lint context, auto-fix suggestions, suppression comments, rule configuration,
//! lint reports, and a rule registry. Pure Rust — no external linter deps.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt;

// ── Severity ────────────────────────────────────────────────────

/// Severity level for a lint diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Severity {
    /// Informational hint.
    Hint,
    /// Style or convention warning.
    Warning,
    /// Likely bug or error.
    Error,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Hint => write!(f, "hint"),
            Self::Warning => write!(f, "warning"),
            Self::Error => write!(f, "error"),
        }
    }
}

// ── Rule Category ───────────────────────────────────────────────

/// Category of a lint rule.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RuleCategory {
    /// Code style and formatting.
    Style,
    /// Possible bugs and logic errors.
    Correctness,
    /// Performance issues.
    Performance,
    /// Security vulnerabilities.
    Security,
    /// Code complexity.
    Complexity,
    /// Deprecated API usage.
    Deprecation,
    /// Custom category.
    Custom(String),
}

impl fmt::Display for RuleCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Style => write!(f, "style"),
            Self::Correctness => write!(f, "correctness"),
            Self::Performance => write!(f, "performance"),
            Self::Security => write!(f, "security"),
            Self::Complexity => write!(f, "complexity"),
            Self::Deprecation => write!(f, "deprecation"),
            Self::Custom(name) => write!(f, "{name}"),
        }
    }
}

// ── Rule Definition ─────────────────────────────────────────────

/// A lint rule definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintRule {
    /// Unique rule identifier (e.g., "no-unused-vars").
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Description of what the rule checks.
    pub description: String,
    /// Default severity.
    pub severity: Severity,
    /// Category.
    pub category: RuleCategory,
    /// Pattern to match (simple substring or line-based pattern).
    pub pattern: LintPattern,
    /// Suggested fix, if any.
    pub fix: Option<FixSuggestion>,
    /// Whether the rule is enabled by default.
    pub default_enabled: bool,
}

/// Pattern for matching lint violations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LintPattern {
    /// Match lines containing a substring.
    Contains(String),
    /// Match lines starting with a prefix (after trimming).
    StartsWith(String),
    /// Match lines ending with a suffix (after trimming).
    EndsWith(String),
    /// Match lines that are exact (after trimming).
    Exact(String),
    /// Match lines matching multiple patterns (all must match).
    All(Vec<LintPattern>),
    /// Match lines matching any of the patterns.
    Any(Vec<LintPattern>),
    /// Negate a pattern.
    Not(Box<LintPattern>),
    /// Match lines longer than N characters.
    LineTooLong(usize),
    /// Match lines with trailing whitespace.
    TrailingWhitespace,
    /// Match consecutive blank lines exceeding N.
    ConsecutiveBlanks(usize),
}

impl LintPattern {
    /// Check if a line matches this pattern.
    pub fn matches_line(&self, line: &str) -> bool {
        match self {
            Self::Contains(s) => line.contains(s.as_str()),
            Self::StartsWith(s) => line.trim_start().starts_with(s.as_str()),
            Self::EndsWith(s) => line.trim_end().ends_with(s.as_str()),
            Self::Exact(s) => line.trim() == s.as_str(),
            Self::All(patterns) => patterns.iter().all(|p| p.matches_line(line)),
            Self::Any(patterns) => patterns.iter().any(|p| p.matches_line(line)),
            Self::Not(inner) => !inner.matches_line(line),
            Self::LineTooLong(max) => line.len() > *max,
            Self::TrailingWhitespace => {
                !line.is_empty() && line.ends_with(|c: char| c == ' ' || c == '\t')
            }
            Self::ConsecutiveBlanks(_) => false, // handled at file level
        }
    }
}

// ── Fix Suggestion ──────────────────────────────────────────────

/// A suggested auto-fix for a lint violation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSuggestion {
    /// Description of the fix.
    pub description: String,
    /// The replacement kind.
    pub kind: FixKind,
}

/// Kind of auto-fix.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FixKind {
    /// Replace the matched text with a replacement.
    Replace { search: String, replacement: String },
    /// Remove the entire line.
    RemoveLine,
    /// Trim trailing whitespace.
    TrimTrailing,
    /// Insert text before the line.
    InsertBefore(String),
    /// Insert text after the line.
    InsertAfter(String),
}

// ── Suppression ─────────────────────────────────────────────────

/// Suppression directives parsed from comments.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Suppression {
    /// Line number where the suppression is declared.
    pub line: usize,
    /// Rule IDs to suppress (empty = suppress all).
    pub rule_ids: Vec<String>,
    /// Scope: next line, this line, or block.
    pub scope: SuppressionScope,
}

/// Scope of a suppression.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SuppressionScope {
    /// Suppress on the next line.
    NextLine,
    /// Suppress on the same line (inline).
    ThisLine,
    /// Suppress until end-of-block marker.
    Block,
}

/// Parse suppression directives from source lines.
///
/// Recognized formats:
/// - `// lint-ignore rule1, rule2` (next line)
/// - `code // lint-ignore-line rule1` (this line)
/// - `// lint-disable rule1` (block start)
/// - `// lint-enable rule1` (block end)
pub fn parse_suppressions(lines: &[&str]) -> Vec<Suppression> {
    let mut suppressions = Vec::new();

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();

        // Inline suppression: `code // lint-ignore-line rules...`
        if let Some(comment_pos) = line.find("// lint-ignore-line") {
            let after = &line[comment_pos + "// lint-ignore-line".len()..];
            let rule_ids: Vec<String> = after
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            suppressions.push(Suppression {
                line: i + 1,
                rule_ids,
                scope: SuppressionScope::ThisLine,
            });
            continue;
        }

        // Next-line suppression.
        if let Some(rest) = trimmed.strip_prefix("// lint-ignore") {
            let rule_ids: Vec<String> = rest
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            suppressions.push(Suppression {
                line: i + 1,
                rule_ids,
                scope: SuppressionScope::NextLine,
            });
            continue;
        }

        // Block suppression.
        if let Some(rest) = trimmed.strip_prefix("// lint-disable") {
            let rule_ids: Vec<String> = rest
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            suppressions.push(Suppression {
                line: i + 1,
                rule_ids,
                scope: SuppressionScope::Block,
            });
        }
    }

    suppressions
}

// ── Lint Diagnostic ─────────────────────────────────────────────

/// A single lint diagnostic (violation found).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    /// Rule ID that triggered this.
    pub rule_id: String,
    /// Severity.
    pub severity: Severity,
    /// Line number (1-based).
    pub line: usize,
    /// Column (1-based, 0 if unknown).
    pub column: usize,
    /// The message.
    pub message: String,
    /// The source line text.
    pub source_line: String,
    /// Suggested fix, if any.
    pub fix: Option<FixSuggestion>,
}

impl fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}: {} [{}] {}",
            self.line, self.column, self.severity, self.rule_id, self.message
        )
    }
}

// ── Rule Configuration ──────────────────────────────────────────

/// Per-rule configuration overrides.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuleConfig {
    /// Enabled rules (by ID). If empty, use defaults.
    pub enabled: HashSet<String>,
    /// Disabled rules (by ID).
    pub disabled: HashSet<String>,
    /// Severity overrides.
    pub severity_overrides: HashMap<String, Severity>,
}

impl Default for RuleConfig {
    fn default() -> Self {
        Self {
            enabled: HashSet::new(),
            disabled: HashSet::new(),
            severity_overrides: HashMap::new(),
        }
    }
}

impl RuleConfig {
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if a rule is enabled.
    pub fn is_enabled(&self, rule: &LintRule) -> bool {
        if self.disabled.contains(&rule.id) {
            return false;
        }
        if !self.enabled.is_empty() {
            return self.enabled.contains(&rule.id);
        }
        rule.default_enabled
    }

    /// Get the effective severity for a rule.
    pub fn effective_severity(&self, rule: &LintRule) -> Severity {
        self.severity_overrides
            .get(&rule.id)
            .copied()
            .unwrap_or(rule.severity)
    }
}

// ── Rule Registry ───────────────────────────────────────────────

/// Registry of lint rules.
#[derive(Debug, Clone, Default)]
pub struct RuleRegistry {
    rules: Vec<LintRule>,
}

impl RuleRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a rule.
    pub fn register(&mut self, rule: LintRule) {
        self.rules.push(rule);
    }

    /// Get all rules.
    pub fn rules(&self) -> &[LintRule] {
        &self.rules
    }

    /// Get rules by category.
    pub fn rules_by_category(&self, category: &RuleCategory) -> Vec<&LintRule> {
        self.rules
            .iter()
            .filter(|r| &r.category == category)
            .collect()
    }

    /// Find a rule by ID.
    pub fn find(&self, id: &str) -> Option<&LintRule> {
        self.rules.iter().find(|r| r.id == id)
    }

    /// Number of registered rules.
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Create a registry with common default rules.
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(LintRule {
            id: "trailing-whitespace".to_string(),
            name: "Trailing Whitespace".to_string(),
            description: "Lines should not have trailing whitespace".to_string(),
            severity: Severity::Warning,
            category: RuleCategory::Style,
            pattern: LintPattern::TrailingWhitespace,
            fix: Some(FixSuggestion {
                description: "Remove trailing whitespace".to_string(),
                kind: FixKind::TrimTrailing,
            }),
            default_enabled: true,
        });
        reg.register(LintRule {
            id: "line-length".to_string(),
            name: "Line Length".to_string(),
            description: "Lines should not exceed 120 characters".to_string(),
            severity: Severity::Warning,
            category: RuleCategory::Style,
            pattern: LintPattern::LineTooLong(120),
            fix: None,
            default_enabled: true,
        });
        reg.register(LintRule {
            id: "no-todo".to_string(),
            name: "No TODO Comments".to_string(),
            description: "TODO comments should be resolved".to_string(),
            severity: Severity::Hint,
            category: RuleCategory::Style,
            pattern: LintPattern::Contains("TODO".to_string()),
            fix: None,
            default_enabled: false,
        });
        reg.register(LintRule {
            id: "no-fixme".to_string(),
            name: "No FIXME Comments".to_string(),
            description: "FIXME comments indicate known bugs".to_string(),
            severity: Severity::Warning,
            category: RuleCategory::Correctness,
            pattern: LintPattern::Contains("FIXME".to_string()),
            fix: None,
            default_enabled: true,
        });
        reg.register(LintRule {
            id: "no-debug-print".to_string(),
            name: "No Debug Prints".to_string(),
            description: "Debug print statements should be removed".to_string(),
            severity: Severity::Warning,
            category: RuleCategory::Correctness,
            pattern: LintPattern::Any(vec![
                LintPattern::Contains("dbg!".to_string()),
                LintPattern::Contains("println!".to_string()),
            ]),
            fix: Some(FixSuggestion {
                description: "Remove debug print".to_string(),
                kind: FixKind::RemoveLine,
            }),
            default_enabled: false,
        });
        reg
    }
}

// ── Lint Context ────────────────────────────────────────────────

/// Context passed to lint checks, containing file information.
#[derive(Debug, Clone)]
pub struct LintContext {
    pub file_path: String,
    pub lines: Vec<String>,
    pub suppressions: Vec<Suppression>,
}

impl LintContext {
    /// Create from source text.
    pub fn from_source(file_path: &str, source: &str) -> Self {
        let lines: Vec<String> = source.lines().map(|l| l.to_string()).collect();
        let line_refs: Vec<&str> = source.lines().collect();
        let suppressions = parse_suppressions(&line_refs);
        Self {
            file_path: file_path.to_string(),
            lines,
            suppressions,
        }
    }

    /// Check if a rule is suppressed at a given line.
    pub fn is_suppressed(&self, rule_id: &str, line: usize) -> bool {
        for sup in &self.suppressions {
            let matches_rule =
                sup.rule_ids.is_empty() || sup.rule_ids.iter().any(|id| id == rule_id);
            if !matches_rule {
                continue;
            }
            match sup.scope {
                SuppressionScope::NextLine => {
                    if line == sup.line + 1 {
                        return true;
                    }
                }
                SuppressionScope::ThisLine => {
                    if line == sup.line {
                        return true;
                    }
                }
                SuppressionScope::Block => {
                    if line >= sup.line {
                        return true;
                    }
                }
            }
        }
        false
    }
}

// ── Lint Engine ─────────────────────────────────────────────────

/// The main lint engine.
pub struct LintEngine {
    pub registry: RuleRegistry,
    pub config: RuleConfig,
}

impl LintEngine {
    /// Create with a registry and config.
    pub fn new(registry: RuleRegistry, config: RuleConfig) -> Self {
        Self { registry, config }
    }

    /// Create with default rules and config.
    pub fn with_defaults() -> Self {
        Self {
            registry: RuleRegistry::with_defaults(),
            config: RuleConfig::default(),
        }
    }

    /// Lint a source file and return diagnostics.
    pub fn lint(&self, ctx: &LintContext) -> Vec<Diagnostic> {
        let mut diagnostics = Vec::new();

        for rule in self.registry.rules() {
            if !self.config.is_enabled(rule) {
                continue;
            }

            let severity = self.config.effective_severity(rule);

            // Check for consecutive blank lines pattern (file-level check).
            if let LintPattern::ConsecutiveBlanks(max) = &rule.pattern {
                let mut consecutive = 0usize;
                for (i, line) in ctx.lines.iter().enumerate() {
                    let line_num = i + 1;
                    if line.trim().is_empty() {
                        consecutive += 1;
                        if consecutive > *max && !ctx.is_suppressed(&rule.id, line_num) {
                            diagnostics.push(Diagnostic {
                                rule_id: rule.id.clone(),
                                severity,
                                line: line_num,
                                column: 0,
                                message: format!(
                                    "{}: {} consecutive blank lines (max {})",
                                    rule.description, consecutive, max
                                ),
                                source_line: String::new(),
                                fix: rule.fix.clone(),
                            });
                        }
                    } else {
                        consecutive = 0;
                    }
                }
                continue;
            }

            // Line-level checks.
            for (i, line) in ctx.lines.iter().enumerate() {
                let line_num = i + 1;
                if ctx.is_suppressed(&rule.id, line_num) {
                    continue;
                }
                if rule.pattern.matches_line(line) {
                    diagnostics.push(Diagnostic {
                        rule_id: rule.id.clone(),
                        severity,
                        line: line_num,
                        column: 1,
                        message: rule.description.clone(),
                        source_line: line.clone(),
                        fix: rule.fix.clone(),
                    });
                }
            }
        }

        diagnostics
    }

    /// Apply auto-fixes and return the modified source.
    pub fn auto_fix(&self, source: &str, diagnostics: &[Diagnostic]) -> String {
        let mut lines: Vec<String> = source.lines().map(|l| l.to_string()).collect();

        // Process fixes in reverse order to preserve line numbers.
        let mut fixable: Vec<&Diagnostic> = diagnostics.iter().filter(|d| d.fix.is_some()).collect();
        fixable.sort_by(|a, b| b.line.cmp(&a.line));

        let mut removed_lines: HashSet<usize> = HashSet::new();

        for diag in fixable {
            if let Some(fix) = &diag.fix {
                let idx = diag.line.saturating_sub(1);
                if idx >= lines.len() {
                    continue;
                }
                match &fix.kind {
                    FixKind::Replace {
                        search,
                        replacement,
                    } => {
                        lines[idx] = lines[idx].replace(search.as_str(), replacement);
                    }
                    FixKind::RemoveLine => {
                        removed_lines.insert(idx);
                    }
                    FixKind::TrimTrailing => {
                        lines[idx] = lines[idx].trim_end().to_string();
                    }
                    FixKind::InsertBefore(text) => {
                        lines.insert(idx, text.clone());
                    }
                    FixKind::InsertAfter(text) => {
                        let insert_at = (idx + 1).min(lines.len());
                        lines.insert(insert_at, text.clone());
                    }
                }
            }
        }

        // Remove marked lines (in reverse to preserve indices).
        let mut sorted_removed: Vec<usize> = removed_lines.into_iter().collect();
        sorted_removed.sort_unstable();
        for idx in sorted_removed.into_iter().rev() {
            if idx < lines.len() {
                lines.remove(idx);
            }
        }

        lines.join("\n")
    }
}

// ── Lint Report ─────────────────────────────────────────────────

/// Summary report of lint results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintReport {
    pub file_path: String,
    pub total_diagnostics: usize,
    pub errors: usize,
    pub warnings: usize,
    pub hints: usize,
    pub fixable: usize,
}

impl LintReport {
    /// Build a report from diagnostics.
    pub fn from_diagnostics(file_path: &str, diagnostics: &[Diagnostic]) -> Self {
        let errors = diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .count();
        let warnings = diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count();
        let hints = diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Hint)
            .count();
        let fixable = diagnostics.iter().filter(|d| d.fix.is_some()).count();

        Self {
            file_path: file_path.to_string(),
            total_diagnostics: diagnostics.len(),
            errors,
            warnings,
            hints,
            fixable,
        }
    }
}

impl fmt::Display for LintReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Lint report for: {}", self.file_path)?;
        writeln!(
            f,
            "  {} errors, {} warnings, {} hints ({} fixable)",
            self.errors, self.warnings, self.hints, self.fixable
        )?;
        writeln!(f, "  Total: {} diagnostics", self.total_diagnostics)
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_engine() -> LintEngine {
        LintEngine::with_defaults()
    }

    #[test]
    fn lint_trailing_whitespace() {
        let engine = make_engine();
        let ctx = LintContext::from_source("test.rs", "let x = 1;  \nlet y = 2;\n");
        let diags = engine.lint(&ctx);
        assert!(diags.iter().any(|d| d.rule_id == "trailing-whitespace"));
    }

    #[test]
    fn lint_long_line() {
        let engine = make_engine();
        let long_line = "x".repeat(130);
        let ctx = LintContext::from_source("test.rs", &long_line);
        let diags = engine.lint(&ctx);
        assert!(diags.iter().any(|d| d.rule_id == "line-length"));
    }

    #[test]
    fn lint_fixme() {
        let engine = make_engine();
        let ctx = LintContext::from_source("test.rs", "// FIXME: broken\nlet x = 1;\n");
        let diags = engine.lint(&ctx);
        assert!(diags.iter().any(|d| d.rule_id == "no-fixme"));
    }

    #[test]
    fn lint_no_false_positive() {
        let engine = make_engine();
        let ctx = LintContext::from_source("test.rs", "let x = 1;\nlet y = 2;\n");
        let diags = engine.lint(&ctx);
        // No trailing whitespace, no FIXME, no long lines.
        assert!(diags.is_empty());
    }

    #[test]
    fn lint_disabled_rule() {
        let engine = make_engine();
        // no-todo is disabled by default.
        let ctx = LintContext::from_source("test.rs", "// TODO: fix this\n");
        let diags = engine.lint(&ctx);
        assert!(!diags.iter().any(|d| d.rule_id == "no-todo"));
    }

    #[test]
    fn lint_config_enable_rule() {
        let mut config = RuleConfig::new();
        config.enabled.insert("no-todo".to_string());
        let engine = LintEngine::new(RuleRegistry::with_defaults(), config);
        let ctx = LintContext::from_source("test.rs", "// TODO: fix this\n");
        let diags = engine.lint(&ctx);
        assert!(diags.iter().any(|d| d.rule_id == "no-todo"));
    }

    #[test]
    fn lint_config_disable_rule() {
        let mut config = RuleConfig::new();
        config.disabled.insert("no-fixme".to_string());
        let engine = LintEngine::new(RuleRegistry::with_defaults(), config);
        let ctx = LintContext::from_source("test.rs", "// FIXME: broken\n");
        let diags = engine.lint(&ctx);
        assert!(!diags.iter().any(|d| d.rule_id == "no-fixme"));
    }

    #[test]
    fn lint_severity_override() {
        let mut config = RuleConfig::new();
        config
            .severity_overrides
            .insert("no-fixme".to_string(), Severity::Error);
        let engine = LintEngine::new(RuleRegistry::with_defaults(), config);
        let ctx = LintContext::from_source("test.rs", "// FIXME: broken\n");
        let diags = engine.lint(&ctx);
        let fixme_diag = diags.iter().find(|d| d.rule_id == "no-fixme").unwrap();
        assert_eq!(fixme_diag.severity, Severity::Error);
    }

    #[test]
    fn lint_suppression_next_line() {
        let engine = make_engine();
        let source = "// lint-ignore trailing-whitespace\nlet x = 1;  \n";
        let ctx = LintContext::from_source("test.rs", source);
        let diags = engine.lint(&ctx);
        // The trailing whitespace on line 2 should be suppressed.
        assert!(!diags.iter().any(|d| d.rule_id == "trailing-whitespace"));
    }

    #[test]
    fn lint_suppression_inline() {
        let engine = make_engine();
        let source = "let x = 1;   // lint-ignore-line trailing-whitespace\n";
        let ctx = LintContext::from_source("test.rs", source);
        let diags = engine.lint(&ctx);
        assert!(
            !diags
                .iter()
                .any(|d| d.rule_id == "trailing-whitespace" && d.line == 1)
        );
    }

    #[test]
    fn auto_fix_trailing_whitespace() {
        let engine = make_engine();
        let source = "let x = 1;  \nlet y = 2;  \n";
        let ctx = LintContext::from_source("test.rs", source);
        let diags = engine.lint(&ctx);
        let fixed = engine.auto_fix(source, &diags);
        // No trailing whitespace after fix.
        for line in fixed.lines() {
            assert!(!line.ends_with(' '), "line still has trailing space: {line:?}");
        }
    }

    #[test]
    fn pattern_contains() {
        let p = LintPattern::Contains("unsafe".to_string());
        assert!(p.matches_line("    unsafe { ptr::read(x) }"));
        assert!(!p.matches_line("    safe code here"));
    }

    #[test]
    fn pattern_starts_with() {
        let p = LintPattern::StartsWith("//".to_string());
        assert!(p.matches_line("  // comment"));
        assert!(!p.matches_line("  let x = 1; // inline"));
    }

    #[test]
    fn pattern_not() {
        let p = LintPattern::Not(Box::new(LintPattern::Contains("ok".to_string())));
        assert!(p.matches_line("bad line"));
        assert!(!p.matches_line("this is ok"));
    }

    #[test]
    fn pattern_all() {
        let p = LintPattern::All(vec![
            LintPattern::Contains("fn".to_string()),
            LintPattern::Contains("unsafe".to_string()),
        ]);
        assert!(p.matches_line("unsafe fn foo()"));
        assert!(!p.matches_line("fn foo()"));
    }

    #[test]
    fn pattern_any() {
        let p = LintPattern::Any(vec![
            LintPattern::Contains("TODO".to_string()),
            LintPattern::Contains("FIXME".to_string()),
        ]);
        assert!(p.matches_line("// TODO fix"));
        assert!(p.matches_line("// FIXME broken"));
        assert!(!p.matches_line("// normal comment"));
    }

    #[test]
    fn rule_registry_find() {
        let reg = RuleRegistry::with_defaults();
        assert!(reg.find("trailing-whitespace").is_some());
        assert!(reg.find("nonexistent").is_none());
    }

    #[test]
    fn rule_registry_by_category() {
        let reg = RuleRegistry::with_defaults();
        let style_rules = reg.rules_by_category(&RuleCategory::Style);
        assert!(style_rules.len() >= 2); // trailing-whitespace and line-length
    }

    #[test]
    fn lint_report() {
        let engine = make_engine();
        let source = "let x = 1;  \n// FIXME: broken\n";
        let ctx = LintContext::from_source("test.rs", source);
        let diags = engine.lint(&ctx);
        let report = LintReport::from_diagnostics("test.rs", &diags);
        assert!(report.total_diagnostics >= 2);
        assert!(report.warnings >= 2);
        assert!(report.fixable >= 1);
    }

    #[test]
    fn diagnostic_display() {
        let d = Diagnostic {
            rule_id: "test-rule".to_string(),
            severity: Severity::Error,
            line: 10,
            column: 5,
            message: "bad code".to_string(),
            source_line: "let x = bad;".to_string(),
            fix: None,
        };
        let s = format!("{d}");
        assert!(s.contains("10:5"));
        assert!(s.contains("error"));
        assert!(s.contains("test-rule"));
    }

    #[test]
    fn severity_ordering() {
        assert!(Severity::Hint < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn lint_report_display() {
        let report = LintReport {
            file_path: "test.rs".to_string(),
            total_diagnostics: 5,
            errors: 1,
            warnings: 3,
            hints: 1,
            fixable: 2,
        };
        let s = format!("{report}");
        assert!(s.contains("1 errors"));
        assert!(s.contains("3 warnings"));
    }

    #[test]
    fn custom_rule() {
        let mut reg = RuleRegistry::new();
        reg.register(LintRule {
            id: "no-panic".to_string(),
            name: "No Panic".to_string(),
            description: "Avoid panic! in production code".to_string(),
            severity: Severity::Error,
            category: RuleCategory::Correctness,
            pattern: LintPattern::Contains("panic!".to_string()),
            fix: None,
            default_enabled: true,
        });
        let engine = LintEngine::new(reg, RuleConfig::default());
        let ctx = LintContext::from_source("test.rs", "    panic!(\"oh no\");\n");
        let diags = engine.lint(&ctx);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].rule_id, "no-panic");
    }

    #[test]
    fn parse_suppressions_multiple() {
        let lines = vec![
            "// lint-ignore trailing-whitespace, line-length",
            "let x = 1;",
            "code // lint-ignore-line no-fixme",
        ];
        let sups = parse_suppressions(&lines);
        assert_eq!(sups.len(), 2);
        assert_eq!(sups[0].scope, SuppressionScope::NextLine);
        assert_eq!(sups[1].scope, SuppressionScope::ThisLine);
    }

    #[test]
    fn category_display() {
        assert_eq!(format!("{}", RuleCategory::Style), "style");
        assert_eq!(
            format!("{}", RuleCategory::Custom("my-cat".to_string())),
            "my-cat"
        );
    }
}
