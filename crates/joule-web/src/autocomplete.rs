//! Completion engine for code and text editors.
//!
//! Provides filter-as-you-type completion with fuzzy matching, multi-source
//! merging, and snippet support. Replaces Monaco/CodeMirror autocomplete.

use std::collections::HashMap;

// ── Completion item ─────────────────────────────────────────────

/// The kind of a completion item.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CompletionKind {
    Text,
    Method,
    Function,
    Constructor,
    Field,
    Variable,
    Class,
    Interface,
    Module,
    Property,
    Unit,
    Value,
    Enum,
    Keyword,
    Snippet,
    Color,
    File,
    Reference,
    Folder,
    Constant,
    Struct,
    Event,
    Operator,
    TypeParameter,
}

/// A single completion item.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompletionItem {
    /// The label shown in the completion list.
    pub label: String,
    /// Optional detail text (e.g., type signature).
    pub detail: Option<String>,
    /// The kind of completion.
    pub kind: CompletionKind,
    /// Text to insert when accepted (defaults to label if None).
    pub insert_text: Option<String>,
    /// Sort key for ordering (defaults to label if None).
    pub sort_key: Option<String>,
    /// Priority from the source (lower = higher priority).
    pub source_priority: u32,
}

impl CompletionItem {
    /// Create a simple completion item.
    pub fn new(label: &str, kind: CompletionKind) -> Self {
        Self {
            label: label.to_string(),
            detail: None,
            kind,
            insert_text: None,
            sort_key: None,
            source_priority: 100,
        }
    }

    /// Set the detail.
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = Some(detail.to_string());
        self
    }

    /// Set the insert text.
    pub fn with_insert_text(mut self, text: &str) -> Self {
        self.insert_text = Some(text.to_string());
        self
    }

    /// Set the sort key.
    pub fn with_sort_key(mut self, key: &str) -> Self {
        self.sort_key = Some(key.to_string());
        self
    }

    /// Set the source priority.
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.source_priority = priority;
        self
    }

    /// Get the effective sort key.
    fn effective_sort_key(&self) -> &str {
        self.sort_key
            .as_deref()
            .unwrap_or(&self.label)
    }

    /// Get the effective insert text.
    pub fn effective_insert_text(&self) -> &str {
        self.insert_text
            .as_deref()
            .unwrap_or(&self.label)
    }
}

// ── Fuzzy matching ──────────────────────────────────────────────

/// Check if `query` is a prefix of `target` (case-insensitive).
fn prefix_match(query: &str, target: &str) -> bool {
    let q = query.to_lowercase();
    let t = target.to_lowercase();
    t.starts_with(&q)
}

/// Fuzzy match: all characters of `query` appear in order in `target`.
/// Returns a score (lower is better) or None if no match.
pub fn fuzzy_match(query: &str, target: &str) -> Option<u32> {
    if query.is_empty() {
        return Some(0);
    }

    let query_lower: Vec<char> = query.chars().map(|c| c.to_lowercase().next().unwrap_or(c)).collect();
    let target_lower: Vec<char> = target.chars().map(|c| c.to_lowercase().next().unwrap_or(c)).collect();
    let target_chars: Vec<char> = target.chars().collect();

    let mut qi = 0;
    let mut score: u32 = 0;
    let mut last_match_idx: Option<usize> = None;

    for (ti, tc) in target_lower.iter().enumerate() {
        if qi < query_lower.len() && *tc == query_lower[qi] {
            // Bonus for prefix match
            if qi == 0 && ti == 0 {
                // exact prefix: no penalty
            } else if let Some(last) = last_match_idx {
                // Penalty for gaps
                let gap = ti - last - 1;
                score += gap as u32;
            } else {
                score += ti as u32;
            }

            // Bonus for case match
            if target_chars[ti] != query.chars().nth(qi).unwrap_or(' ') {
                score += 1; // case mismatch penalty
            }

            last_match_idx = Some(ti);
            qi += 1;
        }
    }

    if qi == query_lower.len() {
        Some(score)
    } else {
        None
    }
}

// ── Completion context ──────────────────────────────────────────

/// Context passed to completion sources.
#[derive(Debug, Clone)]
pub struct CompletionContext {
    /// The text of the current line.
    pub line_text: String,
    /// Column position of the cursor.
    pub column: usize,
    /// Line number (0-based).
    pub line_number: usize,
    /// The trigger character (if completion was triggered by one).
    pub trigger_char: Option<char>,
    /// The word prefix being typed (text from word start to cursor).
    pub prefix: String,
}

impl CompletionContext {
    /// Create a context from a line and cursor position.
    pub fn from_line(line_text: &str, column: usize, line_number: usize) -> Self {
        let col = column.min(line_text.len());
        let prefix_start = line_text[..col]
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + 1)
            .unwrap_or(0);
        let prefix = line_text[prefix_start..col].to_string();

        Self {
            line_text: line_text.to_string(),
            column: col,
            line_number,
            trigger_char: if col > 0 {
                line_text.chars().nth(col - 1)
            } else {
                None
            },
            prefix,
        }
    }
}

// ── Completion source ───────────────────────────────────────────

/// A source of completions.
pub trait CompletionSource {
    /// Provide completions for the given context.
    fn completions(&self, ctx: &CompletionContext) -> Vec<CompletionItem>;

    /// Priority of this source (lower = higher priority).
    fn priority(&self) -> u32 {
        100
    }
}

/// A simple word-list completion source.
#[derive(Debug, Clone)]
pub struct WordSource {
    pub words: Vec<CompletionItem>,
    pub priority: u32,
}

impl WordSource {
    pub fn new(words: Vec<CompletionItem>, priority: u32) -> Self {
        Self { words, priority }
    }

    pub fn from_strings(words: &[&str], kind: CompletionKind, priority: u32) -> Self {
        let items = words
            .iter()
            .map(|w| CompletionItem::new(w, kind).with_priority(priority))
            .collect();
        Self {
            words: items,
            priority,
        }
    }
}

impl CompletionSource for WordSource {
    fn completions(&self, _ctx: &CompletionContext) -> Vec<CompletionItem> {
        self.words.clone()
    }

    fn priority(&self) -> u32 {
        self.priority
    }
}

// ── Snippet parsing ─────────────────────────────────────────────

/// A parsed snippet with tab stops.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Snippet {
    /// The parts of the snippet.
    pub parts: Vec<SnippetPart>,
}

/// A part of a snippet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SnippetPart {
    /// Plain text.
    Text(String),
    /// A tab stop with index (0 = final position).
    TabStop(u32),
    /// A tab stop with a placeholder.
    Placeholder { index: u32, text: String },
}

impl Snippet {
    /// Parse a snippet string (e.g., "fn ${1:name}($2) {\n\t$0\n}").
    pub fn parse(input: &str) -> Self {
        let mut parts = Vec::new();
        let mut chars = input.chars().peekable();
        let mut text_buf = String::new();

        while let Some(ch) = chars.next() {
            if ch == '$' {
                // Flush text
                if !text_buf.is_empty() {
                    parts.push(SnippetPart::Text(std::mem::take(&mut text_buf)));
                }

                if chars.peek() == Some(&'{') {
                    chars.next(); // consume '{'
                    let mut inner = String::new();
                    let mut depth = 1;
                    for c in chars.by_ref() {
                        if c == '{' {
                            depth += 1;
                            inner.push(c);
                        } else if c == '}' {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                            inner.push(c);
                        } else {
                            inner.push(c);
                        }
                    }
                    // Parse inner: "1:placeholder" or just "1"
                    if let Some(colon_pos) = inner.find(':') {
                        let idx_str = &inner[..colon_pos];
                        let placeholder = &inner[colon_pos + 1..];
                        if let Ok(idx) = idx_str.parse::<u32>() {
                            parts.push(SnippetPart::Placeholder {
                                index: idx,
                                text: placeholder.to_string(),
                            });
                        }
                    } else if let Ok(idx) = inner.parse::<u32>() {
                        parts.push(SnippetPart::TabStop(idx));
                    }
                } else {
                    // Simple $N
                    let mut num = String::new();
                    while let Some(&c) = chars.peek() {
                        if c.is_ascii_digit() {
                            num.push(c);
                            chars.next();
                        } else {
                            break;
                        }
                    }
                    if let Ok(idx) = num.parse::<u32>() {
                        parts.push(SnippetPart::TabStop(idx));
                    }
                }
            } else {
                text_buf.push(ch);
            }
        }

        if !text_buf.is_empty() {
            parts.push(SnippetPart::Text(text_buf));
        }

        Snippet { parts }
    }

    /// Expand the snippet with given tab stop values.
    pub fn expand(&self, values: &HashMap<u32, String>) -> String {
        let mut result = String::new();
        for part in &self.parts {
            match part {
                SnippetPart::Text(t) => result.push_str(t),
                SnippetPart::TabStop(idx) => {
                    if let Some(val) = values.get(idx) {
                        result.push_str(val);
                    }
                }
                SnippetPart::Placeholder { index, text } => {
                    if let Some(val) = values.get(index) {
                        result.push_str(val);
                    } else {
                        result.push_str(text);
                    }
                }
            }
        }
        result
    }
}

// ── Completion list ─────────────────────────────────────────────

/// Trigger character configuration.
#[derive(Debug, Clone, Default)]
pub struct TriggerConfig {
    /// Characters that trigger completion.
    pub trigger_chars: Vec<char>,
    /// Minimum prefix length to trigger completion (without trigger char).
    pub min_prefix_length: usize,
}

impl TriggerConfig {
    pub fn new(chars: &[char], min_prefix: usize) -> Self {
        Self {
            trigger_chars: chars.to_vec(),
            min_prefix_length: min_prefix,
        }
    }

    /// Check if completion should be triggered for the given context.
    pub fn should_trigger(&self, ctx: &CompletionContext) -> bool {
        if let Some(tc) = ctx.trigger_char {
            if self.trigger_chars.contains(&tc) {
                return true;
            }
        }
        ctx.prefix.len() >= self.min_prefix_length
    }
}

/// A list of completions with filtering.
#[derive(Debug, Clone)]
pub struct CompletionList {
    /// All available items (unfiltered).
    items: Vec<CompletionItem>,
    /// Trigger configuration.
    pub trigger: TriggerConfig,
}

impl CompletionList {
    /// Create a new empty completion list.
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            trigger: TriggerConfig {
                trigger_chars: Vec::new(),
                min_prefix_length: 1,
            },
        }
    }

    /// Create from sources with a context.
    pub fn from_sources(sources: &[&dyn CompletionSource], ctx: &CompletionContext) -> Self {
        let mut items = Vec::new();
        for source in sources {
            let mut source_items = source.completions(ctx);
            for item in &mut source_items {
                item.source_priority = source.priority();
            }
            items.extend(source_items);
        }
        Self {
            items,
            trigger: TriggerConfig::default(),
        }
    }

    /// Add items.
    pub fn add_items(&mut self, items: impl IntoIterator<Item = CompletionItem>) {
        self.items.extend(items);
    }

    /// Filter items by prefix (case-insensitive prefix match).
    pub fn filter_prefix(&self, prefix: &str) -> Vec<&CompletionItem> {
        if prefix.is_empty() {
            return self.items.iter().collect();
        }
        let mut results: Vec<&CompletionItem> = self
            .items
            .iter()
            .filter(|item| prefix_match(prefix, &item.label))
            .collect();
        results.sort_by(|a, b| {
            a.source_priority
                .cmp(&b.source_priority)
                .then_with(|| a.effective_sort_key().cmp(b.effective_sort_key()))
        });
        results
    }

    /// Filter items by fuzzy match, returning items sorted by match score.
    pub fn filter_fuzzy(&self, query: &str) -> Vec<(&CompletionItem, u32)> {
        if query.is_empty() {
            return self.items.iter().map(|item| (item, 0)).collect();
        }
        let mut results: Vec<(&CompletionItem, u32)> = self
            .items
            .iter()
            .filter_map(|item| {
                fuzzy_match(query, &item.label).map(|score| (item, score))
            })
            .collect();
        results.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.0.source_priority.cmp(&b.0.source_priority))
                .then_with(|| a.0.effective_sort_key().cmp(b.0.effective_sort_key()))
        });
        results
    }

    /// Total number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Is the list empty?
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl Default for CompletionList {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply a completion: replace `prefix_len` characters before cursor with the item's insert text.
pub fn accept_completion(
    line: &str,
    cursor_col: usize,
    prefix_len: usize,
    item: &CompletionItem,
) -> String {
    let start = cursor_col.saturating_sub(prefix_len);
    let insert = item.effective_insert_text();
    format!("{}{insert}{}", &line[..start], &line[cursor_col..])
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fuzzy_match_basic() {
        assert!(fuzzy_match("fb", "FooBar").is_some());
        assert!(fuzzy_match("fob", "FooBar").is_some());
        assert!(fuzzy_match("xyz", "FooBar").is_none());
    }

    #[test]
    fn fuzzy_match_exact() {
        let score = fuzzy_match("foo", "foo").unwrap();
        assert_eq!(score, 0); // exact match, no penalty
    }

    #[test]
    fn prefix_match_basic() {
        assert!(prefix_match("get", "getValue"));
        assert!(prefix_match("Get", "getValue")); // case insensitive
        assert!(!prefix_match("set", "getValue"));
    }

    #[test]
    fn completion_list_prefix_filter() {
        let mut list = CompletionList::new();
        list.add_items(vec![
            CompletionItem::new("getValue", CompletionKind::Method),
            CompletionItem::new("getType", CompletionKind::Method),
            CompletionItem::new("setValue", CompletionKind::Method),
        ]);

        let results = list.filter_prefix("get");
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.label.starts_with("get")));
    }

    #[test]
    fn completion_list_fuzzy_filter() {
        let mut list = CompletionList::new();
        list.add_items(vec![
            CompletionItem::new("backgroundColor", CompletionKind::Property),
            CompletionItem::new("borderColor", CompletionKind::Property),
            CompletionItem::new("fontSize", CompletionKind::Property),
        ]);

        let results = list.filter_fuzzy("bgc");
        assert!(!results.is_empty());
        assert_eq!(results[0].0.label, "backgroundColor");
    }

    #[test]
    fn multi_source_merging() {
        let keywords = WordSource::from_strings(&["fn", "let", "mut"], CompletionKind::Keyword, 10);
        let vars = WordSource::from_strings(&["foo", "bar"], CompletionKind::Variable, 50);

        let ctx = CompletionContext::from_line("f", 1, 0);
        let list = CompletionList::from_sources(&[&keywords as &dyn CompletionSource, &vars], &ctx);

        let results = list.filter_prefix("f");
        assert_eq!(results.len(), 2); // "fn" and "foo"
        // keywords should come first (priority 10 < 50)
        assert_eq!(results[0].label, "fn");
        assert_eq!(results[1].label, "foo");
    }

    #[test]
    fn snippet_parse_simple() {
        let snip = Snippet::parse("for $1 in $2 {\n\t$0\n}");
        assert_eq!(snip.parts.len(), 7);
        assert_eq!(snip.parts[0], SnippetPart::Text("for ".to_string()));
        assert_eq!(snip.parts[1], SnippetPart::TabStop(1));
        assert_eq!(snip.parts[2], SnippetPart::Text(" in ".to_string()));
        assert_eq!(snip.parts[3], SnippetPart::TabStop(2));
    }

    #[test]
    fn snippet_parse_placeholder() {
        let snip = Snippet::parse("fn ${1:name}($2) { $0 }");
        assert!(snip.parts.iter().any(|p| matches!(
            p,
            SnippetPart::Placeholder { index: 1, text } if text == "name"
        )));
    }

    #[test]
    fn snippet_expand() {
        let snip = Snippet::parse("Hello, ${1:World}! $0");
        let mut values = HashMap::new();
        values.insert(1, "Rust".to_string());
        values.insert(0, "".to_string());
        let result = snip.expand(&values);
        assert_eq!(result, "Hello, Rust! ");
    }

    #[test]
    fn snippet_expand_defaults() {
        let snip = Snippet::parse("${1:default}");
        let result = snip.expand(&HashMap::new());
        assert_eq!(result, "default");
    }

    #[test]
    fn accept_completion_replaces_prefix() {
        let line = "let val = foo.ge";
        let item = CompletionItem::new("getValue", CompletionKind::Method);
        let result = accept_completion(line, 16, 2, &item);
        assert_eq!(result, "let val = foo.getValue");
    }

    #[test]
    fn trigger_config() {
        let trigger = TriggerConfig::new(&['.', ':'], 2);
        let ctx1 = CompletionContext::from_line("foo.", 4, 0);
        assert!(trigger.should_trigger(&ctx1));

        let ctx2 = CompletionContext::from_line("fo", 2, 0);
        assert!(trigger.should_trigger(&ctx2));

        let ctx3 = CompletionContext::from_line("f", 1, 0);
        assert!(!trigger.should_trigger(&ctx3));
    }

    #[test]
    fn completion_context_prefix() {
        let ctx = CompletionContext::from_line("let myVar = get", 15, 0);
        assert_eq!(ctx.prefix, "get");
    }

    #[test]
    fn completion_item_builder() {
        let item = CompletionItem::new("println", CompletionKind::Function)
            .with_detail("macro")
            .with_insert_text("println!(\"$1\")")
            .with_sort_key("aaa");
        assert_eq!(item.detail.as_deref(), Some("macro"));
        assert_eq!(item.effective_insert_text(), "println!(\"$1\")");
        assert_eq!(item.effective_sort_key(), "aaa");
    }
}
