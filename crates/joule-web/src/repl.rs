//! REPL (Read-Eval-Print Loop) framework.
//!
//! Provides a configurable REPL engine with command history, input editing
//! simulation, tab completion, multi-line input, built-in commands
//! (`/help`, `/quit`, `/clear`), custom evaluator trait, and session state.
//! Pure Rust — no readline or terminal dependencies.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ──────────────────────────────────────────────────────

/// Errors from REPL operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplError {
    /// Evaluator returned an error.
    EvalError(String),
    /// Input was empty or invalid.
    EmptyInput,
    /// REPL was asked to quit.
    Quit,
    /// Unknown built-in command.
    UnknownCommand(String),
    /// History index out of range.
    HistoryOutOfRange(usize),
}

impl fmt::Display for ReplError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EvalError(msg) => write!(f, "evaluation error: {msg}"),
            Self::EmptyInput => write!(f, "empty input"),
            Self::Quit => write!(f, "quit"),
            Self::UnknownCommand(cmd) => write!(f, "unknown command: {cmd}"),
            Self::HistoryOutOfRange(idx) => write!(f, "history index {idx} out of range"),
            }
    }
}

impl std::error::Error for ReplError {}

// ── Input Line Buffer ───────────────────────────────────────────

/// Simulated line-editing buffer with cursor position.
#[derive(Debug, Clone)]
pub struct LineBuffer {
    /// Characters in the buffer.
    chars: Vec<char>,
    /// Cursor position (0-based, 0..=chars.len()).
    cursor: usize,
}

impl LineBuffer {
    /// Create an empty line buffer.
    pub fn new() -> Self {
        Self {
            chars: Vec::new(),
            cursor: 0,
        }
    }

    /// Create from a string.
    pub fn from_str(s: &str) -> Self {
        let chars: Vec<char> = s.chars().collect();
        let cursor = chars.len();
        Self { chars, cursor }
    }

    /// Get the current contents as a string.
    pub fn contents(&self) -> String {
        self.chars.iter().collect()
    }

    /// Get cursor position.
    pub fn cursor_pos(&self) -> usize {
        self.cursor
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.chars.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.chars.is_empty()
    }

    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, ch: char) {
        self.chars.insert(self.cursor, ch);
        self.cursor += 1;
    }

    /// Insert a string at the cursor position.
    pub fn insert_str(&mut self, s: &str) {
        for ch in s.chars() {
            self.insert_char(ch);
        }
    }

    /// Delete the character before the cursor (backspace).
    pub fn backspace(&mut self) -> bool {
        if self.cursor > 0 {
            self.cursor -= 1;
            self.chars.remove(self.cursor);
            true
        } else {
            false
        }
    }

    /// Delete the character at the cursor position.
    pub fn delete(&mut self) -> bool {
        if self.cursor < self.chars.len() {
            self.chars.remove(self.cursor);
            true
        } else {
            false
        }
    }

    /// Move cursor left.
    pub fn move_left(&mut self) -> bool {
        if self.cursor > 0 {
            self.cursor -= 1;
            true
        } else {
            false
        }
    }

    /// Move cursor right.
    pub fn move_right(&mut self) -> bool {
        if self.cursor < self.chars.len() {
            self.cursor += 1;
            true
        } else {
            false
        }
    }

    /// Move cursor to the start of the line.
    pub fn move_home(&mut self) {
        self.cursor = 0;
    }

    /// Move cursor to the end of the line.
    pub fn move_end(&mut self) {
        self.cursor = self.chars.len();
    }

    /// Move cursor to the previous word boundary.
    pub fn move_word_left(&mut self) {
        // Skip whitespace, then skip non-whitespace.
        while self.cursor > 0 && self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
        while self.cursor > 0 && !self.chars[self.cursor - 1].is_whitespace() {
            self.cursor -= 1;
        }
    }

    /// Move cursor to the next word boundary.
    pub fn move_word_right(&mut self) {
        let len = self.chars.len();
        while self.cursor < len && !self.chars[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
        while self.cursor < len && self.chars[self.cursor].is_whitespace() {
            self.cursor += 1;
        }
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.chars.clear();
        self.cursor = 0;
    }

    /// Set the buffer to a new value, cursor at end.
    pub fn set(&mut self, s: &str) {
        self.chars = s.chars().collect();
        self.cursor = self.chars.len();
    }

    /// Kill text from cursor to end of line, returning the killed text.
    pub fn kill_to_end(&mut self) -> String {
        let killed: String = self.chars[self.cursor..].iter().collect();
        self.chars.truncate(self.cursor);
        killed
    }

    /// Kill text from start to cursor, returning the killed text.
    pub fn kill_to_start(&mut self) -> String {
        let killed: String = self.chars[..self.cursor].iter().collect();
        self.chars = self.chars[self.cursor..].to_vec();
        self.cursor = 0;
        killed
    }
}

impl Default for LineBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ── Command History ─────────────────────────────────────────────

/// Command history with navigation.
#[derive(Debug, Clone)]
pub struct History {
    entries: Vec<String>,
    max_size: usize,
    /// Current navigation position (entries.len() means "no selection").
    position: usize,
}

impl History {
    /// Create with a maximum history size.
    pub fn new(max_size: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_size,
            position: 0,
        }
    }

    /// Add an entry to history.
    pub fn push(&mut self, entry: String) {
        // Don't add duplicates of the most recent entry.
        if self.entries.last().map(|s| s.as_str()) == Some(&entry) {
            self.position = self.entries.len();
            return;
        }
        if self.entries.len() >= self.max_size && self.max_size > 0 {
            self.entries.remove(0);
        }
        self.entries.push(entry);
        self.position = self.entries.len();
    }

    /// Navigate to the previous history entry. Returns the entry text if available.
    pub fn previous(&mut self) -> Option<&str> {
        if self.position > 0 {
            self.position -= 1;
            Some(&self.entries[self.position])
        } else {
            None
        }
    }

    /// Navigate to the next history entry. Returns the entry text or None for current input.
    pub fn next(&mut self) -> Option<&str> {
        if self.position < self.entries.len() {
            self.position += 1;
            if self.position < self.entries.len() {
                Some(&self.entries[self.position])
            } else {
                None // Back to current input
            }
        } else {
            None
        }
    }

    /// Reset navigation to the end (most recent).
    pub fn reset_position(&mut self) {
        self.position = self.entries.len();
    }

    /// Get all history entries.
    pub fn entries(&self) -> &[String] {
        &self.entries
    }

    /// Get entry by index.
    pub fn get(&self, index: usize) -> Option<&str> {
        self.entries.get(index).map(|s| s.as_str())
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all history.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.position = 0;
    }

    /// Search history backwards for entries containing the given substring.
    pub fn search_backward(&self, query: &str) -> Vec<(usize, &str)> {
        self.entries
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, entry)| entry.contains(query))
            .map(|(i, entry)| (i, entry.as_str()))
            .collect()
    }
}

// ── Completion ──────────────────────────────────────────────────

/// A completion candidate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Completion {
    /// The replacement text.
    pub text: String,
    /// Optional display text (for menu).
    pub display: Option<String>,
    /// Optional description.
    pub description: Option<String>,
}

impl Completion {
    /// Create a simple completion.
    pub fn simple(text: &str) -> Self {
        Self {
            text: text.to_string(),
            display: None,
            description: None,
        }
    }

    /// Create a completion with description.
    pub fn with_description(text: &str, desc: &str) -> Self {
        Self {
            text: text.to_string(),
            display: None,
            description: Some(desc.to_string()),
        }
    }
}

/// Trait for providing completions.
pub trait Completer {
    /// Return completions for the given input and cursor position.
    fn complete(&self, line: &str, cursor_pos: usize) -> Vec<Completion>;
}

/// Default completer that uses a static word list.
#[derive(Debug, Clone)]
pub struct WordCompleter {
    words: Vec<String>,
}

impl WordCompleter {
    pub fn new(words: Vec<String>) -> Self {
        Self { words }
    }
}

impl Completer for WordCompleter {
    fn complete(&self, line: &str, cursor_pos: usize) -> Vec<Completion> {
        let before_cursor = &line[..cursor_pos.min(line.len())];
        // Find the word being typed.
        let word_start = before_cursor.rfind(|c: char| c.is_whitespace()).map_or(0, |p| p + 1);
        let partial = &before_cursor[word_start..];
        if partial.is_empty() {
            return Vec::new();
        }
        self.words
            .iter()
            .filter(|w| w.starts_with(partial))
            .map(|w| Completion::simple(w))
            .collect()
    }
}

// ── Evaluator Trait ─────────────────────────────────────────────

/// Result of evaluating a REPL expression.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvalResult {
    /// Output text.
    pub output: String,
    /// Whether the result is an error.
    pub is_error: bool,
}

impl EvalResult {
    /// Create a successful result.
    pub fn ok(output: &str) -> Self {
        Self {
            output: output.to_string(),
            is_error: false,
        }
    }

    /// Create an error result.
    pub fn err(msg: &str) -> Self {
        Self {
            output: msg.to_string(),
            is_error: true,
        }
    }
}

impl fmt::Display for EvalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_error {
            write!(f, "Error: {}", self.output)
        } else {
            write!(f, "{}", self.output)
        }
    }
}

/// Trait for evaluating REPL input.
pub trait Evaluator {
    /// Evaluate the given input and return a result.
    fn eval(&mut self, input: &str) -> EvalResult;

    /// Check if the input is complete (for multi-line mode).
    /// Returns true if the input can be evaluated, false if more lines are needed.
    fn is_complete(&self, input: &str) -> bool;
}

// ── Built-in Commands ───────────────────────────────────────────

/// A built-in REPL command.
#[derive(Debug, Clone)]
pub struct BuiltinCommand {
    pub name: String,
    pub aliases: Vec<String>,
    pub description: String,
}

/// The action a built-in command should trigger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BuiltinAction {
    Help,
    Quit,
    Clear,
    History,
    Custom(String),
}

/// Registry of built-in commands.
#[derive(Debug, Clone)]
pub struct CommandRegistry {
    commands: Vec<(BuiltinCommand, BuiltinAction)>,
}

impl CommandRegistry {
    /// Create with default REPL commands.
    pub fn with_defaults() -> Self {
        let mut reg = Self {
            commands: Vec::new(),
        };
        reg.register(
            BuiltinCommand {
                name: "/help".to_string(),
                aliases: vec!["/h".to_string(), "/?".to_string()],
                description: "Show available commands".to_string(),
            },
            BuiltinAction::Help,
        );
        reg.register(
            BuiltinCommand {
                name: "/quit".to_string(),
                aliases: vec!["/q".to_string(), "/exit".to_string()],
                description: "Exit the REPL".to_string(),
            },
            BuiltinAction::Quit,
        );
        reg.register(
            BuiltinCommand {
                name: "/clear".to_string(),
                aliases: vec!["/cls".to_string()],
                description: "Clear screen output".to_string(),
            },
            BuiltinAction::Clear,
        );
        reg.register(
            BuiltinCommand {
                name: "/history".to_string(),
                aliases: vec!["/hist".to_string()],
                description: "Show command history".to_string(),
            },
            BuiltinAction::History,
        );
        reg
    }

    /// Register a command.
    pub fn register(&mut self, cmd: BuiltinCommand, action: BuiltinAction) {
        self.commands.push((cmd, action));
    }

    /// Look up a command by name or alias.
    pub fn lookup(&self, input: &str) -> Option<&BuiltinAction> {
        let trimmed = input.trim();
        for (cmd, action) in &self.commands {
            if cmd.name == trimmed || cmd.aliases.iter().any(|a| a == trimmed) {
                return Some(action);
            }
        }
        None
    }

    /// Get all commands for help display.
    pub fn all_commands(&self) -> &[(BuiltinCommand, BuiltinAction)] {
        &self.commands
    }

    /// Generate help text.
    pub fn help_text(&self) -> String {
        let mut out = String::from("Available commands:\n");
        for (cmd, _) in &self.commands {
            let aliases = if cmd.aliases.is_empty() {
                String::new()
            } else {
                format!(" ({})", cmd.aliases.join(", "))
            };
            out.push_str(&format!("  {}{} — {}\n", cmd.name, aliases, cmd.description));
        }
        out
    }
}

impl Default for CommandRegistry {
    fn default() -> Self {
        Self::with_defaults()
    }
}

// ── Multi-line Input ────────────────────────────────────────────

/// State for accumulating multi-line input.
#[derive(Debug, Clone)]
pub struct MultiLineInput {
    lines: Vec<String>,
    /// The continuation prompt to show for subsequent lines.
    pub continuation_prompt: String,
}

impl MultiLineInput {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            continuation_prompt: "... ".to_string(),
        }
    }

    /// Add a line.
    pub fn add_line(&mut self, line: &str) {
        self.lines.push(line.to_string());
    }

    /// Get the accumulated input as a single string (lines joined with newlines).
    pub fn combined(&self) -> String {
        self.lines.join("\n")
    }

    /// Number of lines accumulated so far.
    pub fn line_count(&self) -> usize {
        self.lines.len()
    }

    /// Check if there is any accumulated input.
    pub fn is_empty(&self) -> bool {
        self.lines.is_empty()
    }

    /// Clear the accumulated input.
    pub fn clear(&mut self) {
        self.lines.clear();
    }
}

impl Default for MultiLineInput {
    fn default() -> Self {
        Self::new()
    }
}

// ── Session State ───────────────────────────────────────────────

/// Key-value session state for the REPL.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionState {
    values: HashMap<String, String>,
}

impl SessionState {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a session variable.
    pub fn set(&mut self, key: &str, value: &str) {
        self.values.insert(key.to_string(), value.to_string());
    }

    /// Get a session variable.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.values.get(key).map(|s| s.as_str())
    }

    /// Remove a session variable.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.values.remove(key)
    }

    /// Check if a key exists.
    pub fn contains(&self, key: &str) -> bool {
        self.values.contains_key(key)
    }

    /// Get all keys.
    pub fn keys(&self) -> Vec<&str> {
        self.values.keys().map(|k| k.as_str()).collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear all state.
    pub fn clear(&mut self) {
        self.values.clear();
    }
}

// ── Output Record ───────────────────────────────────────────────

/// A single REPL interaction record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionRecord {
    pub input: String,
    pub output: String,
    pub is_error: bool,
    pub timestamp_ms: u64,
}

// ── REPL Engine ─────────────────────────────────────────────────

/// Configuration for the REPL.
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// Primary prompt string.
    pub prompt: String,
    /// Continuation prompt for multi-line input.
    pub continuation_prompt: String,
    /// Maximum history entries to keep.
    pub max_history: usize,
    /// Whether multi-line input is enabled.
    pub multi_line: bool,
    /// Welcome banner displayed at start.
    pub banner: Option<String>,
    /// Goodbye message displayed on quit.
    pub goodbye: Option<String>,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: ">>> ".to_string(),
            continuation_prompt: "... ".to_string(),
            max_history: 1000,
            multi_line: true,
            banner: None,
            goodbye: None,
        }
    }
}

/// The REPL engine.
///
/// Processes input lines through the read-eval-print cycle. Does not handle
/// actual terminal I/O — instead exposes methods for feeding lines, getting
/// prompts, and reading output, so it can be driven by any frontend (terminal,
/// WebSocket, test harness).
pub struct Repl<E: Evaluator> {
    pub config: ReplConfig,
    pub history: History,
    pub commands: CommandRegistry,
    pub session: SessionState,
    evaluator: E,
    multi_line: MultiLineInput,
    output_log: Vec<InteractionRecord>,
    interaction_count: u64,
    cleared: bool,
}

impl<E: Evaluator> Repl<E> {
    /// Create a new REPL with the given evaluator and config.
    pub fn new(evaluator: E, config: ReplConfig) -> Self {
        let max_history = config.max_history;
        let continuation_prompt = config.continuation_prompt.clone();
        Self {
            config,
            history: History::new(max_history),
            commands: CommandRegistry::with_defaults(),
            session: SessionState::new(),
            evaluator,
            multi_line: MultiLineInput {
                lines: Vec::new(),
                continuation_prompt,
            },
            output_log: Vec::new(),
            interaction_count: 0,
            cleared: false,
        }
    }

    /// Get the current prompt string.
    pub fn prompt(&self) -> &str {
        if self.multi_line.is_empty() {
            &self.config.prompt
        } else {
            &self.config.continuation_prompt
        }
    }

    /// Get the banner text, if configured.
    pub fn banner(&self) -> Option<&str> {
        self.config.banner.as_deref()
    }

    /// Process a single line of input. Returns the action result.
    pub fn process_line(&mut self, line: &str) -> Result<Option<EvalResult>, ReplError> {
        let trimmed = line.trim();

        // Handle empty input when not in multi-line mode.
        if trimmed.is_empty() && self.multi_line.is_empty() {
            return Err(ReplError::EmptyInput);
        }

        // Check for built-in commands (only when not in multi-line accumulation).
        if self.multi_line.is_empty() && trimmed.starts_with('/') {
            let action = self.commands.lookup(trimmed).cloned();
            if let Some(action) = action {
                return self.execute_builtin(&action);
            }
            return Err(ReplError::UnknownCommand(trimmed.to_string()));
        }

        // Multi-line accumulation.
        if self.config.multi_line {
            self.multi_line.add_line(line);
            let combined = self.multi_line.combined();
            if self.evaluator.is_complete(&combined) {
                self.multi_line.clear();
                let result = self.evaluate_input(&combined);
                self.history.push(combined);
                return Ok(Some(result));
            }
            return Ok(None); // Need more input.
        }

        // Single-line mode: evaluate immediately.
        self.history.push(line.to_string());
        let result = self.evaluate_input(line);
        Ok(Some(result))
    }

    /// Evaluate input through the evaluator.
    fn evaluate_input(&mut self, input: &str) -> EvalResult {
        self.interaction_count += 1;
        let result = self.evaluator.eval(input);
        self.output_log.push(InteractionRecord {
            input: input.to_string(),
            output: result.output.clone(),
            is_error: result.is_error,
            timestamp_ms: self.interaction_count * 1000, // placeholder
        });
        result
    }

    /// Execute a built-in command action.
    fn execute_builtin(&self, action: &BuiltinAction) -> Result<Option<EvalResult>, ReplError> {
        match action {
            BuiltinAction::Quit => Err(ReplError::Quit),
            BuiltinAction::Help => {
                let text = self.commands.help_text();
                Ok(Some(EvalResult::ok(&text)))
            }
            BuiltinAction::Clear => Ok(Some(EvalResult::ok("[screen cleared]"))),
            BuiltinAction::History => {
                let mut text = String::new();
                for (i, entry) in self.history.entries().iter().enumerate() {
                    text.push_str(&format!("{:4}  {}\n", i + 1, entry));
                }
                if text.is_empty() {
                    text = "No history.".to_string();
                }
                Ok(Some(EvalResult::ok(&text)))
            }
            BuiltinAction::Custom(name) => {
                Ok(Some(EvalResult::ok(&format!("[custom: {name}]"))))
            }
        }
    }

    /// Check whether the REPL is in multi-line input accumulation.
    pub fn is_accumulating(&self) -> bool {
        !self.multi_line.is_empty()
    }

    /// Cancel the current multi-line input.
    pub fn cancel_multi_line(&mut self) {
        self.multi_line.clear();
    }

    /// Get the number of interactions processed.
    pub fn interaction_count(&self) -> u64 {
        self.interaction_count
    }

    /// Get the output log.
    pub fn output_log(&self) -> &[InteractionRecord] {
        &self.output_log
    }

    /// Get mutable access to the evaluator.
    pub fn evaluator_mut(&mut self) -> &mut E {
        &mut self.evaluator
    }

    /// Get access to the evaluator.
    pub fn evaluator(&self) -> &E {
        &self.evaluator
    }

    /// Check whether screen was cleared.
    pub fn was_cleared(&self) -> bool {
        self.cleared
    }
}

// ── Simple Calculator Evaluator (for tests and examples) ────────

/// A simple arithmetic evaluator for testing.
#[derive(Debug, Clone, Default)]
pub struct CalcEvaluator {
    /// Last result for `_` variable.
    pub last_result: Option<f64>,
    /// Named variables.
    pub variables: HashMap<String, f64>,
}

impl CalcEvaluator {
    pub fn new() -> Self {
        Self::default()
    }

    fn parse_and_eval(&mut self, input: &str) -> Result<f64, String> {
        let trimmed = input.trim();

        // Variable assignment: x = expr
        if let Some(eq_pos) = trimmed.find('=') {
            let var_name = trimmed[..eq_pos].trim();
            let expr = trimmed[eq_pos + 1..].trim();
            if var_name.chars().all(|c| c.is_alphanumeric() || c == '_') && !var_name.is_empty() {
                let val = self.eval_expr(expr)?;
                self.variables.insert(var_name.to_string(), val);
                return Ok(val);
            }
        }

        self.eval_expr(trimmed)
    }

    fn eval_expr(&self, expr: &str) -> Result<f64, String> {
        let trimmed = expr.trim();

        // Check for variable reference.
        if trimmed == "_" {
            return self
                .last_result
                .ok_or_else(|| "no previous result".to_string());
        }
        if let Some(val) = self.variables.get(trimmed) {
            return Ok(*val);
        }

        // Simple binary operation: a op b
        if let Some(pos) = trimmed.rfind('+') {
            if pos > 0 {
                let left = self.eval_expr(&trimmed[..pos])?;
                let right = self.eval_expr(&trimmed[pos + 1..])?;
                return Ok(left + right);
            }
        }
        if let Some(pos) = trimmed.rfind('-') {
            if pos > 0 {
                let left = self.eval_expr(&trimmed[..pos])?;
                let right = self.eval_expr(&trimmed[pos + 1..])?;
                return Ok(left - right);
            }
        }
        if let Some(pos) = trimmed.rfind('*') {
            let left = self.eval_expr(&trimmed[..pos])?;
            let right = self.eval_expr(&trimmed[pos + 1..])?;
            return Ok(left * right);
        }
        if let Some(pos) = trimmed.rfind('/') {
            let left = self.eval_expr(&trimmed[..pos])?;
            let right = self.eval_expr(&trimmed[pos + 1..])?;
            if right == 0.0 {
                return Err("division by zero".to_string());
            }
            return Ok(left / right);
        }

        // Try parsing as a number.
        trimmed
            .parse::<f64>()
            .map_err(|_| format!("cannot parse: {trimmed}"))
    }
}

impl Evaluator for CalcEvaluator {
    fn eval(&mut self, input: &str) -> EvalResult {
        match self.parse_and_eval(input) {
            Ok(val) => {
                self.last_result = Some(val);
                EvalResult::ok(&format!("{val}"))
            }
            Err(msg) => EvalResult::err(&msg),
        }
    }

    fn is_complete(&self, input: &str) -> bool {
        // Multi-line: treat lines ending with `\` as incomplete.
        !input.ends_with('\\')
    }
}

// ── Echo Evaluator (simplest possible for testing) ──────────────

/// An evaluator that just echoes the input.
#[derive(Debug, Clone, Default)]
pub struct EchoEvaluator;

impl Evaluator for EchoEvaluator {
    fn eval(&mut self, input: &str) -> EvalResult {
        EvalResult::ok(input)
    }

    fn is_complete(&self, _input: &str) -> bool {
        true
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── LineBuffer tests ────────────────────────────────────────

    #[test]
    fn line_buffer_insert_and_contents() {
        let mut buf = LineBuffer::new();
        buf.insert_char('h');
        buf.insert_char('i');
        assert_eq!(buf.contents(), "hi");
        assert_eq!(buf.cursor_pos(), 2);
        assert_eq!(buf.len(), 2);
    }

    #[test]
    fn line_buffer_backspace() {
        let mut buf = LineBuffer::from_str("hello");
        assert!(buf.backspace());
        assert_eq!(buf.contents(), "hell");
        buf.move_home();
        assert!(!buf.backspace());
    }

    #[test]
    fn line_buffer_delete() {
        let mut buf = LineBuffer::from_str("hello");
        buf.move_home();
        assert!(buf.delete());
        assert_eq!(buf.contents(), "ello");
        buf.move_end();
        assert!(!buf.delete());
    }

    #[test]
    fn line_buffer_cursor_movement() {
        let mut buf = LineBuffer::from_str("abc");
        assert_eq!(buf.cursor_pos(), 3);
        buf.move_left();
        assert_eq!(buf.cursor_pos(), 2);
        buf.move_left();
        buf.move_left();
        assert_eq!(buf.cursor_pos(), 0);
        assert!(!buf.move_left());
        buf.move_right();
        assert_eq!(buf.cursor_pos(), 1);
    }

    #[test]
    fn line_buffer_word_movement() {
        let mut buf = LineBuffer::from_str("hello world foo");
        buf.move_word_left();
        assert_eq!(buf.cursor_pos(), 12); // before "foo"
        buf.move_word_left();
        assert_eq!(buf.cursor_pos(), 6); // before "world"
        buf.move_word_right();
        assert_eq!(buf.cursor_pos(), 12); // after "world "
    }

    #[test]
    fn line_buffer_kill() {
        let mut buf = LineBuffer::from_str("hello world");
        buf.cursor = 5; // before " world"
        let killed = buf.kill_to_end();
        assert_eq!(killed, " world");
        assert_eq!(buf.contents(), "hello");

        let mut buf2 = LineBuffer::from_str("hello world");
        buf2.cursor = 5;
        let killed2 = buf2.kill_to_start();
        assert_eq!(killed2, "hello");
        assert_eq!(buf2.contents(), " world");
    }

    #[test]
    fn line_buffer_insert_str() {
        let mut buf = LineBuffer::from_str("hd");
        buf.cursor = 1;
        buf.insert_str("ello worl");
        assert_eq!(buf.contents(), "hello world");
    }

    // ── History tests ───────────────────────────────────────────

    #[test]
    fn history_push_and_navigate() {
        let mut hist = History::new(100);
        hist.push("first".to_string());
        hist.push("second".to_string());
        hist.push("third".to_string());
        assert_eq!(hist.len(), 3);

        assert_eq!(hist.previous(), Some("third"));
        assert_eq!(hist.previous(), Some("second"));
        assert_eq!(hist.previous(), Some("first"));
        assert_eq!(hist.previous(), None);

        assert_eq!(hist.next(), Some("second"));
        assert_eq!(hist.next(), Some("third"));
        assert_eq!(hist.next(), None); // back to current
    }

    #[test]
    fn history_max_size() {
        let mut hist = History::new(3);
        hist.push("a".to_string());
        hist.push("b".to_string());
        hist.push("c".to_string());
        hist.push("d".to_string());
        assert_eq!(hist.len(), 3);
        assert_eq!(hist.get(0), Some("b"));
    }

    #[test]
    fn history_no_duplicates() {
        let mut hist = History::new(100);
        hist.push("same".to_string());
        hist.push("same".to_string());
        assert_eq!(hist.len(), 1);
    }

    #[test]
    fn history_search() {
        let mut hist = History::new(100);
        hist.push("ls -la".to_string());
        hist.push("cat file.txt".to_string());
        hist.push("ls src/".to_string());
        let results = hist.search_backward("ls");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].1, "ls src/");
        assert_eq!(results[1].1, "ls -la");
    }

    // ── Completion tests ────────────────────────────────────────

    #[test]
    fn word_completer_basic() {
        let wc = WordCompleter::new(vec![
            "help".to_string(),
            "hello".to_string(),
            "quit".to_string(),
        ]);
        let completions = wc.complete("hel", 3);
        assert_eq!(completions.len(), 2);
        assert!(completions.iter().any(|c| c.text == "help"));
        assert!(completions.iter().any(|c| c.text == "hello"));
    }

    #[test]
    fn word_completer_no_match() {
        let wc = WordCompleter::new(vec!["alpha".to_string()]);
        let completions = wc.complete("zz", 2);
        assert!(completions.is_empty());
    }

    #[test]
    fn word_completer_mid_line() {
        let wc = WordCompleter::new(vec!["world".to_string(), "work".to_string()]);
        let completions = wc.complete("hello wor", 9);
        assert_eq!(completions.len(), 2);
    }

    // ── REPL engine tests ───────────────────────────────────────

    #[test]
    fn repl_echo_evaluator() {
        let eval = EchoEvaluator;
        let config = ReplConfig::default();
        let mut repl = Repl::new(eval, config);
        let result = repl.process_line("hello world").unwrap().unwrap();
        assert_eq!(result.output, "hello world");
        assert!(!result.is_error);
    }

    #[test]
    fn repl_quit_command() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        let err = repl.process_line("/quit").unwrap_err();
        assert_eq!(err, ReplError::Quit);
    }

    #[test]
    fn repl_help_command() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        let result = repl.process_line("/help").unwrap().unwrap();
        assert!(result.output.contains("/help"));
        assert!(result.output.contains("/quit"));
    }

    #[test]
    fn repl_unknown_command() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        let err = repl.process_line("/foobar").unwrap_err();
        assert!(matches!(err, ReplError::UnknownCommand(_)));
    }

    #[test]
    fn repl_empty_input() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        let err = repl.process_line("").unwrap_err();
        assert_eq!(err, ReplError::EmptyInput);
    }

    #[test]
    fn repl_history_recorded() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        repl.process_line("first").unwrap();
        repl.process_line("second").unwrap();
        assert_eq!(repl.history.len(), 2);
        assert_eq!(repl.history.get(0), Some("first"));
    }

    #[test]
    fn repl_output_log() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        repl.process_line("test input").unwrap();
        assert_eq!(repl.output_log().len(), 1);
        assert_eq!(repl.output_log()[0].input, "test input");
    }

    #[test]
    fn repl_interaction_count() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        assert_eq!(repl.interaction_count(), 0);
        repl.process_line("a").unwrap();
        repl.process_line("b").unwrap();
        assert_eq!(repl.interaction_count(), 2);
    }

    #[test]
    fn repl_calc_evaluator() {
        let eval = CalcEvaluator::new();
        let mut repl = Repl::new(eval, ReplConfig::default());
        let result = repl.process_line("2+3").unwrap().unwrap();
        assert_eq!(result.output, "5");
    }

    #[test]
    fn repl_multi_line_input() {
        // Evaluator where lines ending with '\' are incomplete.
        let eval = CalcEvaluator::new();
        let mut repl = Repl::new(eval, ReplConfig::default());

        // First line ends with \, so it should be incomplete.
        let r1 = repl.process_line("2+\\").unwrap();
        assert!(r1.is_none()); // still accumulating
        assert!(repl.is_accumulating());

        // Complete the expression.
        let r2 = repl.process_line("3").unwrap();
        assert!(r2.is_some());
        assert!(!repl.is_accumulating());
    }

    #[test]
    fn repl_cancel_multi_line() {
        let eval = CalcEvaluator::new();
        let mut repl = Repl::new(eval, ReplConfig::default());
        repl.process_line("incomplete\\").unwrap();
        assert!(repl.is_accumulating());
        repl.cancel_multi_line();
        assert!(!repl.is_accumulating());
    }

    #[test]
    fn repl_prompt_changes_for_multiline() {
        let eval = CalcEvaluator::new();
        let config = ReplConfig {
            prompt: ">> ".to_string(),
            continuation_prompt: ".. ".to_string(),
            ..ReplConfig::default()
        };
        let mut repl = Repl::new(eval, config);
        assert_eq!(repl.prompt(), ">> ");
        repl.process_line("test\\").unwrap();
        assert_eq!(repl.prompt(), ".. ");
    }

    #[test]
    fn repl_session_state() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        repl.session.set("user", "alice");
        assert_eq!(repl.session.get("user"), Some("alice"));
        assert!(repl.session.contains("user"));
        repl.session.remove("user");
        assert!(!repl.session.contains("user"));
    }

    #[test]
    fn repl_command_aliases() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        // /q is alias for /quit.
        let err = repl.process_line("/q").unwrap_err();
        assert_eq!(err, ReplError::Quit);
        // /exit is alias for /quit.
        let mut repl2 = Repl::new(EchoEvaluator, ReplConfig::default());
        let err2 = repl2.process_line("/exit").unwrap_err();
        assert_eq!(err2, ReplError::Quit);
    }

    #[test]
    fn repl_history_command() {
        let eval = EchoEvaluator;
        let mut repl = Repl::new(eval, ReplConfig::default());
        repl.process_line("first").unwrap();
        repl.process_line("second").unwrap();
        let result = repl.process_line("/history").unwrap().unwrap();
        assert!(result.output.contains("first"));
        assert!(result.output.contains("second"));
    }

    #[test]
    fn multi_line_input_combined() {
        let mut ml = MultiLineInput::new();
        assert!(ml.is_empty());
        ml.add_line("line one");
        ml.add_line("line two");
        assert_eq!(ml.line_count(), 2);
        assert_eq!(ml.combined(), "line one\nline two");
        ml.clear();
        assert!(ml.is_empty());
    }

    #[test]
    fn session_state_clear() {
        let mut state = SessionState::new();
        state.set("a", "1");
        state.set("b", "2");
        assert_eq!(state.len(), 2);
        state.clear();
        assert!(state.is_empty());
    }

    #[test]
    fn eval_result_display() {
        let ok = EvalResult::ok("42");
        assert_eq!(format!("{ok}"), "42");
        let err = EvalResult::err("bad input");
        assert_eq!(format!("{err}"), "Error: bad input");
    }

    #[test]
    fn repl_error_display() {
        let e = ReplError::Quit;
        assert_eq!(format!("{e}"), "quit");
        let e2 = ReplError::UnknownCommand("/xyz".to_string());
        assert!(format!("{e2}").contains("/xyz"));
    }

    #[test]
    fn line_buffer_clear_and_set() {
        let mut buf = LineBuffer::from_str("old");
        buf.clear();
        assert!(buf.is_empty());
        buf.set("new text");
        assert_eq!(buf.contents(), "new text");
        assert_eq!(buf.cursor_pos(), 8);
    }
}
