//! Interactive CLI prompt models for text, password, confirm, select, and multi-select.
//!
//! Provides prompt state machines and validation logic without direct I/O.
//! Each prompt type manages its own state, accepts input events, and produces
//! rendered output strings. Actual terminal I/O is handled by the caller.

use std::fmt;

// ── Validation ──

/// Result of validating prompt input.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    Valid,
    Invalid(String),
}

/// A validation function boxed for storage (for runtime use).
/// For the pure-model approach, we use closures at call sites
/// and provide a simple range validator here.
#[derive(Debug, Clone)]
pub struct RangeValidator {
    pub min: Option<f64>,
    pub max: Option<f64>,
}

impl RangeValidator {
    pub fn new(min: Option<f64>, max: Option<f64>) -> Self {
        Self { min, max }
    }

    pub fn validate(&self, value: f64) -> ValidationResult {
        if let Some(min) = self.min {
            if value < min {
                return ValidationResult::Invalid(format!("Value must be >= {min}"));
            }
        }
        if let Some(max) = self.max {
            if value > max {
                return ValidationResult::Invalid(format!("Value must be <= {max}"));
            }
        }
        ValidationResult::Valid
    }
}

// ── Text Prompt ──

/// A text input prompt with optional default and validation.
#[derive(Debug, Clone)]
pub struct TextPrompt {
    pub label: String,
    pub default: Option<String>,
    pub placeholder: String,
    pub input: String,
    pub error: Option<String>,
    pub required: bool,
}

impl TextPrompt {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            default: None,
            placeholder: String::new(),
            input: String::new(),
            error: None,
            required: false,
        }
    }

    pub fn with_default(mut self, default: &str) -> Self {
        self.default = Some(default.to_string());
        self
    }

    pub fn with_placeholder(mut self, ph: &str) -> Self {
        self.placeholder = ph.to_string();
        self
    }

    pub fn required(mut self) -> Self {
        self.required = true;
        self
    }

    /// Set raw input text.
    pub fn set_input(&mut self, text: &str) {
        self.input = text.to_string();
        self.error = None;
    }

    /// Append a character.
    pub fn push_char(&mut self, ch: char) {
        self.input.push(ch);
        self.error = None;
    }

    /// Delete last character.
    pub fn backspace(&mut self) {
        self.input.pop();
    }

    /// Resolve the final value (input or default).
    pub fn value(&self) -> Option<String> {
        if self.input.is_empty() {
            self.default.clone()
        } else {
            Some(self.input.clone())
        }
    }

    /// Validate and return the final value or an error.
    pub fn submit(&mut self) -> Result<String, String> {
        match self.value() {
            Some(v) if !v.is_empty() => Ok(v),
            Some(v) if !self.required => Ok(v),
            _ => {
                let err = "This field is required".to_string();
                self.error = Some(err.clone());
                Err(err)
            }
        }
    }

    /// Render the prompt line.
    pub fn render(&self) -> String {
        let default_hint = self.default.as_ref()
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        let display = if self.input.is_empty() && !self.placeholder.is_empty() {
            format!("\x1b[2m{}\x1b[0m", self.placeholder)
        } else {
            self.input.clone()
        };
        let err_line = self.error.as_ref()
            .map(|e| format!("\n\x1b[31m  {e}\x1b[0m"))
            .unwrap_or_default();
        format!("? {}{default_hint}: {display}{err_line}", self.label)
    }
}

// ── Password Prompt ──

/// A masked password input prompt.
#[derive(Debug, Clone)]
pub struct PasswordPrompt {
    pub label: String,
    pub input: String,
    pub mask_char: char,
    pub show_length: bool,
}

impl PasswordPrompt {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            input: String::new(),
            mask_char: '*',
            show_length: false,
        }
    }

    pub fn with_mask(mut self, ch: char) -> Self {
        self.mask_char = ch;
        self
    }

    pub fn push_char(&mut self, ch: char) {
        self.input.push(ch);
    }

    pub fn backspace(&mut self) {
        self.input.pop();
    }

    pub fn value(&self) -> &str {
        &self.input
    }

    pub fn masked(&self) -> String {
        self.mask_char.to_string().repeat(self.input.len())
    }

    pub fn render(&self) -> String {
        format!("? {}: {}", self.label, self.masked())
    }
}

// ── Confirm Prompt ──

/// A yes/no confirmation prompt.
#[derive(Debug, Clone)]
pub struct ConfirmPrompt {
    pub label: String,
    pub default: Option<bool>,
    pub answer: Option<bool>,
}

impl ConfirmPrompt {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            default: None,
            answer: None,
        }
    }

    pub fn with_default(mut self, default: bool) -> Self {
        self.default = Some(default);
        self
    }

    /// Process a single character input.
    pub fn handle_char(&mut self, ch: char) {
        match ch.to_ascii_lowercase() {
            'y' => self.answer = Some(true),
            'n' => self.answer = Some(false),
            _ => {}
        }
    }

    /// Submit — returns the answer or default.
    pub fn submit(&self) -> Option<bool> {
        self.answer.or(self.default)
    }

    pub fn render(&self) -> String {
        let hint = match self.default {
            Some(true) => "(Y/n)",
            Some(false) => "(y/N)",
            None => "(y/n)",
        };
        let ans = self.answer.map(|a| if a { "Yes" } else { "No" }).unwrap_or("");
        format!("? {} {hint}: {ans}", self.label)
    }
}

// ── Select Prompt ──

/// Single-select list prompt.
#[derive(Debug, Clone)]
pub struct SelectPrompt {
    pub label: String,
    pub options: Vec<String>,
    pub cursor: usize,
    pub page_size: usize,
}

impl SelectPrompt {
    pub fn new(label: &str, options: Vec<String>) -> Self {
        Self {
            label: label.to_string(),
            cursor: 0,
            page_size: 10,
            options,
        }
    }

    pub fn with_page_size(mut self, size: usize) -> Self {
        self.page_size = size;
        self
    }

    pub fn move_up(&mut self) {
        if self.cursor > 0 {
            self.cursor -= 1;
        } else {
            self.cursor = self.options.len().saturating_sub(1);
        }
    }

    pub fn move_down(&mut self) {
        if self.cursor + 1 < self.options.len() {
            self.cursor += 1;
        } else {
            self.cursor = 0;
        }
    }

    pub fn selected(&self) -> Option<&str> {
        self.options.get(self.cursor).map(|s| s.as_str())
    }

    pub fn selected_index(&self) -> usize {
        self.cursor
    }

    /// Visible page range.
    fn page_range(&self) -> (usize, usize) {
        let start = if self.cursor >= self.page_size {
            self.cursor - self.page_size + 1
        } else {
            0
        };
        let end = (start + self.page_size).min(self.options.len());
        (start, end)
    }

    pub fn render(&self) -> String {
        let mut out = format!("? {}\n", self.label);
        let (start, end) = self.page_range();
        for i in start..end {
            let marker = if i == self.cursor { ">" } else { " " };
            out.push_str(&format!("  {marker} {}\n", self.options[i]));
        }
        if end < self.options.len() {
            out.push_str("  (more...)\n");
        }
        out
    }
}

// ── Multi-Select Prompt ──

/// Multi-select (checkbox) prompt.
#[derive(Debug, Clone)]
pub struct MultiSelectPrompt {
    pub label: String,
    pub options: Vec<String>,
    pub selected: Vec<bool>,
    pub cursor: usize,
    pub page_size: usize,
    pub min_selections: usize,
    pub max_selections: Option<usize>,
}

impl MultiSelectPrompt {
    pub fn new(label: &str, options: Vec<String>) -> Self {
        let len = options.len();
        Self {
            label: label.to_string(),
            selected: vec![false; len],
            cursor: 0,
            page_size: 10,
            options,
            min_selections: 0,
            max_selections: None,
        }
    }

    pub fn with_min(mut self, min: usize) -> Self {
        self.min_selections = min;
        self
    }

    pub fn with_max(mut self, max: usize) -> Self {
        self.max_selections = Some(max);
        self
    }

    pub fn move_up(&mut self) {
        if self.cursor > 0 { self.cursor -= 1; }
        else { self.cursor = self.options.len().saturating_sub(1); }
    }

    pub fn move_down(&mut self) {
        if self.cursor + 1 < self.options.len() { self.cursor += 1; }
        else { self.cursor = 0; }
    }

    /// Toggle the item at cursor.
    pub fn toggle(&mut self) {
        if self.cursor < self.selected.len() {
            let currently = self.selected[self.cursor];
            if currently {
                self.selected[self.cursor] = false;
            } else {
                // Check max.
                let count = self.selected.iter().filter(|&&s| s).count();
                if let Some(max) = self.max_selections {
                    if count >= max { return; }
                }
                self.selected[self.cursor] = true;
            }
        }
    }

    /// Select all.
    pub fn select_all(&mut self) {
        for s in &mut self.selected { *s = true; }
    }

    /// Deselect all.
    pub fn deselect_all(&mut self) {
        for s in &mut self.selected { *s = false; }
    }

    /// Number of selected items.
    pub fn selected_count(&self) -> usize {
        self.selected.iter().filter(|&&s| s).count()
    }

    /// Get all selected values.
    pub fn selected_values(&self) -> Vec<&str> {
        self.options.iter().zip(&self.selected)
            .filter(|(_, sel)| **sel)
            .map(|(opt, _)| opt.as_str())
            .collect()
    }

    /// Validate selection count.
    pub fn validate(&self) -> ValidationResult {
        let count = self.selected_count();
        if count < self.min_selections {
            return ValidationResult::Invalid(
                format!("Select at least {} item(s)", self.min_selections)
            );
        }
        if let Some(max) = self.max_selections {
            if count > max {
                return ValidationResult::Invalid(
                    format!("Select at most {max} item(s)")
                );
            }
        }
        ValidationResult::Valid
    }

    pub fn render(&self) -> String {
        let mut out = format!("? {}\n", self.label);
        let end = self.options.len().min(self.page_size);
        for i in 0..end {
            let cursor_mark = if i == self.cursor { ">" } else { " " };
            let check = if self.selected[i] { "[x]" } else { "[ ]" };
            out.push_str(&format!("  {cursor_mark} {check} {}\n", self.options[i]));
        }
        out
    }
}

// ── Number Prompt ──

/// Numeric input with range validation.
#[derive(Debug, Clone)]
pub struct NumberPrompt {
    pub label: String,
    pub input: String,
    pub default: Option<f64>,
    pub validator: RangeValidator,
    pub error: Option<String>,
    pub allow_float: bool,
}

impl NumberPrompt {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            input: String::new(),
            default: None,
            validator: RangeValidator::new(None, None),
            error: None,
            allow_float: true,
        }
    }

    pub fn with_default(mut self, val: f64) -> Self {
        self.default = Some(val);
        self
    }

    pub fn with_range(mut self, min: f64, max: f64) -> Self {
        self.validator = RangeValidator::new(Some(min), Some(max));
        self
    }

    pub fn integer_only(mut self) -> Self {
        self.allow_float = false;
        self
    }

    pub fn push_char(&mut self, ch: char) {
        if ch.is_ascii_digit() || (ch == '.' && self.allow_float) || (ch == '-' && self.input.is_empty()) {
            self.input.push(ch);
            self.error = None;
        }
    }

    pub fn backspace(&mut self) {
        self.input.pop();
    }

    pub fn submit(&mut self) -> Result<f64, String> {
        let text = if self.input.is_empty() {
            match self.default {
                Some(d) => return Ok(d),
                None => {
                    let err = "A number is required".to_string();
                    self.error = Some(err.clone());
                    return Err(err);
                }
            }
        } else {
            &self.input
        };

        match text.parse::<f64>() {
            Ok(val) => {
                match self.validator.validate(val) {
                    ValidationResult::Valid => Ok(val),
                    ValidationResult::Invalid(e) => {
                        self.error = Some(e.clone());
                        Err(e)
                    }
                }
            }
            Err(_) => {
                let err = "Invalid number".to_string();
                self.error = Some(err.clone());
                Err(err)
            }
        }
    }

    pub fn render(&self) -> String {
        let default_hint = self.default
            .map(|d| format!(" ({d})"))
            .unwrap_or_default();
        let err_line = self.error.as_ref()
            .map(|e| format!("\n\x1b[31m  {e}\x1b[0m"))
            .unwrap_or_default();
        format!("? {}{default_hint}: {}{err_line}", self.label, self.input)
    }
}

// ── Autocomplete ──

/// Simple autocomplete engine for text prompts.
#[derive(Debug, Clone)]
pub struct Autocomplete {
    pub suggestions: Vec<String>,
    pub filtered: Vec<String>,
    pub cursor: Option<usize>,
}

impl Autocomplete {
    pub fn new(suggestions: Vec<String>) -> Self {
        Self {
            filtered: suggestions.clone(),
            suggestions,
            cursor: None,
        }
    }

    /// Filter suggestions by prefix.
    pub fn filter(&mut self, prefix: &str) {
        let lower = prefix.to_ascii_lowercase();
        self.filtered = self.suggestions.iter()
            .filter(|s| s.to_ascii_lowercase().starts_with(&lower))
            .cloned()
            .collect();
        self.cursor = if self.filtered.is_empty() { None } else { Some(0) };
    }

    /// Move cursor down in filtered list.
    pub fn next(&mut self) {
        if let Some(c) = self.cursor.as_mut() {
            *c = (*c + 1) % self.filtered.len();
        }
    }

    /// Move cursor up in filtered list.
    pub fn prev(&mut self) {
        if let Some(c) = self.cursor.as_mut() {
            if *c == 0 {
                *c = self.filtered.len().saturating_sub(1);
            } else {
                *c -= 1;
            }
        }
    }

    /// Get the currently highlighted suggestion.
    pub fn current(&self) -> Option<&str> {
        self.cursor.and_then(|c| self.filtered.get(c).map(|s| s.as_str()))
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_prompt_default() {
        let mut p = TextPrompt::new("Name").with_default("Alice");
        assert_eq!(p.value(), Some("Alice".to_string()));
        p.set_input("Bob");
        assert_eq!(p.value(), Some("Bob".to_string()));
    }

    #[test]
    fn text_prompt_required() {
        let mut p = TextPrompt::new("Name").required();
        let result = p.submit();
        assert!(result.is_err());
        assert!(p.error.is_some());
    }

    #[test]
    fn text_prompt_push_backspace() {
        let mut p = TextPrompt::new("Q");
        p.push_char('h');
        p.push_char('i');
        assert_eq!(p.input, "hi");
        p.backspace();
        assert_eq!(p.input, "h");
    }

    #[test]
    fn password_masked() {
        let mut p = PasswordPrompt::new("Password");
        p.push_char('s');
        p.push_char('e');
        p.push_char('c');
        assert_eq!(p.value(), "sec");
        assert_eq!(p.masked(), "***");
    }

    #[test]
    fn password_custom_mask() {
        let mut p = PasswordPrompt::new("Pin").with_mask('●');
        p.push_char('1');
        p.push_char('2');
        assert_eq!(p.masked(), "●●");
    }

    #[test]
    fn confirm_yes() {
        let mut c = ConfirmPrompt::new("Proceed?");
        c.handle_char('y');
        assert_eq!(c.submit(), Some(true));
    }

    #[test]
    fn confirm_default() {
        let c = ConfirmPrompt::new("Proceed?").with_default(true);
        assert_eq!(c.submit(), Some(true));
    }

    #[test]
    fn confirm_no_answer_no_default() {
        let c = ConfirmPrompt::new("Proceed?");
        assert_eq!(c.submit(), None);
    }

    #[test]
    fn select_navigation() {
        let mut s = SelectPrompt::new("Color", vec!["Red".into(), "Green".into(), "Blue".into()]);
        assert_eq!(s.selected(), Some("Red"));
        s.move_down();
        assert_eq!(s.selected(), Some("Green"));
        s.move_down();
        s.move_down(); // wrap
        assert_eq!(s.selected(), Some("Red"));
        s.move_up(); // wrap to end
        assert_eq!(s.selected(), Some("Blue"));
    }

    #[test]
    fn multi_select_toggle() {
        let mut ms = MultiSelectPrompt::new("Features", vec!["A".into(), "B".into(), "C".into()]);
        ms.toggle(); // select A
        ms.move_down();
        ms.toggle(); // select B
        assert_eq!(ms.selected_count(), 2);
        assert_eq!(ms.selected_values(), vec!["A", "B"]);
    }

    #[test]
    fn multi_select_max() {
        let mut ms = MultiSelectPrompt::new("Pick", vec!["X".into(), "Y".into(), "Z".into()])
            .with_max(1);
        ms.toggle(); // select X
        ms.move_down();
        ms.toggle(); // try select Y — should fail
        assert_eq!(ms.selected_count(), 1);
        assert_eq!(ms.selected_values(), vec!["X"]);
    }

    #[test]
    fn multi_select_validate_min() {
        let ms = MultiSelectPrompt::new("Pick", vec!["A".into()]).with_min(1);
        assert_eq!(ms.validate(), ValidationResult::Invalid("Select at least 1 item(s)".into()));
    }

    #[test]
    fn number_prompt_range() {
        let mut np = NumberPrompt::new("Age").with_range(0.0, 150.0);
        np.input = "200".to_string();
        let result = np.submit();
        assert!(result.is_err());
    }

    #[test]
    fn number_prompt_valid() {
        let mut np = NumberPrompt::new("Score").with_range(0.0, 100.0);
        np.input = "42.5".to_string();
        assert_eq!(np.submit().unwrap(), 42.5);
    }

    #[test]
    fn number_prompt_default() {
        let mut np = NumberPrompt::new("Count").with_default(10.0);
        assert_eq!(np.submit().unwrap(), 10.0);
    }

    #[test]
    fn autocomplete_filter() {
        let mut ac = Autocomplete::new(vec!["apple".into(), "apricot".into(), "banana".into()]);
        ac.filter("ap");
        assert_eq!(ac.filtered.len(), 2);
        assert_eq!(ac.current(), Some("apple"));
        ac.next();
        assert_eq!(ac.current(), Some("apricot"));
    }

    #[test]
    fn autocomplete_no_match() {
        let mut ac = Autocomplete::new(vec!["alpha".into()]);
        ac.filter("zz");
        assert_eq!(ac.filtered.len(), 0);
        assert_eq!(ac.current(), None);
    }

    #[test]
    fn range_validator() {
        let v = RangeValidator::new(Some(1.0), Some(10.0));
        assert_eq!(v.validate(5.0), ValidationResult::Valid);
        assert!(matches!(v.validate(0.0), ValidationResult::Invalid(_)));
        assert!(matches!(v.validate(11.0), ValidationResult::Invalid(_)));
    }

    #[test]
    fn text_prompt_render() {
        let p = TextPrompt::new("Name").with_default("World");
        let out = p.render();
        assert!(out.contains("Name"));
        assert!(out.contains("(World)"));
    }

    #[test]
    fn select_render() {
        let s = SelectPrompt::new("Pick", vec!["One".into(), "Two".into()]);
        let out = s.render();
        assert!(out.contains("> One"));
        assert!(out.contains("  Two"));
    }
}
