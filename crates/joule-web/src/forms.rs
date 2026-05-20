//! Type-safe form state management and validation.
//!
//! Replaces React Hook Form / Formik with a pure-Rust form engine featuring
//! declarative validation rules, dirty/touched tracking, and structured submission.

use std::collections::HashMap;

// ── Validation Rules ──

/// A validation rule that can be applied to a form field.
pub enum ValidationRule {
    Required,
    MinLength(usize),
    MaxLength(usize),
    /// Regex pattern as a string (validated without regex crate using simple matchers).
    Pattern(String),
    Email,
    Min(f64),
    Max(f64),
    Range(f64, f64),
    Custom {
        name: String,
        message: String,
        validator: Box<dyn Fn(&str) -> bool>,
    },
}

impl ValidationRule {
    /// Validate a value against this rule, returning an error message if invalid.
    fn validate(&self, field_name: &str, value: &str) -> Option<String> {
        match self {
            ValidationRule::Required => {
                if value.trim().is_empty() {
                    Some(format!("{field_name} is required"))
                } else {
                    None
                }
            }
            ValidationRule::MinLength(n) => {
                if value.len() < *n {
                    Some(format!(
                        "{field_name} must be at least {n} characters"
                    ))
                } else {
                    None
                }
            }
            ValidationRule::MaxLength(n) => {
                if value.len() > *n {
                    Some(format!(
                        "{field_name} must be at most {n} characters"
                    ))
                } else {
                    None
                }
            }
            ValidationRule::Pattern(pattern) => {
                // Simple pattern check — full regex would need the `regex` crate.
                // We do a basic contains check for the pattern string.
                if !simple_pattern_match(pattern, value) {
                    Some(format!(
                        "{field_name} does not match the required pattern"
                    ))
                } else {
                    None
                }
            }
            ValidationRule::Email => {
                if !is_valid_email(value) {
                    Some(format!(
                        "{field_name} must be a valid email address"
                    ))
                } else {
                    None
                }
            }
            ValidationRule::Min(min) => {
                if let Ok(v) = value.parse::<f64>() {
                    if v < *min {
                        Some(format!("{field_name} must be at least {min}"))
                    } else {
                        None
                    }
                } else {
                    Some(format!("{field_name} must be a number"))
                }
            }
            ValidationRule::Max(max) => {
                if let Ok(v) = value.parse::<f64>() {
                    if v > *max {
                        Some(format!("{field_name} must be at most {max}"))
                    } else {
                        None
                    }
                } else {
                    Some(format!("{field_name} must be a number"))
                }
            }
            ValidationRule::Range(min, max) => {
                if let Ok(v) = value.parse::<f64>() {
                    if v < *min || v > *max {
                        Some(format!(
                            "{field_name} must be between {min} and {max}"
                        ))
                    } else {
                        None
                    }
                } else {
                    Some(format!("{field_name} must be a number"))
                }
            }
            ValidationRule::Custom {
                name: _,
                message,
                validator,
            } => {
                if !validator(value) {
                    Some(message.clone())
                } else {
                    None
                }
            }
        }
    }
}

/// Simple email validation (checks for `@` with text on both sides and a dot after `@`).
fn is_valid_email(value: &str) -> bool {
    let parts: Vec<&str> = value.split('@').collect();
    if parts.len() != 2 {
        return false;
    }
    let local = parts[0];
    let domain = parts[1];
    !local.is_empty() && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.')
}

/// Simple pattern matching — checks if value contains the pattern string.
/// For production use, this would integrate with a regex engine.
fn simple_pattern_match(pattern: &str, value: &str) -> bool {
    value.contains(pattern)
}

// ── Field State ──

/// State of an individual form field.
pub struct FieldState {
    pub name: String,
    pub value: String,
    pub touched: bool,
    pub dirty: bool,
    pub errors: Vec<String>,
    pub rules: Vec<ValidationRule>,
    initial_value: String,
}

impl FieldState {
    /// Create a new field with a name and initial value.
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            value: String::new(),
            touched: false,
            dirty: false,
            errors: Vec::new(),
            rules: Vec::new(),
            initial_value: String::new(),
        }
    }

    /// Run all validation rules on the current value, populating errors.
    /// Returns `true` if all rules pass.
    pub fn validate(&mut self) -> bool {
        self.errors.clear();
        for rule in &self.rules {
            if let Some(err) = rule.validate(&self.name, &self.value) {
                self.errors.push(err);
            }
        }
        self.errors.is_empty()
    }

    /// Set the field value and mark dirty.
    pub fn set_value(&mut self, value: &str) {
        self.value = value.to_string();
        self.dirty = self.value != self.initial_value;
    }

    /// Mark the field as touched (user has interacted with it).
    pub fn touch(&mut self) {
        self.touched = true;
    }

    /// Reset the field to an initial value.
    pub fn reset(&mut self, initial: &str) {
        self.initial_value = initial.to_string();
        self.value = initial.to_string();
        self.touched = false;
        self.dirty = false;
        self.errors.clear();
    }
}

// ── Form State ──

/// Manages the state of an entire form with multiple fields.
pub struct FormState {
    pub fields: HashMap<String, FieldState>,
    pub submitted: bool,
    pub submitting: bool,
    pub submit_count: u32,
}

impl FormState {
    pub fn new() -> Self {
        Self {
            fields: HashMap::new(),
            submitted: false,
            submitting: false,
            submit_count: 0,
        }
    }

    /// Register a field with validation rules. Returns a mutable reference to the field.
    pub fn register(&mut self, name: &str, rules: Vec<ValidationRule>) -> &mut FieldState {
        let field = self.fields.entry(name.to_string()).or_insert_with(|| {
            FieldState::new(name)
        });
        field.rules = rules;
        field
    }

    /// Set the value of a named field.
    pub fn set_value(&mut self, name: &str, value: &str) {
        if let Some(field) = self.fields.get_mut(name) {
            field.set_value(value);
        }
    }

    /// Validate all fields. Returns `true` if the entire form is valid.
    pub fn validate_all(&mut self) -> bool {
        let mut all_valid = true;
        // Collect keys to avoid borrow issues.
        let keys: Vec<String> = self.fields.keys().cloned().collect();
        for key in keys {
            if let Some(field) = self.fields.get_mut(&key) {
                if !field.validate() {
                    all_valid = false;
                }
            }
        }
        all_valid
    }

    /// Check if all fields currently have no errors.
    pub fn is_valid(&self) -> bool {
        self.fields.values().all(|f| f.errors.is_empty())
    }

    /// Check if any field has been modified from its initial value.
    pub fn is_dirty(&self) -> bool {
        self.fields.values().any(|f| f.dirty)
    }

    /// Get all field errors.
    pub fn errors(&self) -> HashMap<&str, Vec<String>> {
        let mut map = HashMap::new();
        for (name, field) in &self.fields {
            if !field.errors.is_empty() {
                map.insert(name.as_str(), field.errors.clone());
            }
        }
        map
    }

    /// Get all field values.
    pub fn values(&self) -> HashMap<&str, &str> {
        self.fields
            .iter()
            .map(|(k, v)| (k.as_str(), v.value.as_str()))
            .collect()
    }

    /// Reset all fields to empty initial values.
    pub fn reset(&mut self) {
        for field in self.fields.values_mut() {
            field.reset("");
        }
        self.submitted = false;
        self.submitting = false;
    }

    /// Submit the form. Validates all fields first.
    /// On success returns field values; on failure returns field errors.
    pub fn submit(&mut self) -> Result<HashMap<String, String>, HashMap<String, Vec<String>>> {
        self.submit_count += 1;
        self.submitted = true;

        if self.validate_all() {
            Ok(self
                .fields
                .iter()
                .map(|(k, v)| (k.clone(), v.value.clone()))
                .collect())
        } else {
            Err(self
                .fields
                .iter()
                .filter(|(_, v)| !v.errors.is_empty())
                .map(|(k, v)| (k.clone(), v.errors.clone()))
                .collect())
        }
    }
}

impl Default for FormState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_required_validation() {
        let mut form = FormState::new();
        form.register("name", vec![ValidationRule::Required]);
        form.set_value("name", "");
        assert!(!form.validate_all());
        let errs = form.errors();
        assert!(errs.get("name").unwrap()[0].contains("required"));
    }

    #[test]
    fn test_required_passes() {
        let mut form = FormState::new();
        form.register("name", vec![ValidationRule::Required]);
        form.set_value("name", "Alice");
        assert!(form.validate_all());
        assert!(form.is_valid());
    }

    #[test]
    fn test_min_length() {
        let mut form = FormState::new();
        form.register("password", vec![ValidationRule::MinLength(8)]);
        form.set_value("password", "short");
        assert!(!form.validate_all());
        let errs = form.errors();
        assert!(errs.get("password").unwrap()[0].contains("at least 8"));
    }

    #[test]
    fn test_max_length() {
        let mut form = FormState::new();
        form.register("bio", vec![ValidationRule::MaxLength(5)]);
        form.set_value("bio", "too long text");
        assert!(!form.validate_all());
        let errs = form.errors();
        assert!(errs.get("bio").unwrap()[0].contains("at most 5"));
    }

    #[test]
    fn test_email_validation() {
        let mut form = FormState::new();
        form.register("email", vec![ValidationRule::Email]);

        form.set_value("email", "bad-email");
        assert!(!form.validate_all());

        form.set_value("email", "user@example.com");
        assert!(form.validate_all());
    }

    #[test]
    fn test_pattern_validation() {
        let mut form = FormState::new();
        form.register("code", vec![ValidationRule::Pattern("ABC".to_string())]);

        form.set_value("code", "xyz");
        assert!(!form.validate_all());

        form.set_value("code", "prefix-ABC-suffix");
        assert!(form.validate_all());
    }

    #[test]
    fn test_range_validation() {
        let mut form = FormState::new();
        form.register("age", vec![ValidationRule::Range(18.0, 99.0)]);

        form.set_value("age", "10");
        assert!(!form.validate_all());

        form.set_value("age", "25");
        assert!(form.validate_all());

        form.set_value("age", "100");
        assert!(!form.validate_all());
    }

    #[test]
    fn test_custom_validator() {
        let mut form = FormState::new();
        form.register(
            "username",
            vec![ValidationRule::Custom {
                name: "no_spaces".to_string(),
                message: "Username must not contain spaces".to_string(),
                validator: Box::new(|v| !v.contains(' ')),
            }],
        );

        form.set_value("username", "has space");
        assert!(!form.validate_all());

        form.set_value("username", "nospace");
        assert!(form.validate_all());
    }

    #[test]
    fn test_form_submit_valid() {
        let mut form = FormState::new();
        form.register("name", vec![ValidationRule::Required]);
        form.set_value("name", "Alice");
        let result = form.submit();
        assert!(result.is_ok());
        let vals = result.unwrap();
        assert_eq!(vals.get("name").unwrap(), "Alice");
        assert_eq!(form.submit_count, 1);
    }

    #[test]
    fn test_form_submit_with_errors() {
        let mut form = FormState::new();
        form.register("name", vec![ValidationRule::Required]);
        form.set_value("name", "");
        let result = form.submit();
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.contains_key("name"));
    }

    #[test]
    fn test_reset_clears_state() {
        let mut form = FormState::new();
        form.register("name", vec![ValidationRule::Required]);
        form.set_value("name", "Alice");
        form.submitted = true;
        form.reset();
        assert!(!form.submitted);
        assert_eq!(form.fields.get("name").unwrap().value, "");
        assert!(!form.fields.get("name").unwrap().dirty);
    }

    #[test]
    fn test_dirty_tracking() {
        let mut form = FormState::new();
        form.register("name", vec![]);
        assert!(!form.is_dirty());
        form.set_value("name", "changed");
        assert!(form.is_dirty());
    }

    #[test]
    fn test_touched_tracking() {
        let mut form = FormState::new();
        form.register("name", vec![]);
        assert!(!form.fields.get("name").unwrap().touched);
        form.fields.get_mut("name").unwrap().touch();
        assert!(form.fields.get("name").unwrap().touched);
    }

    #[test]
    fn test_multiple_field_errors() {
        let mut form = FormState::new();
        form.register(
            "password",
            vec![
                ValidationRule::Required,
                ValidationRule::MinLength(8),
            ],
        );
        form.set_value("password", "");
        form.validate_all();
        let errs = form.errors();
        // Empty string triggers both Required and MinLength.
        assert!(errs.get("password").unwrap().len() >= 2);
    }

    #[test]
    fn test_min_max_numeric() {
        let mut form = FormState::new();
        form.register("score", vec![ValidationRule::Min(0.0), ValidationRule::Max(100.0)]);
        form.set_value("score", "-5");
        assert!(!form.validate_all());

        form.set_value("score", "50");
        assert!(form.validate_all());

        form.set_value("score", "150");
        assert!(!form.validate_all());
    }
}
