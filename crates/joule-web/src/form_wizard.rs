//! Multi-step form wizard — step definitions, validation, navigation, progress.
//!
//! Replaces react-step-wizard, formik multi-step, and vue-form-wizard with a
//! pure-Rust state machine for multi-step forms with conditional steps,
//! validation gates, and summary generation.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Wizard errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WizardError {
    /// Step not found.
    StepNotFound(String),
    /// Validation failed on current step.
    ValidationFailed { step: String, errors: Vec<String> },
    /// Already on the first step.
    AlreadyAtStart,
    /// Already on the last step.
    AlreadyAtEnd,
    /// No steps defined.
    NoSteps,
    /// Cannot skip a required step.
    CannotSkip(String),
}

impl std::fmt::Display for WizardError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StepNotFound(s) => write!(f, "step not found: {s}"),
            Self::ValidationFailed { step, errors } => {
                write!(f, "validation failed on {step}: {}", errors.join("; "))
            }
            Self::AlreadyAtStart => write!(f, "already at first step"),
            Self::AlreadyAtEnd => write!(f, "already at last step"),
            Self::NoSteps => write!(f, "no steps defined"),
            Self::CannotSkip(s) => write!(f, "cannot skip required step: {s}"),
        }
    }
}

impl std::error::Error for WizardError {}

// ── Step Definition ─────────────────────────────────────────────

/// How to determine if a step should be shown.
#[derive(Debug, Clone)]
pub enum StepCondition {
    /// Always show.
    Always,
    /// Show only if a specific field has a specific value.
    FieldEquals { field: String, value: String },
    /// Show only if a specific field is non-empty.
    FieldPresent(String),
    /// Custom predicate.
    Custom(fn(&HashMap<String, String>) -> bool),
}

impl StepCondition {
    pub fn evaluate(&self, data: &HashMap<String, String>) -> bool {
        match self {
            StepCondition::Always => true,
            StepCondition::FieldEquals { field, value } => {
                data.get(field).map_or(false, |v| v == value)
            }
            StepCondition::FieldPresent(field) => {
                data.get(field).map_or(false, |v| !v.trim().is_empty())
            }
            StepCondition::Custom(f) => f(data),
        }
    }
}

/// A single step in the wizard.
#[derive(Debug, Clone)]
pub struct WizardStep {
    /// Unique step identifier.
    pub id: String,
    /// Display title.
    pub title: String,
    /// Description.
    pub description: String,
    /// Fields belonging to this step.
    pub fields: Vec<String>,
    /// Validation function: returns Ok(()) or Err(vec of error messages).
    pub validator: Option<fn(&HashMap<String, String>) -> Result<(), Vec<String>>>,
    /// Condition for showing this step.
    pub condition: StepCondition,
    /// Whether the step has been completed.
    pub completed: bool,
}

impl WizardStep {
    pub fn new(id: impl Into<String>, title: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            title: title.into(),
            description: String::new(),
            fields: Vec::new(),
            validator: None,
            condition: StepCondition::Always,
            completed: false,
        }
    }

    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = desc.into();
        self
    }

    pub fn fields(mut self, fields: Vec<&str>) -> Self {
        self.fields = fields.into_iter().map(|s| s.to_string()).collect();
        self
    }

    pub fn validator(
        mut self,
        v: fn(&HashMap<String, String>) -> Result<(), Vec<String>>,
    ) -> Self {
        self.validator = Some(v);
        self
    }

    pub fn condition(mut self, cond: StepCondition) -> Self {
        self.condition = cond;
        self
    }

    /// Validate this step's data.
    pub fn validate(&self, data: &HashMap<String, String>) -> Result<(), Vec<String>> {
        if let Some(v) = &self.validator {
            v(data)
        } else {
            Ok(())
        }
    }
}

// ── Progress ────────────────────────────────────────────────────

/// Wizard progress info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WizardProgress {
    pub current_step: usize,
    pub total_steps: usize,
    pub completed_steps: usize,
    pub percent: f64,
}

// ── Summary ─────────────────────────────────────────────────────

/// A summary entry for review before submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryEntry {
    pub step_title: String,
    pub fields: Vec<(String, String)>,
}

// ── Wizard ──────────────────────────────────────────────────────

/// Multi-step form wizard state machine.
#[derive(Debug, Clone)]
pub struct FormWizard {
    /// All defined steps (including conditional ones).
    pub steps: Vec<WizardStep>,
    /// Current step index into the *active* steps list.
    pub current: usize,
    /// Collected form data.
    pub data: HashMap<String, String>,
}

impl FormWizard {
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            current: 0,
            data: HashMap::new(),
        }
    }

    pub fn add_step(mut self, step: WizardStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Get the list of currently active (visible) steps.
    pub fn active_steps(&self) -> Vec<&WizardStep> {
        self.steps
            .iter()
            .filter(|s| s.condition.evaluate(&self.data))
            .collect()
    }

    /// Get the current active step.
    pub fn current_step(&self) -> Result<&WizardStep, WizardError> {
        self.active_steps()
            .get(self.current)
            .copied()
            .ok_or(WizardError::NoSteps)
    }

    /// Get the current step id.
    pub fn current_step_id(&self) -> Result<String, WizardError> {
        Ok(self.current_step()?.id.clone())
    }

    /// Set a field value.
    pub fn set_field(&mut self, field: &str, value: &str) {
        self.data.insert(field.to_string(), value.to_string());
    }

    /// Get a field value.
    pub fn get_field(&self, field: &str) -> Option<&str> {
        self.data.get(field).map(|s| s.as_str())
    }

    /// Validate the current step.
    pub fn validate_current(&self) -> Result<(), WizardError> {
        let step = self.current_step()?;
        step.validate(&self.data).map_err(|errors| {
            WizardError::ValidationFailed {
                step: step.id.clone(),
                errors,
            }
        })
    }

    /// Advance to the next step (validates current step first).
    pub fn next(&mut self) -> Result<(), WizardError> {
        let active = self.active_steps();
        if active.is_empty() {
            return Err(WizardError::NoSteps);
        }
        if self.current >= active.len() - 1 {
            return Err(WizardError::AlreadyAtEnd);
        }

        // Validate current step
        let step_id = active[self.current].id.clone();
        let step = active[self.current];
        step.validate(&self.data).map_err(|errors| {
            WizardError::ValidationFailed {
                step: step_id.clone(),
                errors,
            }
        })?;

        // Mark as completed
        self.mark_completed(&step_id);

        self.current += 1;
        Ok(())
    }

    /// Go back to the previous step.
    pub fn prev(&mut self) -> Result<(), WizardError> {
        if self.current == 0 {
            return Err(WizardError::AlreadyAtStart);
        }
        self.current -= 1;
        Ok(())
    }

    /// Go to a specific step by id (does not validate).
    pub fn goto(&mut self, step_id: &str) -> Result<(), WizardError> {
        let active = self.active_steps();
        let pos = active
            .iter()
            .position(|s| s.id == step_id)
            .ok_or_else(|| WizardError::StepNotFound(step_id.to_string()))?;
        self.current = pos;
        Ok(())
    }

    /// Get progress info.
    pub fn progress(&self) -> WizardProgress {
        let active = self.active_steps();
        let total = active.len();
        let completed = active.iter().filter(|s| s.completed).count();
        let percent = if total == 0 {
            0.0
        } else {
            (completed as f64 / total as f64) * 100.0
        };

        WizardProgress {
            current_step: self.current,
            total_steps: total,
            completed_steps: completed,
            percent,
        }
    }

    /// Generate a summary of all completed steps.
    pub fn summary(&self) -> Vec<SummaryEntry> {
        let mut entries = Vec::new();

        for step in self.active_steps() {
            let fields: Vec<(String, String)> = step
                .fields
                .iter()
                .filter_map(|f| {
                    self.data
                        .get(f)
                        .map(|v| (f.clone(), v.clone()))
                })
                .collect();

            if !fields.is_empty() {
                entries.push(SummaryEntry {
                    step_title: step.title.clone(),
                    fields,
                });
            }
        }

        entries
    }

    /// Check if we're on the last active step.
    pub fn is_last_step(&self) -> bool {
        let active = self.active_steps();
        !active.is_empty() && self.current == active.len() - 1
    }

    /// Check if we're on the first step.
    pub fn is_first_step(&self) -> bool {
        self.current == 0
    }

    fn mark_completed(&mut self, step_id: &str) {
        for step in &mut self.steps {
            if step.id == step_id {
                step.completed = true;
                break;
            }
        }
    }
}

impl Default for FormWizard {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn basic_wizard() -> FormWizard {
        FormWizard::new()
            .add_step(
                WizardStep::new("personal", "Personal Info")
                    .fields(vec!["name", "email"])
                    .validator(|data| {
                        let name = data.get("name").map(|s| s.as_str()).unwrap_or("");
                        if name.is_empty() {
                            Err(vec!["name is required".into()])
                        } else {
                            Ok(())
                        }
                    }),
            )
            .add_step(
                WizardStep::new("address", "Address")
                    .fields(vec!["street", "city"]),
            )
            .add_step(
                WizardStep::new("confirm", "Confirm")
                    .description("Review your information"),
            )
    }

    #[test]
    fn starts_at_first_step() {
        let w = basic_wizard();
        assert_eq!(w.current_step_id().unwrap(), "personal");
        assert!(w.is_first_step());
    }

    #[test]
    fn next_validates() {
        let mut w = basic_wizard();
        // Should fail because name is empty.
        assert!(w.next().is_err());

        w.set_field("name", "Alice");
        assert!(w.next().is_ok());
        assert_eq!(w.current_step_id().unwrap(), "address");
    }

    #[test]
    fn prev_navigation() {
        let mut w = basic_wizard();
        w.set_field("name", "Alice");
        w.next().unwrap();
        assert_eq!(w.current_step_id().unwrap(), "address");
        w.prev().unwrap();
        assert_eq!(w.current_step_id().unwrap(), "personal");
    }

    #[test]
    fn prev_at_start_errors() {
        let mut w = basic_wizard();
        assert!(w.prev().is_err());
    }

    #[test]
    fn next_at_end_errors() {
        let mut w = basic_wizard();
        w.set_field("name", "Alice");
        w.next().unwrap(); // → address
        w.next().unwrap(); // → confirm
        assert!(w.next().is_err());
        assert!(w.is_last_step());
    }

    #[test]
    fn goto_step() {
        let mut w = basic_wizard();
        w.goto("confirm").unwrap();
        assert_eq!(w.current_step_id().unwrap(), "confirm");
    }

    #[test]
    fn goto_unknown_step() {
        let mut w = basic_wizard();
        assert!(w.goto("nonexistent").is_err());
    }

    #[test]
    fn progress_tracking() {
        let mut w = basic_wizard();
        let p = w.progress();
        assert_eq!(p.total_steps, 3);
        assert_eq!(p.completed_steps, 0);

        w.set_field("name", "Alice");
        w.next().unwrap();
        let p = w.progress();
        assert_eq!(p.completed_steps, 1);
        assert!((p.percent - 33.333).abs() < 1.0);
    }

    #[test]
    fn conditional_step() {
        let mut w = FormWizard::new()
            .add_step(WizardStep::new("type", "Account Type").fields(vec!["acct_type"]))
            .add_step(
                WizardStep::new("business", "Business Info")
                    .fields(vec!["company"])
                    .condition(StepCondition::FieldEquals {
                        field: "acct_type".into(),
                        value: "business".into(),
                    }),
            )
            .add_step(WizardStep::new("done", "Done"));

        // With "personal", business step is skipped.
        w.set_field("acct_type", "personal");
        assert_eq!(w.active_steps().len(), 2); // type + done

        // With "business", all three are active.
        w.set_field("acct_type", "business");
        assert_eq!(w.active_steps().len(), 3);
    }

    #[test]
    fn summary_generation() {
        let mut w = basic_wizard();
        w.set_field("name", "Alice");
        w.set_field("email", "alice@example.com");
        w.set_field("street", "123 Main St");

        let summary = w.summary();
        assert_eq!(summary.len(), 2); // personal + address have data
        assert_eq!(summary[0].step_title, "Personal Info");
        assert_eq!(summary[0].fields.len(), 2);
    }

    #[test]
    fn field_get_set() {
        let mut w = basic_wizard();
        assert!(w.get_field("name").is_none());
        w.set_field("name", "Bob");
        assert_eq!(w.get_field("name"), Some("Bob"));
    }

    #[test]
    fn empty_wizard() {
        let w = FormWizard::new();
        assert!(w.current_step().is_err());
        assert_eq!(w.progress().total_steps, 0);
    }

    #[test]
    fn step_condition_field_present() {
        let cond = StepCondition::FieldPresent("phone".into());
        let mut data = HashMap::new();
        assert!(!cond.evaluate(&data));
        data.insert("phone".into(), "555-1234".into());
        assert!(cond.evaluate(&data));
    }

    #[test]
    fn validate_current_step() {
        let mut w = basic_wizard();
        assert!(w.validate_current().is_err());
        w.set_field("name", "Alice");
        assert!(w.validate_current().is_ok());
    }
}
