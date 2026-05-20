//! Dynamic form engine — form schema definition (fields, types, validation),
//! conditional field visibility, multi-page forms, form submission, draft
//! save/resume, form versioning, and computed fields.
//!
//! Replaces JavaScript form libraries (Formik, React Hook Form, Yup) with a
//! pure-Rust form engine that validates and tracks form state.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

// ── Errors ──────────────────────────────────────────────────────

/// Form engine domain errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormError {
    /// Form not found.
    FormNotFound(String),
    /// Field not found.
    FieldNotFound(String),
    /// Validation failure with field-level messages.
    ValidationFailed(Vec<FieldValidationError>),
    /// Page index out of bounds.
    PageOutOfBounds { page: usize, total: usize },
    /// Submission not found.
    SubmissionNotFound(String),
    /// Draft not found.
    DraftNotFound(String),
    /// Form version conflict.
    VersionConflict { expected: u32, actual: u32 },
    /// Duplicate form ID.
    DuplicateForm(String),
    /// Computed field expression error.
    ComputeError { field: String, message: String },
}

impl std::fmt::Display for FormError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FormNotFound(id) => write!(f, "form not found: {id}"),
            Self::FieldNotFound(id) => write!(f, "field not found: {id}"),
            Self::ValidationFailed(errs) => {
                write!(f, "validation failed: {} error(s)", errs.len())
            }
            Self::PageOutOfBounds { page, total } => {
                write!(f, "page {page} out of bounds (total: {total})")
            }
            Self::SubmissionNotFound(id) => write!(f, "submission not found: {id}"),
            Self::DraftNotFound(id) => write!(f, "draft not found: {id}"),
            Self::VersionConflict { expected, actual } => {
                write!(f, "version conflict: expected {expected}, got {actual}")
            }
            Self::DuplicateForm(id) => write!(f, "duplicate form: {id}"),
            Self::ComputeError { field, message } => {
                write!(f, "compute error on field {field}: {message}")
            }
        }
    }
}

impl std::error::Error for FormError {}

/// A single field-level validation error.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldValidationError {
    pub field_id: String,
    pub message: String,
}

// ── Enums ───────────────────────────────────────────────────────

/// Supported field types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldType {
    Text,
    TextArea,
    Number,
    Email,
    Phone,
    Date,
    DateTime,
    Boolean,
    Select { options: Vec<SelectOption> },
    MultiSelect { options: Vec<SelectOption> },
    Radio { options: Vec<SelectOption> },
    Checkbox,
    File,
    Hidden,
    Computed { expression: ComputeExpression },
}

/// Select option.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SelectOption {
    pub label: String,
    pub value: String,
}

/// Compute expression for computed fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComputeExpression {
    /// Concatenate values of named fields with a separator.
    Concat { fields: Vec<String>, separator: String },
    /// Sum numeric fields.
    Sum(Vec<String>),
    /// Subtract second field from first.
    Subtract(String, String),
    /// Multiply numeric fields.
    Multiply(Vec<String>),
    /// Static value.
    Static(String),
}

/// Visibility condition for a field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VisibilityCondition {
    /// Always visible.
    Always,
    /// Visible when another field equals a specific value.
    WhenEquals { field_id: String, value: String },
    /// Visible when another field is not empty.
    WhenNotEmpty { field_id: String },
    /// Visible when a boolean field is true.
    WhenTrue { field_id: String },
    /// Visible when all sub-conditions hold.
    And(Vec<VisibilityCondition>),
    /// Visible when any sub-condition holds.
    Or(Vec<VisibilityCondition>),
}

/// Validation rules for a field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValidationRules {
    pub required: bool,
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub pattern: Option<String>,
    pub custom_message: Option<String>,
}

impl Default for ValidationRules {
    fn default() -> Self {
        Self {
            required: false,
            min_length: None,
            max_length: None,
            min_value: None,
            max_value: None,
            pattern: None,
            custom_message: None,
        }
    }
}

/// Submission status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubmissionStatus {
    Draft,
    Submitted,
    Validated,
    Rejected,
}

// ── Data Structures ─────────────────────────────────────────────

/// Definition of a single form field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDef {
    pub id: String,
    pub label: String,
    pub field_type: FieldType,
    pub validation: ValidationRules,
    pub visibility: VisibilityCondition,
    pub placeholder: Option<String>,
    pub default_value: Option<String>,
    pub help_text: Option<String>,
    pub page: usize,
    pub order: u32,
}

/// A form schema (versioned).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormSchema {
    pub id: String,
    pub name: String,
    pub description: Option<String>,
    pub fields: Vec<FieldDef>,
    pub total_pages: usize,
    pub version: u32,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// A form submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormSubmission {
    pub id: String,
    pub form_id: String,
    pub form_version: u32,
    pub values: HashMap<String, String>,
    pub status: SubmissionStatus,
    pub submitted_at: Option<DateTime<Utc>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub validation_errors: Vec<FieldValidationError>,
}

/// A draft form (save/resume).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormDraft {
    pub id: String,
    pub form_id: String,
    pub form_version: u32,
    pub values: HashMap<String, String>,
    pub current_page: usize,
    pub saved_at: DateTime<Utc>,
}

// ── Engine ──────────────────────────────────────────────────────

/// Dynamic form engine.
pub struct FormEngine {
    forms: HashMap<String, FormSchema>,
    submissions: Vec<FormSubmission>,
    drafts: HashMap<String, FormDraft>,
    next_submission_id: u64,
    next_draft_id: u64,
}

impl FormEngine {
    pub fn new() -> Self {
        Self {
            forms: HashMap::new(),
            submissions: Vec::new(),
            drafts: HashMap::new(),
            next_submission_id: 1,
            next_draft_id: 1,
        }
    }

    // ── Form Schema Management ──────────────────────────────────

    /// Register a form schema.
    pub fn register_form(&mut self, schema: FormSchema) -> Result<(), FormError> {
        if self.forms.contains_key(&schema.id) {
            return Err(FormError::DuplicateForm(schema.id.clone()));
        }
        self.forms.insert(schema.id.clone(), schema);
        Ok(())
    }

    /// Update a form schema (bumps version).
    pub fn update_form(
        &mut self,
        form_id: &str,
        fields: Vec<FieldDef>,
        total_pages: usize,
    ) -> Result<u32, FormError> {
        let schema = self
            .forms
            .get_mut(form_id)
            .ok_or_else(|| FormError::FormNotFound(form_id.to_string()))?;
        schema.version += 1;
        schema.fields = fields;
        schema.total_pages = total_pages;
        schema.updated_at = Utc::now();
        Ok(schema.version)
    }

    /// Get a form schema.
    pub fn get_form(&self, id: &str) -> Option<&FormSchema> {
        self.forms.get(id)
    }

    // ── Field Visibility ────────────────────────────────────────

    /// Evaluate whether a field is visible given current form values.
    pub fn is_field_visible(
        &self,
        condition: &VisibilityCondition,
        values: &HashMap<String, String>,
    ) -> bool {
        evaluate_visibility(condition, values)
    }

    /// Get visible fields for a given page.
    pub fn visible_fields_for_page(
        &self,
        form_id: &str,
        page: usize,
        values: &HashMap<String, String>,
    ) -> Result<Vec<&FieldDef>, FormError> {
        let schema = self
            .forms
            .get(form_id)
            .ok_or_else(|| FormError::FormNotFound(form_id.to_string()))?;

        if page >= schema.total_pages {
            return Err(FormError::PageOutOfBounds {
                page,
                total: schema.total_pages,
            });
        }

        let mut fields: Vec<&FieldDef> = schema
            .fields
            .iter()
            .filter(|f| f.page == page && evaluate_visibility(&f.visibility, values))
            .collect();
        fields.sort_by_key(|f| f.order);
        Ok(fields)
    }

    // ── Computed Fields ─────────────────────────────────────────

    /// Compute all computed field values.
    pub fn compute_fields(
        &self,
        form_id: &str,
        values: &HashMap<String, String>,
    ) -> Result<HashMap<String, String>, FormError> {
        let schema = self
            .forms
            .get(form_id)
            .ok_or_else(|| FormError::FormNotFound(form_id.to_string()))?;

        let mut computed = HashMap::new();
        for field in &schema.fields {
            if let FieldType::Computed { expression } = &field.field_type {
                let val = evaluate_compute(expression, values).map_err(|msg| {
                    FormError::ComputeError {
                        field: field.id.clone(),
                        message: msg,
                    }
                })?;
                computed.insert(field.id.clone(), val);
            }
        }
        Ok(computed)
    }

    // ── Validation ──────────────────────────────────────────────

    /// Validate all visible fields.
    pub fn validate(
        &self,
        form_id: &str,
        values: &HashMap<String, String>,
    ) -> Result<(), FormError> {
        let schema = self
            .forms
            .get(form_id)
            .ok_or_else(|| FormError::FormNotFound(form_id.to_string()))?;

        let mut errors = Vec::new();

        for field in &schema.fields {
            // Skip hidden/non-visible fields.
            if !evaluate_visibility(&field.visibility, values) {
                continue;
            }
            // Skip computed fields (they are derived).
            if matches!(field.field_type, FieldType::Computed { .. }) {
                continue;
            }

            let value = values.get(&field.id).map(|v| v.as_str()).unwrap_or("");

            if let Some(err) = validate_field(field, value) {
                errors.push(err);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(FormError::ValidationFailed(errors))
        }
    }

    /// Validate a single page of fields.
    pub fn validate_page(
        &self,
        form_id: &str,
        page: usize,
        values: &HashMap<String, String>,
    ) -> Result<(), FormError> {
        let schema = self
            .forms
            .get(form_id)
            .ok_or_else(|| FormError::FormNotFound(form_id.to_string()))?;

        if page >= schema.total_pages {
            return Err(FormError::PageOutOfBounds {
                page,
                total: schema.total_pages,
            });
        }

        let mut errors = Vec::new();
        for field in &schema.fields {
            if field.page != page {
                continue;
            }
            if !evaluate_visibility(&field.visibility, values) {
                continue;
            }
            if matches!(field.field_type, FieldType::Computed { .. }) {
                continue;
            }
            let value = values.get(&field.id).map(|v| v.as_str()).unwrap_or("");
            if let Some(err) = validate_field(field, value) {
                errors.push(err);
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(FormError::ValidationFailed(errors))
        }
    }

    // ── Submission ──────────────────────────────────────────────

    /// Submit a form.
    pub fn submit(
        &mut self,
        form_id: &str,
        values: HashMap<String, String>,
    ) -> Result<String, FormError> {
        // Validate first.
        self.validate(form_id, &values)?;

        let schema = self.forms.get(form_id).unwrap();
        let now = Utc::now();
        let id = format!("sub-{}", self.next_submission_id);
        self.next_submission_id += 1;

        let submission = FormSubmission {
            id: id.clone(),
            form_id: form_id.to_string(),
            form_version: schema.version,
            values,
            status: SubmissionStatus::Submitted,
            submitted_at: Some(now),
            created_at: now,
            updated_at: now,
            validation_errors: Vec::new(),
        };
        self.submissions.push(submission);
        Ok(id)
    }

    /// Get submission by ID.
    pub fn get_submission(&self, id: &str) -> Option<&FormSubmission> {
        self.submissions.iter().find(|s| s.id == id)
    }

    /// List submissions for a form.
    pub fn submissions_for_form(&self, form_id: &str) -> Vec<&FormSubmission> {
        self.submissions
            .iter()
            .filter(|s| s.form_id == form_id)
            .collect()
    }

    // ── Draft Save/Resume ───────────────────────────────────────

    /// Save a draft.
    pub fn save_draft(
        &mut self,
        form_id: &str,
        values: HashMap<String, String>,
        current_page: usize,
    ) -> Result<String, FormError> {
        let schema = self
            .forms
            .get(form_id)
            .ok_or_else(|| FormError::FormNotFound(form_id.to_string()))?;

        let id = format!("draft-{}", self.next_draft_id);
        self.next_draft_id += 1;

        let draft = FormDraft {
            id: id.clone(),
            form_id: form_id.to_string(),
            form_version: schema.version,
            values,
            current_page,
            saved_at: Utc::now(),
        };
        self.drafts.insert(id.clone(), draft);
        Ok(id)
    }

    /// Resume a draft.
    pub fn get_draft(&self, draft_id: &str) -> Result<&FormDraft, FormError> {
        self.drafts
            .get(draft_id)
            .ok_or_else(|| FormError::DraftNotFound(draft_id.to_string()))
    }

    /// Delete a draft.
    pub fn delete_draft(&mut self, draft_id: &str) -> Result<(), FormError> {
        self.drafts
            .remove(draft_id)
            .ok_or_else(|| FormError::DraftNotFound(draft_id.to_string()))?;
        Ok(())
    }

    /// Submit from a draft.
    pub fn submit_draft(&mut self, draft_id: &str) -> Result<String, FormError> {
        let draft = self
            .drafts
            .remove(draft_id)
            .ok_or_else(|| FormError::DraftNotFound(draft_id.to_string()))?;

        // Check version compatibility.
        let schema = self
            .forms
            .get(&draft.form_id)
            .ok_or_else(|| FormError::FormNotFound(draft.form_id.clone()))?;

        if draft.form_version != schema.version {
            return Err(FormError::VersionConflict {
                expected: schema.version,
                actual: draft.form_version,
            });
        }

        let form_id = draft.form_id.clone();
        self.submit(&form_id, draft.values)
    }

    /// Get form field definitions as JSON (for client rendering).
    pub fn schema_to_json(&self, form_id: &str) -> Result<Value, FormError> {
        let schema = self
            .forms
            .get(form_id)
            .ok_or_else(|| FormError::FormNotFound(form_id.to_string()))?;
        serde_json::to_value(schema).map_err(|e| FormError::ComputeError {
            field: String::new(),
            message: e.to_string(),
        })
    }
}

impl Default for FormEngine {
    fn default() -> Self {
        Self::new()
    }
}

// ── Visibility Evaluation ───────────────────────────────────────

fn evaluate_visibility(condition: &VisibilityCondition, values: &HashMap<String, String>) -> bool {
    match condition {
        VisibilityCondition::Always => true,
        VisibilityCondition::WhenEquals { field_id, value } => {
            values.get(field_id).map_or(false, |v| v == value)
        }
        VisibilityCondition::WhenNotEmpty { field_id } => {
            values.get(field_id).map_or(false, |v| !v.is_empty())
        }
        VisibilityCondition::WhenTrue { field_id } => {
            values
                .get(field_id)
                .map_or(false, |v| v == "true" || v == "1")
        }
        VisibilityCondition::And(conditions) => {
            conditions.iter().all(|c| evaluate_visibility(c, values))
        }
        VisibilityCondition::Or(conditions) => {
            conditions.iter().any(|c| evaluate_visibility(c, values))
        }
    }
}

// ── Compute Evaluation ──────────────────────────────────────────

fn evaluate_compute(
    expr: &ComputeExpression,
    values: &HashMap<String, String>,
) -> Result<String, String> {
    match expr {
        ComputeExpression::Concat { fields, separator } => {
            let parts: Vec<&str> = fields
                .iter()
                .filter_map(|f| values.get(f).map(|v| v.as_str()))
                .collect();
            Ok(parts.join(separator))
        }
        ComputeExpression::Sum(fields) => {
            let mut total: f64 = 0.0;
            for f in fields {
                if let Some(v) = values.get(f) {
                    total += v
                        .parse::<f64>()
                        .map_err(|_| format!("field {f} is not a number: {v}"))?;
                }
            }
            Ok(format_number(total))
        }
        ComputeExpression::Subtract(a, b) => {
            let va = values
                .get(a)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            let vb = values
                .get(b)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(0.0);
            Ok(format_number(va - vb))
        }
        ComputeExpression::Multiply(fields) => {
            let mut result: f64 = 1.0;
            for f in fields {
                if let Some(v) = values.get(f) {
                    result *= v
                        .parse::<f64>()
                        .map_err(|_| format!("field {f} is not a number: {v}"))?;
                }
            }
            Ok(format_number(result))
        }
        ComputeExpression::Static(val) => Ok(val.clone()),
    }
}

fn format_number(n: f64) -> String {
    if n == n.floor() && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{n}")
    }
}

// ── Field Validation ────────────────────────────────────────────

fn validate_field(field: &FieldDef, value: &str) -> Option<FieldValidationError> {
    let rules = &field.validation;

    // Required check.
    if rules.required && value.is_empty() {
        return Some(FieldValidationError {
            field_id: field.id.clone(),
            message: rules
                .custom_message
                .clone()
                .unwrap_or_else(|| format!("{} is required", field.label)),
        });
    }

    // If empty and not required, skip further validation.
    if value.is_empty() {
        return None;
    }

    // Min length.
    if let Some(min) = rules.min_length {
        if value.len() < min {
            return Some(FieldValidationError {
                field_id: field.id.clone(),
                message: format!("{} must be at least {min} characters", field.label),
            });
        }
    }

    // Max length.
    if let Some(max) = rules.max_length {
        if value.len() > max {
            return Some(FieldValidationError {
                field_id: field.id.clone(),
                message: format!("{} must be at most {max} characters", field.label),
            });
        }
    }

    // Numeric range.
    if let Some(min_val) = rules.min_value {
        if let Ok(n) = value.parse::<f64>() {
            if n < min_val {
                return Some(FieldValidationError {
                    field_id: field.id.clone(),
                    message: format!("{} must be at least {min_val}", field.label),
                });
            }
        }
    }

    if let Some(max_val) = rules.max_value {
        if let Ok(n) = value.parse::<f64>() {
            if n > max_val {
                return Some(FieldValidationError {
                    field_id: field.id.clone(),
                    message: format!("{} must be at most {max_val}", field.label),
                });
            }
        }
    }

    // Email basic check.
    if field.field_type == FieldType::Email && !value.contains('@') {
        return Some(FieldValidationError {
            field_id: field.id.clone(),
            message: format!("{} must be a valid email address", field.label),
        });
    }

    None
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn text_field(id: &str, required: bool, page: usize) -> FieldDef {
        FieldDef {
            id: id.to_string(),
            label: id.to_string(),
            field_type: FieldType::Text,
            validation: ValidationRules {
                required,
                ..Default::default()
            },
            visibility: VisibilityCondition::Always,
            placeholder: None,
            default_value: None,
            help_text: None,
            page,
            order: 0,
        }
    }

    fn make_schema(id: &str, fields: Vec<FieldDef>, pages: usize) -> FormSchema {
        FormSchema {
            id: id.to_string(),
            name: format!("Form {id}"),
            description: None,
            fields,
            total_pages: pages,
            version: 1,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn setup_engine() -> FormEngine {
        let mut engine = FormEngine::new();
        let schema = make_schema(
            "f1",
            vec![
                text_field("name", true, 0),
                text_field("email", false, 0),
                text_field("bio", false, 1),
            ],
            2,
        );
        engine.register_form(schema).unwrap();
        engine
    }

    #[test]
    fn test_register_form() {
        let engine = setup_engine();
        assert!(engine.get_form("f1").is_some());
    }

    #[test]
    fn test_duplicate_form() {
        let mut engine = setup_engine();
        let schema = make_schema("f1", vec![], 1);
        let err = engine.register_form(schema).unwrap_err();
        assert!(matches!(err, FormError::DuplicateForm(_)));
    }

    #[test]
    fn test_validate_required_field() {
        let engine = setup_engine();
        let values = HashMap::new();
        let err = engine.validate("f1", &values).unwrap_err();
        if let FormError::ValidationFailed(errs) = err {
            assert_eq!(errs.len(), 1);
            assert_eq!(errs[0].field_id, "name");
        } else {
            panic!("expected ValidationFailed");
        }
    }

    #[test]
    fn test_validate_pass() {
        let engine = setup_engine();
        let mut values = HashMap::new();
        values.insert("name".into(), "Alice".into());
        assert!(engine.validate("f1", &values).is_ok());
    }

    #[test]
    fn test_min_length_validation() {
        let mut engine = FormEngine::new();
        let mut field = text_field("name", false, 0);
        field.validation.min_length = Some(3);
        let schema = make_schema("f1", vec![field], 1);
        engine.register_form(schema).unwrap();

        let mut values = HashMap::new();
        values.insert("name".into(), "Ab".into());
        let err = engine.validate("f1", &values).unwrap_err();
        assert!(matches!(err, FormError::ValidationFailed(_)));
    }

    #[test]
    fn test_max_length_validation() {
        let mut engine = FormEngine::new();
        let mut field = text_field("name", false, 0);
        field.validation.max_length = Some(5);
        let schema = make_schema("f1", vec![field], 1);
        engine.register_form(schema).unwrap();

        let mut values = HashMap::new();
        values.insert("name".into(), "TooLongName".into());
        let err = engine.validate("f1", &values).unwrap_err();
        assert!(matches!(err, FormError::ValidationFailed(_)));
    }

    #[test]
    fn test_numeric_range_validation() {
        let mut engine = FormEngine::new();
        let field = FieldDef {
            id: "age".into(),
            label: "Age".into(),
            field_type: FieldType::Number,
            validation: ValidationRules {
                min_value: Some(0.0),
                max_value: Some(120.0),
                ..Default::default()
            },
            visibility: VisibilityCondition::Always,
            placeholder: None,
            default_value: None,
            help_text: None,
            page: 0,
            order: 0,
        };
        let schema = make_schema("f1", vec![field], 1);
        engine.register_form(schema).unwrap();

        let mut values = HashMap::new();
        values.insert("age".into(), "150".into());
        let err = engine.validate("f1", &values).unwrap_err();
        assert!(matches!(err, FormError::ValidationFailed(_)));
    }

    #[test]
    fn test_email_validation() {
        let mut engine = FormEngine::new();
        let field = FieldDef {
            id: "email".into(),
            label: "Email".into(),
            field_type: FieldType::Email,
            validation: ValidationRules::default(),
            visibility: VisibilityCondition::Always,
            placeholder: None,
            default_value: None,
            help_text: None,
            page: 0,
            order: 0,
        };
        let schema = make_schema("f1", vec![field], 1);
        engine.register_form(schema).unwrap();

        let mut values = HashMap::new();
        values.insert("email".into(), "notanemail".into());
        let err = engine.validate("f1", &values).unwrap_err();
        assert!(matches!(err, FormError::ValidationFailed(_)));
    }

    #[test]
    fn test_conditional_visibility() {
        let mut engine = FormEngine::new();
        let field1 = text_field("country", false, 0);
        let field2 = FieldDef {
            id: "state".into(),
            label: "State".into(),
            field_type: FieldType::Text,
            validation: ValidationRules {
                required: true,
                ..Default::default()
            },
            visibility: VisibilityCondition::WhenEquals {
                field_id: "country".into(),
                value: "US".into(),
            },
            placeholder: None,
            default_value: None,
            help_text: None,
            page: 0,
            order: 1,
        };
        let schema = make_schema("f1", vec![field1, field2], 1);
        engine.register_form(schema).unwrap();

        // Without country=US, state field is hidden, so validation passes.
        let mut values = HashMap::new();
        values.insert("country".into(), "UK".into());
        assert!(engine.validate("f1", &values).is_ok());

        // With country=US, state is visible and required.
        values.insert("country".into(), "US".into());
        let err = engine.validate("f1", &values).unwrap_err();
        assert!(matches!(err, FormError::ValidationFailed(_)));
    }

    #[test]
    fn test_multi_page_visible_fields() {
        let engine = setup_engine();
        let values = HashMap::new();
        let page0 = engine.visible_fields_for_page("f1", 0, &values).unwrap();
        assert_eq!(page0.len(), 2); // name, email
        let page1 = engine.visible_fields_for_page("f1", 1, &values).unwrap();
        assert_eq!(page1.len(), 1); // bio
    }

    #[test]
    fn test_page_out_of_bounds() {
        let engine = setup_engine();
        let err = engine
            .visible_fields_for_page("f1", 5, &HashMap::new())
            .unwrap_err();
        assert!(matches!(err, FormError::PageOutOfBounds { .. }));
    }

    #[test]
    fn test_computed_field_sum() {
        let mut engine = FormEngine::new();
        let f1 = text_field("price", false, 0);
        let f2 = text_field("qty", false, 0);
        let f3 = FieldDef {
            id: "total".into(),
            label: "Total".into(),
            field_type: FieldType::Computed {
                expression: ComputeExpression::Sum(vec!["price".into(), "qty".into()]),
            },
            validation: ValidationRules::default(),
            visibility: VisibilityCondition::Always,
            placeholder: None,
            default_value: None,
            help_text: None,
            page: 0,
            order: 2,
        };
        let schema = make_schema("f1", vec![f1, f2, f3], 1);
        engine.register_form(schema).unwrap();

        let mut values = HashMap::new();
        values.insert("price".into(), "10".into());
        values.insert("qty".into(), "5".into());
        let computed = engine.compute_fields("f1", &values).unwrap();
        assert_eq!(computed.get("total"), Some(&"15".to_string()));
    }

    #[test]
    fn test_computed_field_concat() {
        let mut engine = FormEngine::new();
        let f1 = text_field("first", false, 0);
        let f2 = text_field("last", false, 0);
        let f3 = FieldDef {
            id: "full_name".into(),
            label: "Full Name".into(),
            field_type: FieldType::Computed {
                expression: ComputeExpression::Concat {
                    fields: vec!["first".into(), "last".into()],
                    separator: " ".into(),
                },
            },
            validation: ValidationRules::default(),
            visibility: VisibilityCondition::Always,
            placeholder: None,
            default_value: None,
            help_text: None,
            page: 0,
            order: 2,
        };
        let schema = make_schema("f1", vec![f1, f2, f3], 1);
        engine.register_form(schema).unwrap();

        let mut values = HashMap::new();
        values.insert("first".into(), "Jane".into());
        values.insert("last".into(), "Doe".into());
        let computed = engine.compute_fields("f1", &values).unwrap();
        assert_eq!(computed.get("full_name"), Some(&"Jane Doe".to_string()));
    }

    #[test]
    fn test_submit_form() {
        let mut engine = setup_engine();
        let mut values = HashMap::new();
        values.insert("name".into(), "Alice".into());
        let id = engine.submit("f1", values).unwrap();
        let sub = engine.get_submission(&id).unwrap();
        assert_eq!(sub.status, SubmissionStatus::Submitted);
    }

    #[test]
    fn test_submit_fails_validation() {
        let mut engine = setup_engine();
        let values = HashMap::new();
        let err = engine.submit("f1", values).unwrap_err();
        assert!(matches!(err, FormError::ValidationFailed(_)));
    }

    #[test]
    fn test_draft_save_resume() {
        let mut engine = setup_engine();
        let mut values = HashMap::new();
        values.insert("name".into(), "Partial".into());
        let draft_id = engine.save_draft("f1", values, 0).unwrap();
        let draft = engine.get_draft(&draft_id).unwrap();
        assert_eq!(draft.current_page, 0);
        assert_eq!(draft.values.get("name"), Some(&"Partial".to_string()));
    }

    #[test]
    fn test_submit_from_draft() {
        let mut engine = setup_engine();
        let mut values = HashMap::new();
        values.insert("name".into(), "FromDraft".into());
        let draft_id = engine.save_draft("f1", values, 0).unwrap();
        let sub_id = engine.submit_draft(&draft_id).unwrap();
        let sub = engine.get_submission(&sub_id).unwrap();
        assert_eq!(sub.values.get("name"), Some(&"FromDraft".to_string()));
    }

    #[test]
    fn test_delete_draft() {
        let mut engine = setup_engine();
        let values = HashMap::new();
        let draft_id = engine.save_draft("f1", values, 0).unwrap();
        engine.delete_draft(&draft_id).unwrap();
        let err = engine.get_draft(&draft_id).unwrap_err();
        assert!(matches!(err, FormError::DraftNotFound(_)));
    }

    #[test]
    fn test_form_versioning() {
        let mut engine = setup_engine();
        let new_version = engine
            .update_form("f1", vec![text_field("name", true, 0)], 1)
            .unwrap();
        assert_eq!(new_version, 2);
    }

    #[test]
    fn test_version_conflict_on_draft_submit() {
        let mut engine = setup_engine();
        let mut values = HashMap::new();
        values.insert("name".into(), "Test".into());
        let draft_id = engine.save_draft("f1", values, 0).unwrap();

        // Update form version.
        engine
            .update_form("f1", vec![text_field("name", true, 0)], 1)
            .unwrap();

        let err = engine.submit_draft(&draft_id).unwrap_err();
        assert!(matches!(err, FormError::VersionConflict { .. }));
    }

    #[test]
    fn test_visibility_and_conditions() {
        let cond = VisibilityCondition::And(vec![
            VisibilityCondition::WhenNotEmpty {
                field_id: "a".into(),
            },
            VisibilityCondition::WhenTrue {
                field_id: "b".into(),
            },
        ]);
        let mut values = HashMap::new();
        assert!(!evaluate_visibility(&cond, &values));
        values.insert("a".into(), "x".into());
        values.insert("b".into(), "true".into());
        assert!(evaluate_visibility(&cond, &values));
    }

    #[test]
    fn test_or_visibility() {
        let cond = VisibilityCondition::Or(vec![
            VisibilityCondition::WhenEquals {
                field_id: "x".into(),
                value: "1".into(),
            },
            VisibilityCondition::WhenEquals {
                field_id: "y".into(),
                value: "2".into(),
            },
        ]);
        let mut values = HashMap::new();
        values.insert("y".into(), "2".into());
        assert!(evaluate_visibility(&cond, &values));
    }

    #[test]
    fn test_schema_to_json() {
        let engine = setup_engine();
        let json = engine.schema_to_json("f1").unwrap();
        assert!(json.is_object());
        assert_eq!(json["name"], "Form f1");
    }

    #[test]
    fn test_validate_page() {
        let engine = setup_engine();
        let values = HashMap::new();
        // Page 0 has required "name" field — fails.
        let err = engine.validate_page("f1", 0, &values).unwrap_err();
        assert!(matches!(err, FormError::ValidationFailed(_)));
        // Page 1 has only optional "bio" — passes.
        assert!(engine.validate_page("f1", 1, &values).is_ok());
    }

    #[test]
    fn test_submissions_for_form() {
        let mut engine = setup_engine();
        let mut v = HashMap::new();
        v.insert("name".into(), "A".into());
        engine.submit("f1", v.clone()).unwrap();
        engine.submit("f1", v).unwrap();
        assert_eq!(engine.submissions_for_form("f1").len(), 2);
    }

    #[test]
    fn test_computed_multiply() {
        let expr = ComputeExpression::Multiply(vec!["a".into(), "b".into()]);
        let mut vals = HashMap::new();
        vals.insert("a".into(), "3".into());
        vals.insert("b".into(), "7".into());
        let result = evaluate_compute(&expr, &vals).unwrap();
        assert_eq!(result, "21");
    }

    #[test]
    fn test_computed_subtract() {
        let expr = ComputeExpression::Subtract("a".into(), "b".into());
        let mut vals = HashMap::new();
        vals.insert("a".into(), "10".into());
        vals.insert("b".into(), "3".into());
        let result = evaluate_compute(&expr, &vals).unwrap();
        assert_eq!(result, "7");
    }

    #[test]
    fn test_computed_static() {
        let expr = ComputeExpression::Static("constant".into());
        let result = evaluate_compute(&expr, &HashMap::new()).unwrap();
        assert_eq!(result, "constant");
    }
}
