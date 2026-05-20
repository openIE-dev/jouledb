//! Dynamic field arrays — add, remove, reorder, validate, and constrain items.
//!
//! Replaces react-hook-form's useFieldArray, Formik FieldArray, and similar JS
//! libraries with a pure-Rust dynamic array model supporting constraints,
//! uniqueness, and bulk operations.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

// ── Errors ──────────────────────────────────────────────────────

/// Field array errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FieldArrayError {
    /// Index out of bounds.
    IndexOutOfBounds { index: usize, len: usize },
    /// Minimum count constraint violated.
    MinCount { min: usize, current: usize },
    /// Maximum count constraint reached.
    MaxCount { max: usize },
    /// Duplicate value violates unique constraint.
    DuplicateValue(String),
    /// Item validation failed.
    ValidationFailed { index: usize, message: String },
}

impl std::fmt::Display for FieldArrayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IndexOutOfBounds { index, len } => {
                write!(f, "index {index} out of bounds (len {len})")
            }
            Self::MinCount { min, current } => {
                write!(f, "minimum {min} items required, have {current}")
            }
            Self::MaxCount { max } => write!(f, "maximum {max} items reached"),
            Self::DuplicateValue(v) => write!(f, "duplicate value: {v}"),
            Self::ValidationFailed { index, message } => {
                write!(f, "item {index} validation failed: {message}")
            }
        }
    }
}

impl std::error::Error for FieldArrayError {}

// ── Constraints ─────────────────────────────────────────────────

/// Constraints for a field array.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldArrayConstraints {
    /// Minimum number of items.
    pub min_count: Option<usize>,
    /// Maximum number of items.
    pub max_count: Option<usize>,
    /// If set, the field used for uniqueness checking.
    pub unique_field: Option<String>,
}

impl Default for FieldArrayConstraints {
    fn default() -> Self {
        Self {
            min_count: None,
            max_count: None,
            unique_field: None,
        }
    }
}

impl FieldArrayConstraints {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn min(mut self, min: usize) -> Self {
        self.min_count = Some(min);
        self
    }

    pub fn max(mut self, max: usize) -> Self {
        self.max_count = Some(max);
        self
    }

    pub fn unique_on(mut self, field: &str) -> Self {
        self.unique_field = Some(field.to_string());
        self
    }
}

// ── Field Array Item ────────────────────────────────────────────

/// A single item in the field array with key-value fields.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldItem {
    pub fields: std::collections::HashMap<String, String>,
}

impl FieldItem {
    pub fn new() -> Self {
        Self {
            fields: std::collections::HashMap::new(),
        }
    }

    pub fn with_field(mut self, key: &str, value: &str) -> Self {
        self.fields.insert(key.to_string(), value.to_string());
        self
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.fields.get(key).map(|s| s.as_str())
    }

    pub fn set(&mut self, key: &str, value: &str) {
        self.fields.insert(key.to_string(), value.to_string());
    }
}

impl Default for FieldItem {
    fn default() -> Self {
        Self::new()
    }
}

// ── Field Array ─────────────────────────────────────────────────

/// A dynamic field array with constraints and validation.
#[derive(Debug, Clone)]
pub struct FieldArray {
    pub items: Vec<FieldItem>,
    pub constraints: FieldArrayConstraints,
    /// Optional per-item validator.
    pub item_validator: Option<fn(&FieldItem) -> Result<(), String>>,
}

impl FieldArray {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
            constraints: FieldArrayConstraints::default(),
            item_validator: None,
        }
    }

    pub fn with_constraints(mut self, c: FieldArrayConstraints) -> Self {
        self.constraints = c;
        self
    }

    pub fn with_validator(mut self, v: fn(&FieldItem) -> Result<(), String>) -> Self {
        self.item_validator = Some(v);
        self
    }

    /// Number of items.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Add an item to the end.
    pub fn push(&mut self, item: FieldItem) -> Result<(), FieldArrayError> {
        if let Some(max) = self.constraints.max_count {
            if self.items.len() >= max {
                return Err(FieldArrayError::MaxCount { max });
            }
        }
        self.check_unique(&item)?;
        if let Some(v) = &self.item_validator {
            v(&item).map_err(|msg| FieldArrayError::ValidationFailed {
                index: self.items.len(),
                message: msg,
            })?;
        }
        self.items.push(item);
        Ok(())
    }

    /// Insert an item at a specific index.
    pub fn insert(&mut self, index: usize, item: FieldItem) -> Result<(), FieldArrayError> {
        if index > self.items.len() {
            return Err(FieldArrayError::IndexOutOfBounds {
                index,
                len: self.items.len(),
            });
        }
        if let Some(max) = self.constraints.max_count {
            if self.items.len() >= max {
                return Err(FieldArrayError::MaxCount { max });
            }
        }
        self.check_unique(&item)?;
        self.items.insert(index, item);
        Ok(())
    }

    /// Remove an item at a specific index.
    pub fn remove(&mut self, index: usize) -> Result<FieldItem, FieldArrayError> {
        if index >= self.items.len() {
            return Err(FieldArrayError::IndexOutOfBounds {
                index,
                len: self.items.len(),
            });
        }
        if let Some(min) = self.constraints.min_count {
            if self.items.len() <= min {
                return Err(FieldArrayError::MinCount {
                    min,
                    current: self.items.len(),
                });
            }
        }
        Ok(self.items.remove(index))
    }

    /// Move an item up (swap with previous).
    pub fn move_up(&mut self, index: usize) -> Result<(), FieldArrayError> {
        if index == 0 || index >= self.items.len() {
            return Err(FieldArrayError::IndexOutOfBounds {
                index,
                len: self.items.len(),
            });
        }
        self.items.swap(index, index - 1);
        Ok(())
    }

    /// Move an item down (swap with next).
    pub fn move_down(&mut self, index: usize) -> Result<(), FieldArrayError> {
        if index >= self.items.len() - 1 {
            return Err(FieldArrayError::IndexOutOfBounds {
                index,
                len: self.items.len(),
            });
        }
        self.items.swap(index, index + 1);
        Ok(())
    }

    /// Replace an item at a specific index.
    pub fn replace(&mut self, index: usize, item: FieldItem) -> Result<(), FieldArrayError> {
        if index >= self.items.len() {
            return Err(FieldArrayError::IndexOutOfBounds {
                index,
                len: self.items.len(),
            });
        }
        // Check uniqueness against all items except the one being replaced.
        if let Some(field_name) = &self.constraints.unique_field {
            if let Some(val) = item.get(field_name) {
                for (i, existing) in self.items.iter().enumerate() {
                    if i != index {
                        if let Some(existing_val) = existing.get(field_name) {
                            if existing_val == val {
                                return Err(FieldArrayError::DuplicateValue(val.to_string()));
                            }
                        }
                    }
                }
            }
        }
        self.items[index] = item;
        Ok(())
    }

    /// Bulk append multiple items.
    pub fn extend(&mut self, items: Vec<FieldItem>) -> Result<(), FieldArrayError> {
        for item in items {
            self.push(item)?;
        }
        Ok(())
    }

    /// Clear all items (respects min_count — returns error if min > 0).
    pub fn clear(&mut self) -> Result<(), FieldArrayError> {
        if let Some(min) = self.constraints.min_count {
            if min > 0 {
                return Err(FieldArrayError::MinCount {
                    min,
                    current: self.items.len(),
                });
            }
        }
        self.items.clear();
        Ok(())
    }

    /// Validate all items. Returns a list of errors.
    pub fn validate_all(&self) -> Vec<FieldArrayError> {
        let mut errors = Vec::new();

        if let Some(min) = self.constraints.min_count {
            if self.items.len() < min {
                errors.push(FieldArrayError::MinCount {
                    min,
                    current: self.items.len(),
                });
            }
        }

        if let Some(v) = &self.item_validator {
            for (i, item) in self.items.iter().enumerate() {
                if let Err(msg) = v(item) {
                    errors.push(FieldArrayError::ValidationFailed {
                        index: i,
                        message: msg,
                    });
                }
            }
        }

        // Check uniqueness across all items.
        if let Some(field_name) = &self.constraints.unique_field {
            let mut seen = HashSet::new();
            for (i, item) in self.items.iter().enumerate() {
                if let Some(val) = item.get(field_name) {
                    if !seen.insert(val.to_string()) {
                        errors.push(FieldArrayError::DuplicateValue(format!(
                            "index {i}: {val}"
                        )));
                    }
                }
            }
        }

        errors
    }

    /// Get an item by index.
    pub fn get(&self, index: usize) -> Option<&FieldItem> {
        self.items.get(index)
    }

    fn check_unique(&self, item: &FieldItem) -> Result<(), FieldArrayError> {
        if let Some(field_name) = &self.constraints.unique_field {
            if let Some(val) = item.get(field_name) {
                for existing in &self.items {
                    if let Some(existing_val) = existing.get(field_name) {
                        if existing_val == val {
                            return Err(FieldArrayError::DuplicateValue(val.to_string()));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl Default for FieldArray {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn item(name: &str) -> FieldItem {
        FieldItem::new().with_field("name", name)
    }

    #[test]
    fn push_and_len() {
        let mut arr = FieldArray::new();
        arr.push(item("Alice")).unwrap();
        arr.push(item("Bob")).unwrap();
        assert_eq!(arr.len(), 2);
    }

    #[test]
    fn max_count_enforced() {
        let mut arr =
            FieldArray::new().with_constraints(FieldArrayConstraints::new().max(2));
        arr.push(item("A")).unwrap();
        arr.push(item("B")).unwrap();
        assert!(arr.push(item("C")).is_err());
    }

    #[test]
    fn min_count_on_remove() {
        let mut arr =
            FieldArray::new().with_constraints(FieldArrayConstraints::new().min(1));
        arr.push(item("A")).unwrap();
        assert!(arr.remove(0).is_err()); // would go below min
    }

    #[test]
    fn insert_at_index() {
        let mut arr = FieldArray::new();
        arr.push(item("A")).unwrap();
        arr.push(item("C")).unwrap();
        arr.insert(1, item("B")).unwrap();
        assert_eq!(arr.get(1).unwrap().get("name"), Some("B"));
    }

    #[test]
    fn insert_out_of_bounds() {
        let mut arr = FieldArray::new();
        assert!(arr.insert(5, item("X")).is_err());
    }

    #[test]
    fn move_up_down() {
        let mut arr = FieldArray::new();
        arr.push(item("A")).unwrap();
        arr.push(item("B")).unwrap();
        arr.push(item("C")).unwrap();

        arr.move_down(0).unwrap();
        assert_eq!(arr.get(0).unwrap().get("name"), Some("B"));
        assert_eq!(arr.get(1).unwrap().get("name"), Some("A"));

        arr.move_up(1).unwrap();
        assert_eq!(arr.get(0).unwrap().get("name"), Some("A"));
    }

    #[test]
    fn move_up_first_errors() {
        let mut arr = FieldArray::new();
        arr.push(item("A")).unwrap();
        assert!(arr.move_up(0).is_err());
    }

    #[test]
    fn unique_constraint() {
        let mut arr = FieldArray::new()
            .with_constraints(FieldArrayConstraints::new().unique_on("name"));
        arr.push(item("Alice")).unwrap();
        assert!(arr.push(item("Alice")).is_err());
        arr.push(item("Bob")).unwrap();
    }

    #[test]
    fn replace_item() {
        let mut arr = FieldArray::new();
        arr.push(item("A")).unwrap();
        arr.replace(0, item("B")).unwrap();
        assert_eq!(arr.get(0).unwrap().get("name"), Some("B"));
    }

    #[test]
    fn bulk_extend() {
        let mut arr = FieldArray::new();
        arr.extend(vec![item("A"), item("B"), item("C")]).unwrap();
        assert_eq!(arr.len(), 3);
    }

    #[test]
    fn item_validator() {
        let mut arr = FieldArray::new().with_validator(|item| {
            let name = item.get("name").unwrap_or("");
            if name.is_empty() {
                Err("name is required".into())
            } else {
                Ok(())
            }
        });
        assert!(arr.push(FieldItem::new()).is_err());
        assert!(arr.push(item("Alice")).is_ok());
    }

    #[test]
    fn validate_all() {
        let mut arr = FieldArray::new()
            .with_constraints(FieldArrayConstraints::new().min(3))
            .with_validator(|item| {
                if item.get("name").unwrap_or("").is_empty() {
                    Err("empty name".into())
                } else {
                    Ok(())
                }
            });
        arr.items.push(item("A")); // bypass push validation for testing
        let errors = arr.validate_all();
        assert!(!errors.is_empty()); // min_count not met
    }

    #[test]
    fn clear_respects_min() {
        let mut arr =
            FieldArray::new().with_constraints(FieldArrayConstraints::new().min(1));
        arr.push(item("A")).unwrap();
        assert!(arr.clear().is_err());
    }

    #[test]
    fn clear_no_min() {
        let mut arr = FieldArray::new();
        arr.push(item("A")).unwrap();
        arr.clear().unwrap();
        assert!(arr.is_empty());
    }

    #[test]
    fn remove_returns_item() {
        let mut arr = FieldArray::new();
        arr.push(item("Alice")).unwrap();
        let removed = arr.remove(0).unwrap();
        assert_eq!(removed.get("name"), Some("Alice"));
        assert!(arr.is_empty());
    }
}
