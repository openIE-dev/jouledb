//! User profile management — profile fields, avatar handling, preferences,
//! profile completeness scoring, field validation, and profile versioning.
//!
//! Replaces gravatar, profile-builder, and similar JS/TS user profile
//! libraries with a pure-Rust profile lifecycle engine.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

// ── Errors ─────────────────────────────────────────────────────

/// Profile engine errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    /// User not found.
    UserNotFound(String),
    /// Duplicate user.
    DuplicateUser(String),
    /// Invalid field value.
    InvalidField { field: String, reason: String },
    /// Field is required but missing.
    RequiredField(String),
    /// Version conflict during update.
    VersionConflict { expected: u64, actual: u64 },
    /// Avatar too large.
    AvatarTooLarge { max_bytes: usize, actual_bytes: usize },
    /// Invalid preference key.
    InvalidPreference(String),
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UserNotFound(id) => write!(f, "user not found: {id}"),
            Self::DuplicateUser(id) => write!(f, "duplicate user: {id}"),
            Self::InvalidField { field, reason } => {
                write!(f, "invalid field {field}: {reason}")
            }
            Self::RequiredField(field) => write!(f, "required field missing: {field}"),
            Self::VersionConflict { expected, actual } => {
                write!(f, "version conflict: expected {expected}, got {actual}")
            }
            Self::AvatarTooLarge {
                max_bytes,
                actual_bytes,
            } => write!(f, "avatar too large: {actual_bytes} > {max_bytes}"),
            Self::InvalidPreference(key) => write!(f, "invalid preference: {key}"),
        }
    }
}

impl std::error::Error for ProfileError {}

// ── Types ──────────────────────────────────────────────────────

/// Avatar data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Avatar {
    /// URL to an external avatar image.
    Url(String),
    /// Inline image data (format, bytes).
    Inline { mime_type: String, data: Vec<u8> },
    /// Generated from initials.
    Initials(String),
    /// No avatar set.
    None,
}

/// Supported profile field types.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldValue {
    Text(String),
    Number(i64),
    Bool(bool),
    Date(String),
    Url(String),
    List(Vec<String>),
}

impl fmt::Display for FieldValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(s) => write!(f, "{s}"),
            Self::Number(n) => write!(f, "{n}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Date(d) => write!(f, "{d}"),
            Self::Url(u) => write!(f, "{u}"),
            Self::List(items) => write!(f, "{}", items.join(", ")),
        }
    }
}

/// A user preference.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Preference {
    pub key: String,
    pub value: String,
    pub category: String,
}

/// A snapshot of a profile at a particular version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfileVersion {
    pub version: u64,
    pub timestamp_secs: u64,
    pub fields: HashMap<String, FieldValue>,
    pub change_summary: String,
}

/// A user profile.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserProfile {
    pub user_id: String,
    pub display_name: String,
    pub email: Option<String>,
    pub avatar: Avatar,
    pub bio: Option<String>,
    pub fields: HashMap<String, FieldValue>,
    pub preferences: HashMap<String, Preference>,
    pub version: u64,
    pub created_at_secs: u64,
    pub updated_at_secs: u64,
}

impl UserProfile {
    pub fn new(user_id: impl Into<String>, display_name: impl Into<String>, now_secs: u64) -> Self {
        Self {
            user_id: user_id.into(),
            display_name: display_name.into(),
            email: None,
            avatar: Avatar::None,
            bio: None,
            fields: HashMap::new(),
            preferences: HashMap::new(),
            version: 1,
            created_at_secs: now_secs,
            updated_at_secs: now_secs,
        }
    }
}

/// Field definition for schema-driven validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldDefinition {
    pub name: String,
    pub required: bool,
    pub max_length: Option<usize>,
    pub min_length: Option<usize>,
    pub pattern: Option<String>,
    pub weight: f64,
}

/// Profile completeness score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessScore {
    pub score: f64,
    pub max_score: f64,
    pub percentage: f64,
    pub missing_fields: Vec<String>,
    pub completed_fields: Vec<String>,
}

// ── Validation ─────────────────────────────────────────────────

/// Validate a field value against a definition.
fn validate_field(
    def: &FieldDefinition,
    value: &FieldValue,
) -> Result<(), ProfileError> {
    match value {
        FieldValue::Text(s) | FieldValue::Url(s) | FieldValue::Date(s) => {
            if let Some(max) = def.max_length {
                if s.len() > max {
                    return Err(ProfileError::InvalidField {
                        field: def.name.clone(),
                        reason: format!("exceeds max length {max}"),
                    });
                }
            }
            if let Some(min) = def.min_length {
                if s.len() < min {
                    return Err(ProfileError::InvalidField {
                        field: def.name.clone(),
                        reason: format!("below min length {min}"),
                    });
                }
            }
        }
        FieldValue::List(items) => {
            if let Some(max) = def.max_length {
                if items.len() > max {
                    return Err(ProfileError::InvalidField {
                        field: def.name.clone(),
                        reason: format!("list exceeds max items {max}"),
                    });
                }
            }
        }
        _ => {}
    }
    Ok(())
}

// ── Engine ─────────────────────────────────────────────────────

/// Configuration for the profile engine.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    pub max_avatar_bytes: usize,
    pub field_definitions: Vec<FieldDefinition>,
    pub max_versions: usize,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            max_avatar_bytes: 1024 * 1024, // 1 MB
            field_definitions: Vec::new(),
            max_versions: 50,
        }
    }
}

/// The profile engine.
#[derive(Debug, Clone)]
pub struct ProfileEngine {
    config: ProfileConfig,
    profiles: HashMap<String, UserProfile>,
    versions: HashMap<String, Vec<ProfileVersion>>,
}

impl ProfileEngine {
    pub fn new(config: ProfileConfig) -> Self {
        Self {
            config,
            profiles: HashMap::new(),
            versions: HashMap::new(),
        }
    }

    /// Create a new user profile.
    pub fn create_profile(
        &mut self,
        user_id: &str,
        display_name: &str,
        now_secs: u64,
    ) -> Result<&UserProfile, ProfileError> {
        if self.profiles.contains_key(user_id) {
            return Err(ProfileError::DuplicateUser(user_id.to_string()));
        }

        let profile = UserProfile::new(user_id, display_name, now_secs);
        self.profiles.insert(user_id.to_string(), profile);

        // Store initial version.
        let version = ProfileVersion {
            version: 1,
            timestamp_secs: now_secs,
            fields: HashMap::new(),
            change_summary: "profile created".to_string(),
        };
        self.versions
            .insert(user_id.to_string(), vec![version]);

        Ok(self.profiles.get(user_id).unwrap())
    }

    /// Get a profile by user ID.
    pub fn get_profile(&self, user_id: &str) -> Result<&UserProfile, ProfileError> {
        self.profiles
            .get(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))
    }

    /// Update profile fields. Uses optimistic concurrency via version check.
    pub fn update_fields(
        &mut self,
        user_id: &str,
        fields: HashMap<String, FieldValue>,
        expected_version: u64,
        now_secs: u64,
        summary: &str,
    ) -> Result<&UserProfile, ProfileError> {
        let profile = self
            .profiles
            .get_mut(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;

        if profile.version != expected_version {
            return Err(ProfileError::VersionConflict {
                expected: expected_version,
                actual: profile.version,
            });
        }

        // Validate fields against definitions.
        for (name, value) in &fields {
            if let Some(def) = self.config.field_definitions.iter().find(|d| d.name == *name) {
                validate_field(def, value)?;
            }
        }

        for (key, value) in &fields {
            profile.fields.insert(key.clone(), value.clone());
        }

        profile.version += 1;
        profile.updated_at_secs = now_secs;

        // Store version snapshot.
        let version_entry = ProfileVersion {
            version: profile.version,
            timestamp_secs: now_secs,
            fields: profile.fields.clone(),
            change_summary: summary.to_string(),
        };

        let history = self.versions.entry(user_id.to_string()).or_default();
        history.push(version_entry);

        // Trim old versions.
        if history.len() > self.config.max_versions {
            let excess = history.len() - self.config.max_versions;
            history.drain(..excess);
        }

        Ok(self.profiles.get(user_id).unwrap())
    }

    /// Update display name.
    pub fn set_display_name(
        &mut self,
        user_id: &str,
        name: &str,
        now_secs: u64,
    ) -> Result<(), ProfileError> {
        let profile = self
            .profiles
            .get_mut(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;
        profile.display_name = name.to_string();
        profile.updated_at_secs = now_secs;
        profile.version += 1;
        Ok(())
    }

    /// Set avatar.
    pub fn set_avatar(
        &mut self,
        user_id: &str,
        avatar: Avatar,
        now_secs: u64,
    ) -> Result<(), ProfileError> {
        // Check size for inline avatars.
        if let Avatar::Inline { data, .. } = &avatar {
            if data.len() > self.config.max_avatar_bytes {
                return Err(ProfileError::AvatarTooLarge {
                    max_bytes: self.config.max_avatar_bytes,
                    actual_bytes: data.len(),
                });
            }
        }

        let profile = self
            .profiles
            .get_mut(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;
        profile.avatar = avatar;
        profile.updated_at_secs = now_secs;
        profile.version += 1;
        Ok(())
    }

    /// Set email.
    pub fn set_email(
        &mut self,
        user_id: &str,
        email: &str,
        now_secs: u64,
    ) -> Result<(), ProfileError> {
        if !email.contains('@') || email.len() < 3 {
            return Err(ProfileError::InvalidField {
                field: "email".to_string(),
                reason: "invalid email format".to_string(),
            });
        }
        let profile = self
            .profiles
            .get_mut(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;
        profile.email = Some(email.to_string());
        profile.updated_at_secs = now_secs;
        profile.version += 1;
        Ok(())
    }

    /// Set bio.
    pub fn set_bio(
        &mut self,
        user_id: &str,
        bio: &str,
        now_secs: u64,
    ) -> Result<(), ProfileError> {
        let profile = self
            .profiles
            .get_mut(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;
        profile.bio = Some(bio.to_string());
        profile.updated_at_secs = now_secs;
        profile.version += 1;
        Ok(())
    }

    /// Set a preference.
    pub fn set_preference(
        &mut self,
        user_id: &str,
        key: &str,
        value: &str,
        category: &str,
    ) -> Result<(), ProfileError> {
        if key.is_empty() {
            return Err(ProfileError::InvalidPreference(key.to_string()));
        }
        let profile = self
            .profiles
            .get_mut(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;
        profile.preferences.insert(
            key.to_string(),
            Preference {
                key: key.to_string(),
                value: value.to_string(),
                category: category.to_string(),
            },
        );
        Ok(())
    }

    /// Get a preference.
    pub fn get_preference(
        &self,
        user_id: &str,
        key: &str,
    ) -> Result<Option<&Preference>, ProfileError> {
        let profile = self
            .profiles
            .get(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;
        Ok(profile.preferences.get(key))
    }

    /// Calculate profile completeness based on field definitions.
    pub fn completeness(&self, user_id: &str) -> Result<CompletenessScore, ProfileError> {
        let profile = self
            .profiles
            .get(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;

        let mut total_weight = 0.0;
        let mut achieved = 0.0;
        let mut missing = Vec::new();
        let mut completed = Vec::new();

        // Built-in fields.
        let builtin_weight = 1.0;

        // display_name (always present after creation).
        total_weight += builtin_weight;
        if !profile.display_name.is_empty() {
            achieved += builtin_weight;
            completed.push("display_name".to_string());
        } else {
            missing.push("display_name".to_string());
        }

        // email
        total_weight += builtin_weight;
        if profile.email.is_some() {
            achieved += builtin_weight;
            completed.push("email".to_string());
        } else {
            missing.push("email".to_string());
        }

        // avatar
        total_weight += builtin_weight;
        if profile.avatar != Avatar::None {
            achieved += builtin_weight;
            completed.push("avatar".to_string());
        } else {
            missing.push("avatar".to_string());
        }

        // bio
        total_weight += builtin_weight;
        if profile.bio.is_some() {
            achieved += builtin_weight;
            completed.push("bio".to_string());
        } else {
            missing.push("bio".to_string());
        }

        // Custom field definitions.
        for def in &self.config.field_definitions {
            total_weight += def.weight;
            if profile.fields.contains_key(&def.name) {
                achieved += def.weight;
                completed.push(def.name.clone());
            } else {
                missing.push(def.name.clone());
            }
        }

        let percentage = if total_weight > 0.0 {
            (achieved / total_weight) * 100.0
        } else {
            100.0
        };

        Ok(CompletenessScore {
            score: achieved,
            max_score: total_weight,
            percentage,
            missing_fields: missing,
            completed_fields: completed,
        })
    }

    /// Get version history for a profile.
    pub fn version_history(&self, user_id: &str) -> Result<&[ProfileVersion], ProfileError> {
        self.versions
            .get(user_id)
            .map(|v| v.as_slice())
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))
    }

    /// Get a specific version.
    pub fn get_version(
        &self,
        user_id: &str,
        version: u64,
    ) -> Result<Option<&ProfileVersion>, ProfileError> {
        let history = self
            .versions
            .get(user_id)
            .ok_or_else(|| ProfileError::UserNotFound(user_id.to_string()))?;
        Ok(history.iter().find(|v| v.version == version))
    }

    /// Total profiles.
    pub fn count(&self) -> usize {
        self.profiles.len()
    }

    /// Delete a profile.
    pub fn delete_profile(&mut self, user_id: &str) -> Result<(), ProfileError> {
        if self.profiles.remove(user_id).is_none() {
            return Err(ProfileError::UserNotFound(user_id.to_string()));
        }
        self.versions.remove(user_id);
        Ok(())
    }
}

impl Default for ProfileEngine {
    fn default() -> Self {
        Self::new(ProfileConfig::default())
    }
}

// ── Tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_fields() -> ProfileConfig {
        ProfileConfig {
            max_avatar_bytes: 1024,
            max_versions: 5,
            field_definitions: vec![
                FieldDefinition {
                    name: "location".to_string(),
                    required: false,
                    max_length: Some(100),
                    min_length: None,
                    pattern: None,
                    weight: 2.0,
                },
                FieldDefinition {
                    name: "website".to_string(),
                    required: false,
                    max_length: Some(200),
                    min_length: Some(5),
                    pattern: None,
                    weight: 1.0,
                },
            ],
        }
    }

    #[test]
    fn test_create_profile() {
        let mut engine = ProfileEngine::default();
        let profile = engine.create_profile("u1", "Alice", 1000).unwrap();
        assert_eq!(profile.user_id, "u1");
        assert_eq!(profile.display_name, "Alice");
        assert_eq!(profile.version, 1);
    }

    #[test]
    fn test_duplicate_profile() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        let err = engine.create_profile("u1", "Bob", 1001).unwrap_err();
        assert!(matches!(err, ProfileError::DuplicateUser(_)));
    }

    #[test]
    fn test_get_profile() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        let profile = engine.get_profile("u1").unwrap();
        assert_eq!(profile.display_name, "Alice");
    }

    #[test]
    fn test_get_profile_not_found() {
        let engine = ProfileEngine::default();
        let err = engine.get_profile("ghost").unwrap_err();
        assert!(matches!(err, ProfileError::UserNotFound(_)));
    }

    #[test]
    fn test_update_fields() {
        let mut engine = ProfileEngine::new(config_with_fields());
        engine.create_profile("u1", "Alice", 1000).unwrap();

        let mut fields = HashMap::new();
        fields.insert(
            "location".to_string(),
            FieldValue::Text("NYC".to_string()),
        );

        let profile = engine
            .update_fields("u1", fields, 1, 2000, "added location")
            .unwrap();
        assert_eq!(profile.version, 2);
        assert!(profile.fields.contains_key("location"));
    }

    #[test]
    fn test_version_conflict() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();

        let err = engine
            .update_fields("u1", HashMap::new(), 99, 2000, "test")
            .unwrap_err();
        assert!(matches!(err, ProfileError::VersionConflict { .. }));
    }

    #[test]
    fn test_field_validation_max_length() {
        let mut engine = ProfileEngine::new(config_with_fields());
        engine.create_profile("u1", "Alice", 1000).unwrap();

        let mut fields = HashMap::new();
        fields.insert(
            "location".to_string(),
            FieldValue::Text("x".repeat(200)),
        );

        let err = engine
            .update_fields("u1", fields, 1, 2000, "test")
            .unwrap_err();
        assert!(matches!(err, ProfileError::InvalidField { .. }));
    }

    #[test]
    fn test_field_validation_min_length() {
        let mut engine = ProfileEngine::new(config_with_fields());
        engine.create_profile("u1", "Alice", 1000).unwrap();

        let mut fields = HashMap::new();
        fields.insert(
            "website".to_string(),
            FieldValue::Text("ab".to_string()),
        );

        let err = engine
            .update_fields("u1", fields, 1, 2000, "test")
            .unwrap_err();
        assert!(matches!(err, ProfileError::InvalidField { .. }));
    }

    #[test]
    fn test_set_display_name() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine.set_display_name("u1", "Alice Smith", 2000).unwrap();
        let profile = engine.get_profile("u1").unwrap();
        assert_eq!(profile.display_name, "Alice Smith");
    }

    #[test]
    fn test_set_avatar_url() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine
            .set_avatar(
                "u1",
                Avatar::Url("https://example.com/avatar.png".to_string()),
                2000,
            )
            .unwrap();
        let profile = engine.get_profile("u1").unwrap();
        assert!(matches!(profile.avatar, Avatar::Url(_)));
    }

    #[test]
    fn test_set_avatar_too_large() {
        let mut engine = ProfileEngine::new(ProfileConfig {
            max_avatar_bytes: 10,
            ..Default::default()
        });
        engine.create_profile("u1", "Alice", 1000).unwrap();

        let err = engine
            .set_avatar(
                "u1",
                Avatar::Inline {
                    mime_type: "image/png".to_string(),
                    data: vec![0u8; 100],
                },
                2000,
            )
            .unwrap_err();
        assert!(matches!(err, ProfileError::AvatarTooLarge { .. }));
    }

    #[test]
    fn test_set_email() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine.set_email("u1", "alice@example.com", 2000).unwrap();
        let profile = engine.get_profile("u1").unwrap();
        assert_eq!(profile.email.as_deref(), Some("alice@example.com"));
    }

    #[test]
    fn test_set_invalid_email() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        let err = engine.set_email("u1", "bad", 2000).unwrap_err();
        assert!(matches!(err, ProfileError::InvalidField { .. }));
    }

    #[test]
    fn test_set_bio() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine.set_bio("u1", "Hello world!", 2000).unwrap();
        let profile = engine.get_profile("u1").unwrap();
        assert_eq!(profile.bio.as_deref(), Some("Hello world!"));
    }

    #[test]
    fn test_preferences() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine
            .set_preference("u1", "theme", "dark", "appearance")
            .unwrap();

        let pref = engine.get_preference("u1", "theme").unwrap().unwrap();
        assert_eq!(pref.value, "dark");
        assert_eq!(pref.category, "appearance");
    }

    #[test]
    fn test_preference_invalid_key() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        let err = engine
            .set_preference("u1", "", "value", "cat")
            .unwrap_err();
        assert!(matches!(err, ProfileError::InvalidPreference(_)));
    }

    #[test]
    fn test_completeness_minimal() {
        let mut engine = ProfileEngine::new(config_with_fields());
        engine.create_profile("u1", "Alice", 1000).unwrap();

        let score = engine.completeness("u1").unwrap();
        // Only display_name is filled (weight 1.0), email/avatar/bio/location/website missing.
        // Total weight: 4 * 1.0 (builtins) + 2.0 + 1.0 = 7.0
        // Achieved: 1.0 (display_name)
        assert!((score.max_score - 7.0).abs() < f64::EPSILON);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
        assert!(score.missing_fields.contains(&"email".to_string()));
        assert!(score.completed_fields.contains(&"display_name".to_string()));
    }

    #[test]
    fn test_completeness_full() {
        let mut engine = ProfileEngine::new(config_with_fields());
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine.set_email("u1", "a@b.com", 1001).unwrap();
        engine
            .set_avatar("u1", Avatar::Initials("A".to_string()), 1002)
            .unwrap();
        engine.set_bio("u1", "Hi!", 1003).unwrap();

        let mut fields = HashMap::new();
        fields.insert(
            "location".to_string(),
            FieldValue::Text("NYC".to_string()),
        );
        fields.insert(
            "website".to_string(),
            FieldValue::Url("https://example.com".to_string()),
        );
        engine
            .update_fields("u1", fields, engine.get_profile("u1").unwrap().version, 1004, "add fields")
            .unwrap();

        let score = engine.completeness("u1").unwrap();
        assert!((score.percentage - 100.0).abs() < f64::EPSILON);
        assert!(score.missing_fields.is_empty());
    }

    #[test]
    fn test_version_history() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine
            .update_fields("u1", HashMap::new(), 1, 2000, "update 1")
            .unwrap();

        let history = engine.version_history("u1").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].version, 1);
        assert_eq!(history[1].version, 2);
    }

    #[test]
    fn test_version_trimming() {
        let mut engine = ProfileEngine::new(ProfileConfig {
            max_versions: 3,
            ..Default::default()
        });
        engine.create_profile("u1", "Alice", 1000).unwrap();

        for i in 1..=5 {
            let v = engine.get_profile("u1").unwrap().version;
            engine
                .update_fields("u1", HashMap::new(), v, 1000 + i * 100, &format!("v{}", i + 1))
                .unwrap();
        }

        let history = engine.version_history("u1").unwrap();
        assert!(history.len() <= 3);
    }

    #[test]
    fn test_get_specific_version() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine
            .update_fields("u1", HashMap::new(), 1, 2000, "v2")
            .unwrap();

        let v1 = engine.get_version("u1", 1).unwrap().unwrap();
        assert_eq!(v1.change_summary, "profile created");

        let v2 = engine.get_version("u1", 2).unwrap().unwrap();
        assert_eq!(v2.change_summary, "v2");

        assert!(engine.get_version("u1", 99).unwrap().is_none());
    }

    #[test]
    fn test_delete_profile() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine.delete_profile("u1").unwrap();
        assert_eq!(engine.count(), 0);
        assert!(engine.get_profile("u1").is_err());
    }

    #[test]
    fn test_delete_nonexistent() {
        let mut engine = ProfileEngine::default();
        let err = engine.delete_profile("ghost").unwrap_err();
        assert!(matches!(err, ProfileError::UserNotFound(_)));
    }

    #[test]
    fn test_field_value_display() {
        assert_eq!(FieldValue::Text("hi".to_string()).to_string(), "hi");
        assert_eq!(FieldValue::Number(42).to_string(), "42");
        assert_eq!(FieldValue::Bool(true).to_string(), "true");
        assert_eq!(
            FieldValue::List(vec!["a".to_string(), "b".to_string()]).to_string(),
            "a, b"
        );
    }

    #[test]
    fn test_error_display() {
        let e = ProfileError::VersionConflict {
            expected: 1,
            actual: 2,
        };
        assert_eq!(e.to_string(), "version conflict: expected 1, got 2");
    }

    #[test]
    fn test_avatar_initials() {
        let mut engine = ProfileEngine::default();
        engine.create_profile("u1", "Alice", 1000).unwrap();
        engine
            .set_avatar("u1", Avatar::Initials("AS".to_string()), 2000)
            .unwrap();
        let profile = engine.get_profile("u1").unwrap();
        assert_eq!(profile.avatar, Avatar::Initials("AS".to_string()));
    }

    #[test]
    fn test_default_engine() {
        let engine = ProfileEngine::default();
        assert_eq!(engine.count(), 0);
    }

    #[test]
    fn test_list_field_validation() {
        let config = ProfileConfig {
            field_definitions: vec![FieldDefinition {
                name: "tags".to_string(),
                required: false,
                max_length: Some(3),
                min_length: None,
                pattern: None,
                weight: 1.0,
            }],
            ..Default::default()
        };
        let mut engine = ProfileEngine::new(config);
        engine.create_profile("u1", "Alice", 1000).unwrap();

        let mut fields = HashMap::new();
        fields.insert(
            "tags".to_string(),
            FieldValue::List(vec!["a".into(), "b".into(), "c".into(), "d".into()]),
        );

        let err = engine
            .update_fields("u1", fields, 1, 2000, "test")
            .unwrap_err();
        assert!(matches!(err, ProfileError::InvalidField { .. }));
    }
}
